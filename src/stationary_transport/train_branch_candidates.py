#!/usr/bin/env python3
"""Physics-only automatic branch selection for S1--S4 transport cases."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stationary_transport.cases import get_case
from stationary_transport.flexible_branches import FlexibleTransportCP
from stationary_transport.train import (
    inflow_loss, physical_intensity, quadrature, reference_metrics, sample_x,
    transport_residual, transport_residual_separable,
)


FAMILIES = (
    "fixed_physics", "fourier", "mlp", "physics_correction",
    "modified_mlp", "physics_modified", "physics_physics",
    "physics_mlp", "mlp_physics", "mlp_mlp",
    "homogeneous_uniform_correction", "homogeneous_routed_correction",
    "permuted_physics_correction",
    "collision_uniform_correction", "collision_routed_correction",
    "collision_width_uniform_correction", "collision_width_routed_correction",
)
DEFAULT_FAMILIES = ("fixed_physics", "fourier", "mlp", "physics_correction")


def initialization_family(family: str) -> str:
    """Return the canonical initialization stream for paired controls."""
    if family in {
        "permuted_physics_correction",
        "homogeneous_uniform_correction",
        "homogeneous_routed_correction",
        "collision_uniform_correction", "collision_routed_correction",
        "collision_width_uniform_correction", "collision_width_routed_correction",
    }:
        return "physics_correction"
    return family


def objective(case, model, x, mu, weight, kernel, boundary_weight):
    pde, _, _ = transport_residual(case, model, x, mu, weight, kernel)
    boundary = inflow_loss(case, model, mu)
    return pde + boundary_weight * boundary, pde, boundary


def dsa_moment_indicator(case, model, mu, weight, kernel, grid_size: int):
    """Physics-only scalar-flux error indicator for branch selection.

    Let ``R`` be the angular transport residual.  Its zeroth and first
    angular moments satisfy the low-order balance equations.  Eliminating the
    current with the P1 relation gives the DSA error equation

        -d/dx (D d e_phi/dx) + sigma_a e_phi
          = R_0 - d/dx (R_1/sigma_t),   D = 1/(3 sigma_t).

    The norm of ``e_phi`` estimates the scalar-flux error caused by the
    candidate residual.  It uses only the PDE coefficients and residual; the
    evaluation reference is deliberately absent.  Harmonic face diffusion
    is used at material interfaces so both adjacent cells share one current.
    """
    if grid_size < 16:
        raise ValueError("DSA grid must contain at least 16 cells")
    device = mu.device
    dtype = mu.dtype
    dx = 1.0 / float(grid_size)
    x = ((torch.arange(grid_size, device=device, dtype=dtype) + 0.5)
         * dx)
    with torch.enable_grad():
        _, residual, psi = transport_residual_separable(
            case, model, x, mu, weight, kernel,
        )
    residual = residual.detach()
    psi = psi.detach()
    scattering, absorption = case.coefficients_torch(x)
    total = (scattering + absorption).clamp_min(1.0e-6)

    # Unnormalised angular moments match phi = integral psi dmu used by the
    # reference metric.  The first-moment divergence is essential in a
    # heterogeneous slab; using R0 alone confounds current and absorption.
    r0 = (residual * weight[None, :]).sum(dim=1)
    r1 = (residual * mu[None, :] * weight[None, :]).sum(dim=1)
    scaled_r1 = r1 / total
    derivative_r1 = torch.empty_like(scaled_r1)
    derivative_r1[1:-1] = (
        scaled_r1[2:] - scaled_r1[:-2]
    ) / (2.0 * dx)
    derivative_r1[0] = (scaled_r1[1] - scaled_r1[0]) / dx
    derivative_r1[-1] = (scaled_r1[-1] - scaled_r1[-2]) / dx
    source = r0 - derivative_r1

    diffusion = 1.0 / (3.0 * total)
    face = (2.0 * diffusion[:-1] * diffusion[1:]
            / (diffusion[:-1] + diffusion[1:]).clamp_min(1.0e-12))
    matrix = torch.zeros(
        (grid_size, grid_size), device=device, dtype=dtype,
    )
    inverse_dx2 = 1.0 / (dx * dx)
    matrix.diagonal().copy_(absorption)
    matrix.diagonal()[:-1].add_(face * inverse_dx2)
    matrix.diagonal()[1:].add_(face * inverse_dx2)
    matrix.diagonal(offset=1).copy_(-face * inverse_dx2)
    matrix.diagonal(offset=-1).copy_(-face * inverse_dx2)
    # Vacuum-error closure at the two exterior faces.  The factor two is the
    # cell-centred distance from the boundary face to the first/last centre.
    matrix[0, 0] += 2.0 * diffusion[0] * inverse_dx2
    matrix[-1, -1] += 2.0 * diffusion[-1] * inverse_dx2
    correction = torch.linalg.solve(matrix, source[:, None]).squeeze(1)

    scalar_flux = psi @ weight
    flux_rms = scalar_flux.square().mean().sqrt().clamp_min(1.0e-8)
    indicator = correction.square().mean().sqrt() / flux_rms
    r0_indicator = (
        (r0 / absorption.clamp_min(1.0e-4)).square().mean().sqrt()
        / flux_rms
    )
    r1_indicator = (
        scaled_r1.square().mean().sqrt() / flux_rms
    )
    region_values = []
    edges = (0.0, *case.material_edges, 1.0)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (x >= lower) & (x < upper)
        if upper == 1.0:
            mask = (x >= lower) & (x <= upper)
        region_values.append(float(
            correction[mask].square().mean().sqrt() / flux_rms
        ))
    return {
        "dsa_moment_indicator": float(indicator),
        "zeroth_moment_indicator": float(r0_indicator),
        "first_moment_indicator": float(r1_indicator),
        "dsa_region_indicators": region_values,
    }


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    case = get_case(args.case, args.task_parameter)
    reference = (args.reference if args.reference is not None else
                 ROOT / "results/iclr2027/references/stationary_transport" /
                 case.name / "reference.npz")
    mu, weight, kernel = quadrature(case, args.quadrature_order, device)
    validation = sample_x(
        case, args.validation_x, device, args.interface_fraction,
        args.interface_width, seed=2026080201,
    )
    output = (args.output / case.name / args.variant / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    if history_path.exists() and not args.overwrite:
        raise FileExistsError(output)
    history_path.write_text("")

    candidates = {}
    for family in args.families:
        # Keep each family initialization identical between the automatic and
        # fixed-family ablations, even when only one family is instantiated.
        seed_family = initialization_family(family)
        index = FAMILIES.index(seed_family)
        torch.manual_seed(args.seed + 1009 * (index + 1))
        model = FlexibleTransportCP(
            case, family, rank=args.cp_rank, width=args.width, depth=args.depth,
            physics_modes=args.physics_modes,
        ).to(device)
        candidates[family] = {
            "model": model,
            "optimizer": torch.optim.Adam(model.parameters(), lr=args.probe_lr),
            "scheduler": None,
        }
        candidates[family]["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
            candidates[family]["optimizer"], T_max=max(1, args.probe_steps),
            eta_min=args.probe_lr*args.scheduler_min_factor,
        )
    started = time.perf_counter()
    probe_started = started
    for step in range(1, args.probe_steps + 1):
        torch.manual_seed(args.seed + 20000 + step)
        x = sample_x(
            case, args.batch_x, device, args.interface_fraction,
            args.interface_width,
        )
        for candidate in candidates.values():
            model, optimizer = candidate["model"], candidate["optimizer"]
            loss, _, _ = objective(
                case, model, x, mu, weight, kernel, args.boundary_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            candidate["scheduler"].step()

    probe_seconds = time.perf_counter() - probe_started
    probe = []
    parameter_counts = sorted(
        item["model"].trainable_count for item in candidates.values()
    )
    parameter_median = parameter_counts[len(parameter_counts) // 2]
    for family, candidate in candidates.items():
        model = candidate["model"]
        model.eval()
        physics = float(objective(
            case, model, validation, mu, weight, kernel,
            args.boundary_weight,
        )[0].detach())
        selector_diagnostics = (
            dsa_moment_indicator(
                case, model, mu, weight, kernel, args.dsa_grid,
            ) if args.selector == "dsa_moment" else {}
        )
        relative, flux = reference_metrics(case, model, reference, device)
        parameters = model.trainable_count
        physics_indicator = (
            selector_diagnostics["dsa_moment_indicator"]
            if args.selector == "dsa_moment" else physics
        )
        selection_score = (
            math.log(max(physics_indicator, 1e-20))
            + args.parameter_penalty * parameters / parameter_median
        )
        probe.append({
            "family": family, "physics_validation": physics,
            "relative_l2_evaluation_only": relative,
            "scalar_flux_relative_l2_evaluation_only": flux,
            "trainable_parameters": parameters,
            "selector": args.selector,
            "selection_indicator": physics_indicator,
            "selection_score": selection_score,
            **selector_diagnostics,
        })
    winner = (args.fixed_family if args.fixed_family is not None else
              min(probe, key=lambda row: row["selection_score"])["family"])
    if args.selection_only:
        payload = {
            "family": "stationary_transport", "case": case.name,
            "mode": "physics_only_branch_selection_stability",
            "candidate_families": list(args.families), "probe": probe,
            "selected_branch_family": winner,
            "selector": args.selector,
            "task_parameter": args.task_parameter, "seed": args.seed,
            "reference_role": "evaluation_only", "dataloss": False,
            "output": str(output),
        }
        (output / "configuration.json").write_text(
            json.dumps({**vars(args), **payload}, indent=2, default=str)
        )
        (output / "summary.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2)); return
    del candidates
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # The probe is a selector, not additional optimization for the winning
    # profile.  Rebuild the winner from its declared seed so that automatic
    # selection and the fixed-profile control both receive exactly
    # ``refine_steps`` formal updates from the same initialization law.
    seed_winner = initialization_family(winner)
    winner_index = FAMILIES.index(seed_winner)
    torch.manual_seed(args.seed + 1009 * (winner_index + 1))
    model = FlexibleTransportCP(
        case, winner, rank=args.cp_rank, width=args.width, depth=args.depth,
        physics_modes=args.physics_modes,
    ).to(device)
    configuration = {
        **vars(args), "output": str(output),
        "selected_branch_family": winner,
        "candidate_families": list(args.families),
        "probe": probe,
        "formal_reinitialization": True,
        "formal_optimizer_updates": args.refine_steps,
        "reference_role": "evaluation_only", "dataloss": False,
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2, default=str)
    )
    formal_started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.refine_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.refine_steps),
        eta_min=args.refine_lr*args.scheduler_min_factor,
    )
    best_physics, best_state, best_row = math.inf, None, None
    minima = {"relative_l2": math.inf, "scalar_flux_relative_l2": math.inf}
    for step in range(args.refine_steps + 1):
        x = sample_x(
            case, args.batch_x, device, args.interface_fraction,
            args.interface_width,
        )
        loss, pde, boundary = objective(
            case, model, x, mu, weight, kernel, args.boundary_weight
        )
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
        if step in (0, 1) or step % args.eval_every == 0 or step == args.refine_steps:
            model.eval()
            locked = float(objective(
                case, model, validation, mu, weight, kernel,
                args.boundary_weight,
            )[0].detach())
            relative, flux = reference_metrics(case, model, reference, device)
            model.train()
            minima["relative_l2"] = min(minima["relative_l2"], relative)
            minima["scalar_flux_relative_l2"] = min(
                minima["scalar_flux_relative_l2"], flux
            )
            row = {
                "step": step, "physics_validation": locked,
                "loss": float(loss.detach()), "pde": float(pde.detach()),
                "boundary": float(boundary.detach()),
                "relative_l2_evaluation_only": relative,
                "scalar_flux_relative_l2_evaluation_only": flux,
                "seconds": time.perf_counter() - formal_started,
            }
            with history_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if locked < best_physics:
                best_physics, best_row = locked, row
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no checkpoint")
    torch.save({"model": best_state, "row": best_row}, output / "best_physics.pt")
    summary = {
        "family": "stationary_transport", "case": case.name,
        "mode": ("fixed_dimension_specialized_branch"
                 if args.fixed_family is not None else
                 "automatic_flexible_dimension_branch_selection"),
        "candidate_families": list(args.families), "probe": probe,
        "selected_branch_family": winner,
        "selector": args.selector,
        "rank": model.rank, "trainable_parameters": model.trainable_count,
        "probe_steps_per_candidate": args.probe_steps,
        "refine_steps": args.refine_steps,
        "formal_reinitialization": True,
        "best_step": best_row["step"],
        "best_physics_validation": best_physics,
        "physics_selected_relative_l2": best_row["relative_l2_evaluation_only"],
        "physics_selected_scalar_flux_relative_l2":
            best_row["scalar_flux_relative_l2_evaluation_only"],
        "minimum_monitored_relative_l2": minima["relative_l2"],
        "minimum_monitored_scalar_flux_relative_l2":
            minima["scalar_flux_relative_l2"],
        "probe_compute_seconds": probe_seconds,
        "selected_model_compute_seconds": time.perf_counter() - formal_started,
        "seconds": time.perf_counter() - started,
        "reference_role": "evaluation_only", "dataloss": False,
        "output": str(output),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--families", nargs="+", choices=FAMILIES,
                   default=list(DEFAULT_FAMILIES),
                   help="candidate branch families; one value gives a fixed-family ablation")
    p.add_argument("--fixed-family", choices=FAMILIES,
                   help="use this predeclared branch in the main run; no probe updates are performed")
    p.add_argument("--task-parameter", type=float, default=None)
    p.add_argument("--reference", type=Path, default=None,
                   help="evaluation-only reference for this target parameter")
    p.add_argument("--probe-steps", type=int, default=1200)
    p.add_argument("--refine-steps", type=int, default=3000)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--refine-lr", type=float, default=5e-4)
    p.add_argument("--scheduler-min-factor", type=float, default=0.05)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--physics-modes", type=int, default=None)
    p.add_argument(
        "--cp-rank", type=int, default=None,
        help="override the case CP rank for a trained rank-sensitivity audit",
    )
    p.add_argument("--batch-x", type=int, default=64)
    p.add_argument("--validation-x", type=int, default=128)
    p.add_argument("--quadrature-order", type=int, default=32)
    p.add_argument("--interface-fraction", type=float, default=0.25)
    p.add_argument("--interface-width", type=float, default=0.02)
    p.add_argument("--boundary-weight", type=float, default=10.0)
    p.add_argument("--parameter-penalty", type=float, default=0.02)
    p.add_argument(
        "--selector", choices=("raw_residual", "dsa_moment"),
        default="raw_residual",
        help="physics-only indicator used to rank the probed branches",
    )
    p.add_argument("--dsa-grid", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--variant", default="fourier_mlp_physics_correction_v1")
    p.add_argument("--output", type=Path, default=(
        ROOT / "results/multiphysics_capacity_routing/stationary_transport"
    ))
    p.add_argument("--selection-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.fixed_family is not None:
        args.families = [args.fixed_family]
        args.probe_steps = 0
    run(args)
