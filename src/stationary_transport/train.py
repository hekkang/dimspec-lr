#!/usr/bin/env python3
"""Physics-only separable solver for stationary continuous-angle transport."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent/"dimensionwise_12case"))

from model import DimensionWiseCP  # noqa: E402
from common.residual_jacobian_bank import grow_from_residual_jacobian  # noqa: E402
from stationary_transport.cases import TransportCase, get_case  # noqa: E402


def quadrature(case: TransportCase, order: int, device: torch.device):
    mu, weight = leggauss(order)
    vandermonde = legvander(mu, len(case.legendre_coefficients)-1)
    kernel = ((vandermonde*np.asarray(case.legendre_coefficients)[None, :])
              @ vandermonde.T)
    return (torch.tensor(mu, device=device, dtype=torch.float32),
            torch.tensor(weight, device=device, dtype=torch.float32),
            torch.tensor(kernel, device=device, dtype=torch.float32))


def sample_x(case: TransportCase, count: int, device: torch.device,
             interface_fraction: float, width: float,
             seed: int | None = None) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    interface_count = (int(round(count*interface_fraction))
                       if case.material_edges else 0)
    uniform_count = count-interface_count
    values = [torch.rand(uniform_count, device=device, generator=generator)]
    if interface_count:
        edges = torch.tensor(case.material_edges, device=device)
        selected = torch.randint(len(edges), (interface_count,), device=device,
                                 generator=generator)
        jitter = (2.0*torch.rand(interface_count, device=device,
                                generator=generator)-1.0)*width
        values.append((edges[selected]+jitter).clamp(0.0, 1.0))
    return torch.cat(values).reshape(-1)


def transport_residual(case: TransportCase, model: torch.nn.Module,
                       x: torch.Tensor, mu: torch.Tensor,
                       weight: torch.Tensor, kernel: torch.Tensor):
    if hasattr(model, "separable_factors"):
        return transport_residual_separable(case, model, x, mu, weight, kernel)
    nx, nmu = x.numel(), mu.numel()
    xx = x[:, None].expand(nx, nmu)
    mm = mu[None, :].expand(nx, nmu)
    points = torch.stack((xx, mm), dim=-1).reshape(-1, 2).requires_grad_(True)
    psi = physical_intensity(case, model, points)
    derivative = torch.autograd.grad(psi.sum(), points, create_graph=True)[0][:, 0]
    psi = psi.reshape(nx, nmu)
    derivative = derivative.reshape(nx, nmu)
    scattering, absorption = case.coefficients_torch(x)
    angular_integral = (psi*weight[None, :])@kernel.T
    residual = (mu[None, :]*derivative
                +(scattering+absorption)[:, None]*psi
                -0.5*scattering[:, None]*angular_integral)
    scale = ((scattering+absorption)[:, None].square()*psi.detach().square()
             +mu[None, :].square()*derivative.detach().square()).mean()
    scale = scale.clamp_min(1.0)
    angular_mean_residual = 0.5*(residual*weight[None, :]).sum(dim=1)
    # In the near-diffusion regime the isotropic collision eigenvalue is the
    # absorption fraction.  An unweighted least-squares residual therefore
    # under-penalizes precisely the scalar-flux error by this small spectral
    # gap.  The bounded inverse-gap factor is a residual preconditioner, not a
    # data term; it also enforces current conservation in pure-scattering S1.
    total = scattering+absorption
    isotropic_gap = absorption/total.clamp_min(1e-6)
    inverse_gap = 1.0/(isotropic_gap+0.05)
    moment_loss = (inverse_gap*angular_mean_residual).square().mean()/scale
    # AP-style micro--macro conditioning.  This is an algebraic decomposition
    # of the same transport residual, not a different output model: the mu
    # branch still supplies every continuous angular low-rank factor.
    macro = angular_mean_residual[:, None]
    macro_losses = [moment_loss]
    if len(case.legendre_coefficients) > 1:
        first_moment = 1.5*(residual*mu[None, :]*weight[None, :]).sum(dim=1)
        macro = macro+first_moment[:, None]*mu[None, :]
        macro_losses.append(first_moment.square().mean()/scale)
    micro = residual-macro
    micro_scale = (residual.detach().square().mean()).clamp_min(1.0)
    micro_loss = micro.square().mean()/micro_scale
    ap_loss = torch.stack(macro_losses).mean()+0.25*micro_loss
    return (residual.square().mean()/scale
            +case.moment_precondition_weight*ap_loss), residual, psi


def transport_residual_separable(case: TransportCase, model: torch.nn.Module,
                                 x: torch.Tensor, mu: torch.Tensor,
                                 weight: torch.Tensor,
                                 kernel: torch.Tensor):
    """Equivalent tensor-grid residual without repeated axis evaluations.

    This preserves the continuous-angle CP model and the same dense angular
    scattering matrix.  Only the execution order changes: X(x) and M(mu) are
    evaluated separately, then contracted into the quadrature grid.
    """
    x_unique = x.detach().clone().requires_grad_(True)
    (left, right), (left_x, _) = torch.func.jvp(
        lambda coordinate: model.separable_factors(coordinate, mu),
        (x_unique,), (torch.ones_like(x_unique),),
    )
    normalizer = getattr(model, "cp_normalizer", math.sqrt(model.rank))
    raw = (left @ right.T) / normalizer
    raw_x = (left_x @ right.T) / normalizer

    mm = mu[None, :]
    positive = mm > 0.0
    distance = torch.where(positive, x_unique[:, None], 1.0-x_unique[:, None])
    distance_x = torch.where(positive, torch.ones_like(mm), -torch.ones_like(mm))
    if case.lifting_kind == "linear":
        lifting = torch.where(
            positive,
            torch.full_like(distance, case.left_inflow),
            torch.full_like(distance, case.right_inflow),
        )
        lifting_x = torch.zeros_like(lifting)
    elif case.lifting_kind == "uncollided":
        safe_mu = mm.abs().clamp_min(1e-6)
        tau_left = case.optical_depth_torch(x_unique, from_left=True)[:, None]
        tau_right = case.optical_depth_torch(x_unique, from_left=False)[:, None]
        lifting = torch.where(
            positive,
            case.left_inflow*torch.exp(-tau_left/safe_mu),
            case.right_inflow*torch.exp(-tau_right/safe_mu),
        )
        scattering, absorption = case.coefficients_torch(x_unique)
        total = (scattering+absorption)[:, None]
        lifting_x = torch.where(
            positive, -total*lifting/safe_mu, total*lifting/safe_mu,
        )
    else:
        raise ValueError(f"unknown transport lifting {case.lifting_kind!r}")

    psi = lifting+distance*raw
    derivative = lifting_x+distance_x*raw+distance*raw_x
    scattering, absorption = case.coefficients_torch(x_unique)
    angular_integral = (psi*weight[None, :])@kernel.T
    residual = (mm*derivative
                +(scattering+absorption)[:, None]*psi
                -0.5*scattering[:, None]*angular_integral)
    scale = ((scattering+absorption)[:, None].square()*psi.detach().square()
             +mm.square()*derivative.detach().square()).mean().clamp_min(1.0)
    angular_mean_residual = 0.5*(residual*weight[None, :]).sum(dim=1)
    total = scattering+absorption
    isotropic_gap = absorption/total.clamp_min(1e-6)
    inverse_gap = 1.0/(isotropic_gap+0.05)
    moment_loss = (inverse_gap*angular_mean_residual).square().mean()/scale
    macro = angular_mean_residual[:, None]
    macro_losses = [moment_loss]
    if len(case.legendre_coefficients) > 1:
        first_moment = 1.5*(residual*mm*weight[None, :]).sum(dim=1)
        macro = macro+first_moment[:, None]*mm
        macro_losses.append(first_moment.square().mean()/scale)
    micro = residual-macro
    micro_scale = residual.detach().square().mean().clamp_min(1.0)
    micro_loss = micro.square().mean()/micro_scale
    ap_loss = torch.stack(macro_losses).mean()+0.25*micro_loss
    loss = (residual.square().mean()/scale
            +case.moment_precondition_weight*ap_loss)
    return loss, residual, psi


def inflow_loss(case: TransportCase, model: torch.nn.Module,
                mu: torch.Tensor) -> torch.Tensor:
    """Diagnostic loss for the analytically hard-constrained inflow traces."""
    positive = mu > 0.0
    left = torch.stack((torch.zeros_like(mu[positive]), mu[positive]), dim=1)
    right = torch.stack((torch.ones_like(mu[~positive]), mu[~positive]), dim=1)
    return 0.5*((physical_intensity(case, model, left)-case.left_inflow).square().mean()
                +(physical_intensity(case, model, right)-case.right_inflow).square().mean())


def physical_intensity(case: TransportCase, model: torch.nn.Module,
                       points: torch.Tensor) -> torch.Tensor:
    """Lift the raw separable field into an exact half-range inflow solution.

    For positive directions only the left trace is prescribed, hence the
    trainable correction is multiplied by ``x``.  For negative directions
    only the right trace is prescribed and the multiplier is ``1-x``.  This
    removes the thin, under-sampled boundary spikes admitted by a soft loss
    while leaving both outgoing traces free.
    """
    x, mu = points[:, 0], points[:, 1]
    positive = mu > 0.0
    distance = torch.where(positive, x, 1.0-x)
    if case.lifting_kind == "linear":
        lifting = torch.where(
            positive,
            torch.full_like(x, case.left_inflow),
            torch.full_like(x, case.right_inflow),
        )
    elif case.lifting_kind == "uncollided":
        safe_mu = mu.abs().clamp_min(1e-6)
        tau_left = case.optical_depth_torch(x, from_left=True)
        tau_right = case.optical_depth_torch(x, from_left=False)
        lifting = torch.where(
            positive,
            case.left_inflow*torch.exp(-tau_left/safe_mu),
            case.right_inflow*torch.exp(-tau_right/safe_mu),
        )
    else:
        raise ValueError(f"unknown transport lifting {case.lifting_kind!r}")
    return lifting+distance*model(points)


def configure_mapping(model: DimensionWiseCP, path: Path | None,
                      initialization: str, parameter: float):
    if path is None:
        return None
    bank = torch.load(path, map_location="cpu", weights_only=False)
    for axis, branch in enumerate(model.branches):
        key = f"axis{axis}"
        mapping = bank["mappings"][key]
        if tuple(mapping.shape) != tuple(branch.mapping.shape):
            raise ValueError(f"{key} Mapping shape mismatch")
        branch.mapping.copy_(mapping.to(branch.mapping))
        if initialization == "zero":
            branch.latent.data.zero_()
        elif initialization == "mean":
            branch.latent.data.copy_(bank["initial_latents"][key].to(branch.latent))
        elif initialization == "parameter":
            decoder = bank["parameter_decoders"][key]
            powers = torch.as_tensor(parameter).pow(torch.arange(
                decoder.shape[0], dtype=decoder.dtype,
            ))
            branch.latent.data.copy_((powers@decoder).to(branch.latent))
        elif initialization == "target_krylov":
            coordinates = bank.get("target_krylov_latents", {}).get(key)
            if coordinates is None:
                raise ValueError(
                    "mapping bank does not contain target Krylov initialization"
                )
            branch.latent.data.copy_(coordinates.to(branch.latent))
    return bank


def active_count(model: DimensionWiseCP) -> int:
    return sum(branch.active_latent_count for branch in model.branches)


def set_initial_count(model: DimensionWiseCP, count: int) -> None:
    for branch in model.branches:
        active = min(count, branch.latent.numel())
        branch.active_latent_size.fill_(active)
        with torch.no_grad():
            branch.latent[active:].zero_()


def grow(model: DimensionWiseCP, count: int) -> bool:
    changed = False
    for branch in model.branches:
        old = int(branch.active_latent_size)
        new = min(branch.latent.numel(), old+count)
        branch.active_latent_size.fill_(new)
        changed = changed or new > old
    return changed


@torch.no_grad()
def reference_metrics(case: TransportCase, model: torch.nn.Module, path: Path,
                      device: torch.device, chunk: int = 65536):
    data = np.load(path)
    x = torch.tensor(data["x"], device=device, dtype=torch.float32)
    mu = torch.tensor(data["mu"], device=device, dtype=torch.float32)
    weights = torch.tensor(data["weights"], device=device, dtype=torch.float32)
    reference = torch.tensor(data["psi"], device=device, dtype=torch.float32)
    points = torch.stack(torch.meshgrid(x, mu, indexing="ij"), dim=-1).reshape(-1, 2)
    pieces = [physical_intensity(case, model, points[start:start+chunk])
              for start in range(0, points.shape[0], chunk)]
    prediction = torch.cat(pieces).reshape_as(reference)
    relative = (torch.linalg.vector_norm(prediction-reference)
                /torch.linalg.vector_norm(reference).clamp_min(1e-12))
    predicted_flux = prediction@weights
    reference_flux = reference@weights
    flux_relative = (torch.linalg.vector_norm(predicted_flux-reference_flux)
                     /torch.linalg.vector_norm(reference_flux).clamp_min(1e-12))
    return float(relative), float(flux_relative)


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    case = get_case(args.case, args.parameter)
    variant = ("latent_fullspan_implicit" if args.mode == "full_weight"
               else "latent_compressed")
    model = DimensionWiseCP(case, variant, args.latent_fraction,
                            args.mapping_seed).to(device)
    with torch.no_grad():
        for branch in model.branches:
            branch.base.mul_(args.base_scale)
    bank = configure_mapping(model, args.mapping_bank, args.mapping_init,
                             case.parameter)
    if args.mode == "latent":
        if bank is None:
            raise ValueError("latent mode requires --mapping-bank")
        set_initial_count(model, args.initial_latent_count)
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=device,
                                    weights_only=False)
        model.load_state_dict(resume_payload["model"])
    trainable_names = [name for name, value in model.named_parameters()
                       if value.requires_grad]
    if args.mode == "latent" and any(not name.endswith(".latent")
                                      for name in trainable_names):
        raise RuntimeError(f"non-latent target parameter: {trainable_names}")
    mu, weight, kernel = quadrature(case, args.quadrature_order, device)
    validation_x = sample_x(case, args.validation_x, device,
                            args.interface_fraction, args.interface_width,
                            seed=2026072811)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration.update({
        "output": str(args.output),
        "reference": str(args.reference) if args.reference else None,
        "mapping_bank": str(args.mapping_bank) if args.mapping_bank else None,
        "task_kind": case.task_kind, "case_parameter": case.parameter,
        "reference_role": ("evaluation_only" if args.reference else "not_loaded"),
        "dataloss": False,
        "resume": str(args.resume) if args.resume else None,
    })
    atomic_json(output/"configuration.json", configuration)
    history, rank_events = [], []
    best_physics, best_state, bad = math.inf, None, 0
    over_target_checks = 0
    proposal = None
    started = time.perf_counter()
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            x = sample_x(case, args.batch_x, device, args.interface_fraction,
                         args.interface_width)
            pde, _, _ = transport_residual(case, model, x, mu, weight, kernel)
            boundary = inflow_loss(case, model, mu)
            loss = pde+args.boundary_weight*boundary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        if step % args.eval_every:
            continue
        pde, residual, _ = transport_residual(
            case, model, validation_x, mu, weight, kernel,
        )
        boundary = inflow_loss(case, model, mu)
        physics = float(pde+args.boundary_weight*boundary)
        if args.reference is None:
            relative, flux_relative = None, None
        else:
            relative, flux_relative = reference_metrics(
                case, model, args.reference, device,
            )
        row = {
            "step": step, "physics_validation": physics,
            "pde_validation": float(pde), "inflow_validation": float(boundary),
            "residual_rms": float(residual.square().mean().sqrt()),
            "relative_l2_evaluation_only": relative,
            "scalar_flux_relative_l2_evaluation_only": flux_relative,
            "active_latent_rank": active_count(model),
            "elapsed_seconds": time.perf_counter()-started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if physics < best_physics:
            best_physics, bad = physics, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "configuration": configuration,
                        "physics_validation": best_physics, "step": step},
                       output/"best_physics.pt")
        else:
            bad += 1
        if args.mode != "latent" or not args.adaptive_latent_rank:
            continue
        if proposal is None:
            over_target_checks = (over_target_checks+1
                                  if physics > args.rank_residual_target else 0)
        if proposal is not None:
            proposal["checks"] += 1
            if physics > args.rank_rollback_factor*proposal["baseline"]:
                model.load_state_dict(proposal["state"])
                event = {"step": step, "status": "rollback",
                         "baseline": proposal["baseline"], "rejected": physics,
                         "active_latent_rank": active_count(model)}
                rank_events.append(event)
                proposal, bad, over_target_checks = None, 0, 0
            elif proposal["checks"] >= args.rank_probation_evals:
                event = {"step": step, "status": "accept_growth",
                         "physics_validation": physics,
                         "active_latent_rank": active_count(model)}
                rank_events.append(event)
                proposal = None
                over_target_checks = 0
        elif (physics > args.rank_residual_target
              and (bad >= args.rank_patience
                   or over_target_checks >= args.rank_patience)):
            state, old = copy.deepcopy(model.state_dict()), active_count(model)
            jacobian_event = None
            if args.dynamic_jacobian_growth:
                def target_physics():
                    target_pde, _, _ = transport_residual(
                        case, model, validation_x, mu, weight, kernel,
                    )
                    return target_pde+args.boundary_weight*inflow_loss(
                        case, model, mu,
                    )

                jacobian_event = grow_from_residual_jacobian(
                    model, [("ap_transport", target_physics)],
                    directions_per_event=args.jacobian_directions_per_growth,
                )
                grew = jacobian_event["accepted_directions"] > 0
            else:
                grew = grow(model, args.rank_growth_count)
            if grew:
                proposal = {"state": state, "baseline": physics, "checks": 0}
                rank_events.append({"step": step,
                                    "status": ("jacobian_grow" if jacobian_event else "grow"),
                                    "old": old,
                                    "new": active_count(model),
                                    "physics_validation": physics,
                                    "jacobian_bank": jacobian_event})
                bad, over_target_checks = 0, 0
    if best_state is None:
        raise RuntimeError("no physics checkpoint")
    model.load_state_dict(best_state)
    if args.reference is None:
        selected_relative, selected_flux = None, None
    else:
        selected_relative, selected_flux = reference_metrics(
            case, model, args.reference, device,
        )
    monitored = [row["relative_l2_evaluation_only"] for row in history
                 if row["relative_l2_evaluation_only"] is not None]
    summary = {
        "family": "stationary_transport", "case": case.name,
        "task_kind": case.task_kind, "task_parameter": case.parameter,
        "mode": args.mode, "strict_latent_only": args.mode == "latent",
        "trainable_names": trainable_names,
        "trainable_parameters": sum(value.numel() for value in model.parameters()
                                    if value.requires_grad),
        "active_latent_at_selection": active_count(model),
        "best_physics_validation": best_physics,
        "physics_selected_relative_l2": selected_relative,
        "physics_selected_scalar_flux_relative_l2": selected_flux,
        "minimum_monitored_relative_l2": min(monitored) if monitored else None,
        "rank_events": rank_events,
        "reference_role": ("evaluation_only" if args.reference else "not_loaded"),
        "data_assisted": False, "total_seconds": time.perf_counter()-started,
        "output": str(output),
    }
    atomic_json(output/"history.json", history)
    atomic_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--case", choices=("S1", "S2", "S3", "S4"), required=True)
    value.add_argument("--parameter", type=float, default=None)
    value.add_argument("--mode", choices=("full_weight", "latent"),
                       default="full_weight")
    value.add_argument("--mapping-bank", type=Path, default=None)
    value.add_argument("--mapping-init", choices=("zero", "mean", "parameter",
                                                   "target_krylov"),
                       default="zero")
    value.add_argument("--latent-fraction", type=float, default=0.5)
    value.add_argument("--mapping-seed", type=int, default=20260728)
    value.add_argument("--base-scale", type=float, default=0.05)
    value.add_argument("--initial-latent-count", type=int, default=4)
    value.add_argument("--adaptive-latent-rank", action=argparse.BooleanOptionalAction,
                       default=True)
    value.add_argument("--rank-growth-count", type=int, default=4)
    value.add_argument("--rank-residual-target", type=float, default=5e-4)
    value.add_argument("--rank-patience", type=int, default=2)
    value.add_argument("--rank-rollback-factor", type=float, default=1.05)
    value.add_argument("--rank-probation-evals", type=int, default=3)
    value.add_argument("--dynamic-jacobian-growth",
                       action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--jacobian-directions-per-growth", type=int, default=1)
    value.add_argument("--quadrature-order", type=int, default=32)
    value.add_argument("--steps", type=int, default=3000)
    value.add_argument("--batch-x", type=int, default=64)
    value.add_argument("--validation-x", type=int, default=128)
    value.add_argument("--interface-fraction", type=float, default=0.25)
    value.add_argument("--interface-width", type=float, default=0.02)
    value.add_argument("--boundary-weight", type=float, default=10.0)
    value.add_argument("--lr", type=float, default=2e-3)
    value.add_argument("--grad-clip", type=float, default=10.0)
    value.add_argument("--eval-every", type=int, default=100)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--reference", type=Path, default=None,
                       help="evaluation-only SN reference; omit for source-bank training")
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--resume", type=Path, default=None,
                       help="continue from a physics-selected model/bank state")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
