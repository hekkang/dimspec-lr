#!/usr/bin/env python3
"""Physics-only full-span or frozen-Mapping solver for A1--A4."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DIMENSIONWISE = HERE.parent / "dimensionwise_12case"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(DIMENSIONWISE))

from model import DimensionWiseCP  # noqa: E402
from common.residual_jacobian_bank import grow_from_residual_jacobian  # noqa: E402
from steady_anisotropic.cases import AnisotropicCase, get_case  # noqa: E402


def sample(count: int, device: torch.device, seed: int | None = None,
           requires_grad: bool = True) -> torch.Tensor:
    if seed is None:
        value = torch.rand(count, 2, device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        value = torch.rand(count, 2, generator=generator, device=device)
    return value.requires_grad_(requires_grad)


def operator(case: AnisotropicCase, model: torch.nn.Module,
             points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = points.requires_grad_(True)
    value = model(points)
    gradient = torch.autograd.grad(value.sum(), points, create_graph=True)[0]
    hessian_columns = [torch.autograd.grad(
        gradient[:, axis].sum(), points, create_graph=True,
    )[0] for axis in range(2)]
    hessian = torch.stack(hessian_columns, dim=1)
    hessian = 0.5*(hessian+hessian.transpose(1, 2))
    diffusion = case.diffusion(points)
    convection = case.convection(points)
    reaction = case.reaction(points)
    result = -(diffusion*hessian).sum(dim=(1, 2))
    result = result+(convection*gradient).sum(dim=1)+reaction*value
    return value, result


def physics_loss(case: AnisotropicCase, model: torch.nn.Module,
                 points: torch.Tensor) -> tuple[torch.Tensor, dict]:
    _, prediction = operator(case, model, points)
    forcing = case.forcing(points)
    residual = prediction-forcing
    scale = forcing.square().mean().detach().clamp_min(1e-8)
    pde = residual.square().mean()/scale
    return pde, {"pde": pde, "residual_rms": residual.square().mean().sqrt(),
                 "forcing_rms": scale.sqrt()}


def build_inverse_symbol_target(case: AnisotropicCase,
                                points_per_axis: int,
                                device: torch.device) -> dict | None:
    """Build a projected anisotropic operator and prescribed forcing.

    For A1/A2, ``-div(D grad)`` is diagonal in the fixed sine dictionary.
    The mixed derivative and convection matrices are retained. Dividing the
    coefficient residual by the positive diffusion diagonal is a physics
    preconditioner, not an interior solution-data loss.
    """
    if case.feature_kinds != ("sine_pi", "sine_pi"):
        return None
    q = int(points_per_axis)
    coordinate = (torch.arange(q, device=device, dtype=torch.float32)+0.5)/q
    xx, yy = torch.meshgrid(coordinate, coordinate, indexing="ij")
    points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    forcing = case.forcing(points).reshape(q, q)
    mx = torch.arange(1, case.max_modes[0]+1, device=device)[None, :]
    my = torch.arange(1, case.max_modes[1]+1, device=device)[None, :]
    basis_x = torch.sin(torch.pi*coordinate[:, None]*mx)
    basis_y = torch.sin(torch.pi*coordinate[:, None]*my)
    forcing_coefficients = (4.0/(q*q))*(basis_x.T@forcing@basis_y)
    diffusion = case.diffusion(points[:1])[0]
    convection = case.convection(points[:1])[0]
    reaction = case.reaction(points[:1])[0]
    mode_x = torch.arange(1, case.max_modes[0]+1, device=device)[:, None]
    mode_y = torch.arange(1, case.max_modes[1]+1, device=device)[None, :]
    diagonal = (diffusion[0, 0]*(torch.pi*mode_x).square()
                +diffusion[1, 1]*(torch.pi*mode_y).square()+reaction)
    cosine_x = torch.cos(torch.pi*coordinate[:, None]*mx)
    cosine_y = torch.cos(torch.pi*coordinate[:, None]*my)
    # Project d/dx sin(m*pi*x)=m*pi*cos(m*pi*x) back to the sine basis.
    derivative_x = ((2.0/q)*(basis_x.T@cosine_x)
                    *(torch.pi*mx.reshape(1, -1)))
    derivative_y = ((2.0/q)*(basis_y.T@cosine_y)
                    *(torch.pi*my.reshape(1, -1)))
    return {
        "forcing": forcing_coefficients.detach(),
        "diagonal": diagonal.detach(),
        "diffusion": diffusion.detach(),
        "convection": convection.detach(),
        "reaction": reaction.detach(),
        "derivative_x": derivative_x.detach(),
        "derivative_y": derivative_y.detach(),
    }


def inverse_symbol_loss(model: DimensionWiseCP,
                        data: dict):
    # ``DimensionWiseCP`` may expose a larger allocated rank than the active
    # CP rank used by Grow--Freeze--Refine.  Restrict the projected operator
    # to the represented columns; otherwise inactive columns leak into the
    # physics loss and a rank birth is no longer function preserving.
    active = int(getattr(model, "active_rank", model.rank))
    left = model.branches[0].effective_coefficient()[:, :active]
    right = model.branches[1].effective_coefficient()[:, :active]
    coefficients = left@right.T/math.sqrt(model.rank)
    derivative_x, derivative_y = data["derivative_x"], data["derivative_y"]
    diffusion, convection = data["diffusion"], data["convection"]
    operator_coefficients = data["diagonal"]*coefficients
    operator_coefficients = operator_coefficients-2.0*diffusion[0, 1]*(
        derivative_x@coefficients@derivative_y.T
    )
    operator_coefficients = operator_coefficients+convection[0]*(
        derivative_x@coefficients
    )+convection[1]*(coefficients@derivative_y.T)
    residual = operator_coefficients-data["forcing"]
    preconditioned = residual/data["diagonal"]
    forcing_solution_scale = data["forcing"]/data["diagonal"]
    loss = (preconditioned.square().sum()
            /forcing_solution_scale.square().sum().clamp_min(1e-12))
    return loss, {"pde": loss,
                  "residual_rms": preconditioned.square().mean().sqrt(),
                  "forcing_rms": forcing_solution_scale.square().mean().sqrt()}


def boundary_loss(case: AnisotropicCase, model: torch.nn.Module,
                  per_face: int, device: torch.device,
                  seed: int | None = None) -> torch.Tensor:
    if case.feature_kinds == ("sine_pi", "sine_pi"):
        return torch.zeros((), device=device)
    values = []
    for axis in range(2):
        for face_index, endpoint in enumerate((0.0, 1.0)):
            face_seed = None if seed is None else seed+10*axis+face_index
            points = sample(per_face, device, face_seed, requires_grad=False)
            points[:, axis] = endpoint
            values.append(model(points).square().mean())
    return torch.stack(values).mean()


@torch.no_grad()
def reference_metric(case: AnisotropicCase, model: torch.nn.Module,
                     points: torch.Tensor) -> float:
    reference = case.exact(points)
    prediction = model(points)
    return float(torch.linalg.vector_norm(prediction-reference)
                 / torch.linalg.vector_norm(reference).clamp_min(1e-12))


def configure_mapping(model: DimensionWiseCP, path: Path | None,
                      initialization: str, parameter: float) -> dict | None:
    if path is None:
        return None
    bank = torch.load(path, map_location="cpu", weights_only=False)
    for axis, branch in enumerate(model.branches):
        key = f"axis{axis}"
        learned = bank["mappings"][key]
        if tuple(learned.shape) != tuple(branch.mapping.shape):
            raise ValueError(f"{key} Mapping shape mismatch")
        branch.mapping.copy_(learned.to(branch.mapping))
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
        else:
            raise ValueError(initialization)
    return bank


def active_count(model: DimensionWiseCP) -> int:
    return sum(branch.active_latent_count for branch in model.branches)


def set_initial_count(model: DimensionWiseCP, count: int) -> None:
    for branch in model.branches:
        active = min(count, branch.latent.numel())
        branch.active_latent_size.fill_(active)
        # Inactive coordinates must not carry a hidden initialization.  If
        # they were non-zero, a rank-growth event would change the represented
        # field discontinuously before the newly admitted coordinates receive
        # an optimizer step.
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


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    case = get_case(args.case, args.parameter)
    if args.cp_rank is not None:
        if args.cp_rank < 1:
            raise ValueError("--cp-rank must be positive")
        case = replace(case, rank=args.cp_rank)
    branch_control = (
        "permuted" if args.permute_branches else args.branch_control
    )
    if branch_control == "permuted":
        # Reverse the coordinate assignment while preserving rank, all atom
        # counts, neural components, and therefore the total parameter count.
        # Only the representation-to-coordinate correspondence changes.
        case = replace(
            case,
            feature_kinds=tuple(reversed(case.feature_kinds)),
            max_modes=tuple(reversed(case.max_modes)),
            dictionary_parameters=tuple(reversed(case.dictionary_parameters)),
        )
    elif branch_control == "param_matched_homogeneous":
        if case.name != "A4":
            raise ValueError(
                "the locked parameter-matched homogeneous control is defined "
                "only for A4"
            )
        # Both axes receive exactly the same generic hybrid dictionary and
        # the same number of atoms.  M=52 gives 10,176 trainable coefficients,
        # within 0.5% of the 10,128-parameter DimSpec A4 model.
        case = replace(
            case,
            feature_kinds=("hybrid", "hybrid"),
            max_modes=(52, 52),
            dictionary_parameters=(None, None),
        )
    elif branch_control == "capacity_routed_homogeneous":
        if case.name != "A4":
            raise ValueError(
                "the locked capacity-routed homogeneous control is defined "
                "only for A4"
            )
        # The branch family remains homogeneous, but the difficult streamwise
        # coordinate receives more generic atoms.  The 72/31 split gives
        # 10,080 parameters, within 0.5% of the DimSpec model, and closely
        # matches its per-axis coefficient allocation without including a
        # boundary-layer atom family.
        case = replace(
            case,
            feature_kinds=("hybrid", "hybrid"),
            max_modes=(72, 31),
            dictionary_parameters=(None, None),
        )
    elif branch_control in {"boundary_homogeneous_uniform", "boundary_homogeneous_routed"}:
        if case.name != "A4":
            raise ValueError("boundary homogeneous controls require A4")
        # Both axes have the same B dictionary, including the operator-scale
        # atom. Match 10,128 coefficients within 0.5%, with unchanged rank 24.
        modes = ((44, 44) if branch_control.endswith("uniform") else (64, 24))
        case = replace(case, feature_kinds=("boundary_layer", "boundary_layer"),
                       max_modes=modes,
                       dictionary_parameters=(case.task_parameter, case.task_parameter))
    variant = ("latent_fullspan_implicit" if args.mode == "full_weight"
               else "latent_compressed")
    model = DimensionWiseCP(case, variant, args.latent_fraction,
                            args.mapping_seed).to(device)
    # A high-order dictionary differentiated twice magnifies an O(1) random
    # CP initialization by O(k^2) per branch.  Keep a small non-zero base so
    # product gradients exist without swamping the manufactured source.
    with torch.no_grad():
        for branch in model.branches:
            branch.base.mul_(args.base_scale)
    bank = configure_mapping(model, args.mapping_bank, args.mapping_init,
                             case.task_parameter)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    validation = sample(args.validation, device, 2026072801)
    evaluation = sample(args.evaluation, device, 2026072802,
                        requires_grad=False)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration.update({
        "output": str(args.output),
        "resolved_branch_control": branch_control,
        "mapping_bank": str(args.mapping_bank) if args.mapping_bank else None,
        "case_parameter": case.task_parameter,
        "task_kind": case.task_kind,
        "reference_role": "evaluation_only",
        "dataloss": False,
        "resume": str(args.resume) if args.resume else None,
    })
    atomic_json(output/"configuration.json", configuration)
    history, rank_events = [], []
    best_physics = math.inf
    best_state = None
    bad_validations = 0
    over_target_checks = 0
    proposal = None
    started = time.perf_counter()
    inverse_target = build_inverse_symbol_target(
        case, args.spectral_points_per_axis, device,
    )
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            if inverse_target is None:
                points = sample(args.batch, device)
                pde, _ = physics_loss(case, model, points)
            else:
                pde, _ = inverse_symbol_loss(model, inverse_target)
            boundary = boundary_loss(case, model, args.boundary_batch, device)
            loss = pde+args.boundary_weight*boundary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        if step % args.eval_every != 0:
            continue
        if inverse_target is None:
            pde_validation, parts = physics_loss(case, model, validation)
        else:
            pde_validation, parts = inverse_symbol_loss(model, inverse_target)
        boundary_validation = boundary_loss(
            case, model, args.boundary_batch, device, 2026072803,
        )
        physics_validation = float(
            pde_validation+args.boundary_weight*boundary_validation
        )
        relative = reference_metric(case, model, evaluation)
        row = {
            "step": step, "physics_validation": physics_validation,
            "pde_validation": float(parts["pde"]),
            "boundary_validation": float(boundary_validation),
            "relative_l2_evaluation_only": relative,
            "active_latent_rank": active_count(model),
            "elapsed_seconds": time.perf_counter()-started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if physics_validation < best_physics:
            best_physics = physics_validation
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "configuration": configuration,
                        "physics_validation": best_physics,
                        "step": step}, output/"best_physics.pt")
            bad_validations = 0
        else:
            bad_validations += 1
        if args.mode != "latent" or not args.adaptive_latent_rank:
            continue
        if proposal is None:
            over_target_checks = (over_target_checks+1
                                  if physics_validation > args.rank_residual_target
                                  else 0)
        if proposal is not None:
            proposal["checks"] += 1
            if physics_validation > args.rank_rollback_factor*proposal["baseline"]:
                model.load_state_dict(proposal["state"])
                event = {"step": step, "status": "rollback",
                         "baseline": proposal["baseline"],
                         "rejected": physics_validation,
                         "active_latent_rank": active_count(model)}
                rank_events.append(event)
                print(json.dumps({"rank_event": event}), flush=True)
                proposal = None
                bad_validations = 0
                over_target_checks = 0
            elif proposal["checks"] >= args.rank_probation_evals:
                event = {"step": step, "status": "accept_growth",
                         "physics_validation": physics_validation,
                         "active_latent_rank": active_count(model)}
                rank_events.append(event)
                print(json.dumps({"rank_event": event}), flush=True)
                proposal = None
                over_target_checks = 0
        elif (physics_validation > args.rank_residual_target
              and (bad_validations >= args.rank_patience
                   or over_target_checks >= args.rank_patience)):
            state = copy.deepcopy(model.state_dict())
            old = active_count(model)
            jacobian_event = None
            if args.dynamic_jacobian_growth:
                def target_physics():
                    if inverse_target is None:
                        target_pde, _ = physics_loss(case, model, validation)
                    else:
                        target_pde, _ = inverse_symbol_loss(model, inverse_target)
                    target_boundary = boundary_loss(
                        case, model, args.boundary_batch, device, 2026072803,
                    )
                    return target_pde+args.boundary_weight*target_boundary

                jacobian_event = grow_from_residual_jacobian(
                    model, [("pde_boundary", target_physics)],
                    directions_per_event=args.jacobian_directions_per_growth,
                )
                grew = jacobian_event["accepted_directions"] > 0
            else:
                grew = grow(model, args.rank_growth_count)
            if grew:
                proposal = {"state": state, "baseline": physics_validation,
                            "checks": 0}
                event = {"step": step,
                         "status": ("jacobian_grow" if jacobian_event else "grow"),
                         "old": old,
                         "new": active_count(model),
                         "physics_validation": physics_validation,
                         "jacobian_bank": jacobian_event}
                rank_events.append(event)
                print(json.dumps({"rank_event": event}), flush=True)
                bad_validations = 0
                over_target_checks = 0
    if best_state is None:
        raise RuntimeError("no physics checkpoint was selected")
    model.load_state_dict(best_state)
    selected_relative = reference_metric(case, model, evaluation)
    minimum_relative = min(row["relative_l2_evaluation_only"] for row in history)
    summary = {
        "family": "steady_anisotropic_cdr", "case": case.name,
        "task_kind": case.task_kind, "task_parameter": case.task_parameter,
        "mode": args.mode, "strict_latent_only": args.mode == "latent",
        "trainable_names": trainable_names,
        "trainable_parameters": sum(value.numel() for value in model.parameters()
                                    if value.requires_grad),
        "active_latent_at_selection": active_count(model),
        "best_physics_validation": best_physics,
        "physics_selected_relative_l2": selected_relative,
        "minimum_monitored_relative_l2": minimum_relative,
        "rank_events": rank_events, "reference_role": "evaluation_only",
        "data_assisted": False, "total_seconds": time.perf_counter()-started,
        "output": str(output),
    }
    atomic_json(output/"history.json", history)
    atomic_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--case", choices=("A1", "A2", "A3", "A4"),
                       required=True)
    value.add_argument("--parameter", type=float, default=None)
    value.add_argument(
        "--cp-rank", type=int, default=None,
        help="override the benchmark CP rank for a controlled fixed-rank run",
    )
    value.add_argument(
        "--permute-branches", action="store_true",
        help="swap the two dimension-specific branch assignments",
    )
    value.add_argument(
        "--branch-control",
        choices=(
            "correct", "permuted", "param_matched_homogeneous",
            "capacity_routed_homogeneous",
            "boundary_homogeneous_uniform", "boundary_homogeneous_routed",
        ),
        default="correct",
        help=(
            "locked A4 causal-control profile; --permute-branches is retained "
            "as a compatibility alias for 'permuted'"
        ),
    )
    value.add_argument("--mode", choices=("full_weight", "latent"),
                       default="full_weight")
    value.add_argument("--mapping-bank", type=Path, default=None)
    value.add_argument("--mapping-init", choices=("zero", "mean", "parameter",
                                                   "target_krylov"),
                       default="zero")
    value.add_argument("--latent-fraction", type=float, default=0.5)
    value.add_argument("--mapping-seed", type=int, default=20260728)
    value.add_argument("--base-scale", type=float, default=0.01)
    value.add_argument("--initial-latent-count", type=int, default=4)
    value.add_argument("--adaptive-latent-rank", action=argparse.BooleanOptionalAction,
                       default=True)
    value.add_argument("--rank-growth-count", type=int, default=4)
    value.add_argument("--rank-residual-target", type=float, default=1e-4)
    value.add_argument("--rank-patience", type=int, default=2)
    value.add_argument("--rank-rollback-factor", type=float, default=1.05)
    value.add_argument("--rank-probation-evals", type=int, default=3)
    value.add_argument("--dynamic-jacobian-growth",
                       action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--jacobian-directions-per-growth", type=int, default=1)
    value.add_argument("--steps", type=int, default=3000)
    value.add_argument("--batch", type=int, default=2048)
    value.add_argument("--validation", type=int, default=4096)
    value.add_argument("--evaluation", type=int, default=16384)
    value.add_argument("--boundary-batch", type=int, default=256)
    value.add_argument("--boundary-weight", type=float, default=10.0)
    value.add_argument("--spectral-points-per-axis", type=int, default=128)
    value.add_argument("--lr", type=float, default=2e-3)
    value.add_argument("--grad-clip", type=float, default=10.0)
    value.add_argument("--eval-every", type=int, default=100)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--resume", type=Path, default=None,
                       help="continue from a physics-selected model/bank state")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
