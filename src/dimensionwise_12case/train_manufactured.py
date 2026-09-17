#!/usr/bin/env python3
"""Train H1--H4/K1--K4 with one dimension-wise low-rank framework."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from cases import get_case, parameterize_case
from model import DimensionWiseCP


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/dimensionwise_12case"


def with_amplitude(case, amplitude: float):
    """Create a parameter-family member without using interior target data."""
    if abs(amplitude-1.0) < 1e-12:
        return case
    base_exact, base_forcing = case.exact, case.forcing

    def exact(x):
        return amplitude*base_exact(x)

    if case.family == "helmholtz":
        def forcing(x):
            return amplitude*base_forcing(x)
    else:
        def forcing(x):
            query = x.detach().clone().requires_grad_(True)
            value = exact(query)
            first = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
            second = [torch.autograd.grad(
                first[:, axis].sum(), query, create_graph=True
            )[0][:, axis] for axis in range(case.dimension)]
            result = second[case.time_axis]-sum(
                item for axis, item in enumerate(second)
                if axis != case.time_axis
            )+value.square()
            return result.detach()
    return replace(case, exact=exact, forcing=forcing)


def build_target_case(args):
    """Build one transfer task while preserving old amplitude experiments."""
    case = get_case(args.case)
    task_value = args.task_value
    if args.task_kind == "amplitude":
        task_value = args.amplitude if task_value is None else task_value
        case = with_amplitude(case, task_value)
    else:
        if task_value is None:
            raise ValueError("--task-value is required for non-amplitude tasks")
        if abs(args.amplitude-1.0) > 1e-12:
            raise ValueError("do not combine --amplitude with a non-amplitude task")
        case = parameterize_case(case, args.task_kind, task_value)
    return case, float(task_value)


def sample(bounds, count, device, requires_grad=True):
    unit = torch.rand(count, len(bounds), device=device)
    lower = torch.tensor([value[0] for value in bounds], device=device)
    upper = torch.tensor([value[1] for value in bounds], device=device)
    return (lower + unit * (upper - lower)).requires_grad_(requires_grad)


def midpoint_tensor_grid(bounds, points_per_axis, device):
    """Deterministic axis-balanced grid used by the spectral PDE objective."""
    coordinates = []
    for lower, upper in bounds:
        unit = (torch.arange(points_per_axis, device=device)+0.5)/points_per_axis
        coordinates.append(lower+unit*(upper-lower))
    return torch.cartesian_prod(*coordinates).reshape(-1, len(bounds))


def second_derivatives(value, x):
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = []
    for axis in range(x.shape[1]):
        second.append(torch.autograd.grad(first[:, axis].sum(), x, create_graph=True)[0][:, axis])
    return second


def residual(case, model, x):
    # The proposed dictionary model exposes analytic diagonal derivatives.
    # Dense and neural SPINN baselines use the ordinary autograd path; keeping
    # both here lets every representation share exactly the same PDE operator.
    if hasattr(model, "forward_with_diagonal_second"):
        prediction, second = model.forward_with_diagonal_second(x)
    else:
        prediction = model(x)
        second = second_derivatives(prediction, x)
    if case.family == "helmholtz":
        coefficient = (
            case.coefficient(x) if case.coefficient is not None
            else torch.ones_like(prediction)
        )
        operator = sum(second) + coefficient * prediction
    else:
        operator = second[case.time_axis] - sum(
            value for axis, value in enumerate(second) if axis != case.time_axis
        ) + prediction.square()
    forcing = case.forcing(x)
    return prediction, operator - forcing, operator, forcing


def helmholtz_spectral_loss(case, model, points_per_axis):
    """PDE-only inverse-symbol-preconditioned residual for 2-D Helmholtz.

    Projecting the strong residual onto the fixed sine dictionary and dividing
    each component by its Helmholtz eigenvalue produces a relative solution-
    error surrogate.  This removes the O(k^2) bias that otherwise makes Adam
    fit high-frequency residuals with poorly conditioned coefficient updates.
    No reference solution or interior data are used.
    """
    if case.dimension != 2 or any(
        kind != "sine_pi" for kind in case.feature_kinds
    ):
        raise ValueError("spectral Helmholtz loss currently requires 2-D sine dictionaries")
    device = next(model.parameters()).device
    x = midpoint_tensor_grid(case.bounds, points_per_axis, device)
    _, equation_residual, _, forcing = residual(case, model, x)
    q = points_per_axis
    residual_grid = equation_residual.reshape(q, q)
    forcing_grid = forcing.reshape(q, q)
    bases = [branch.dictionary(x.new_tensor(
        [(case.bounds[axis][0]+(index+0.5)*(case.bounds[axis][1]-case.bounds[axis][0])/q)
         for index in range(q)]
    )) for axis, branch in enumerate(model.branches)]
    # The common normalization cancels in the relative objective.
    residual_modes = bases[0].T@residual_grid@bases[1]/float(q*q)
    forcing_modes = bases[0].T@forcing_grid@bases[1]/float(q*q)
    k0 = torch.arange(1, bases[0].shape[1]+1, device=device)[:, None]
    k1 = torch.arange(1, bases[1].shape[1]+1, device=device)[None, :]
    eigenvalue = 1.0-(torch.pi*k0).square()-(torch.pi*k1).square()
    preconditioned = residual_modes/eigenvalue
    forcing_solution = forcing_modes/eigenvalue
    return (preconditioned.square().sum()
            /forcing_solution.detach().square().sum().clamp_min(1e-12))


def boundary_initial_loss(case, model, count, device):
    if case.family == "helmholtz":
        # sine dictionaries impose homogeneous boundaries exactly.
        return torch.zeros((), device=device)
    dimension = case.dimension
    time_axis = case.time_axis
    spatial_axes = [axis for axis in range(dimension) if axis != time_axis]
    per_face = max(8, count // (2 * len(spatial_axes) + 2))
    losses = []
    for axis in spatial_axes:
        for endpoint in case.bounds[axis]:
            x = sample(case.bounds, per_face, device, requires_grad=False)
            x[:, axis] = endpoint
            losses.append((model(x) - case.exact(x)).square().mean())
    x0 = sample(case.bounds, per_face * 2, device, requires_grad=True)
    x0_data = x0.detach().clone()
    x0_data[:, time_axis] = case.bounds[time_axis][0]
    x0 = x0_data.requires_grad_(True)
    pred0 = model(x0)
    exact0_graph = case.exact(x0)
    pred_dt = torch.autograd.grad(pred0.sum(), x0, create_graph=True)[0][:, time_axis]
    exact_dt = torch.autograd.grad(
        exact0_graph.sum(), x0, create_graph=False
    )[0][:, time_axis].detach()
    exact0 = exact0_graph.detach()
    losses.extend(((pred0 - exact0).square().mean(), (pred_dt - exact_dt).square().mean()))
    return torch.stack(losses).mean()


def deterministic_boundary_initial_loss(case, model, count, device, seed):
    """Evaluate constraints on a reproducible batch (needed by L-BFGS)."""
    cuda_devices = ([device.index if device.index is not None else 0]
                    if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        return boundary_initial_loss(case, model, count, device)


@torch.no_grad()
def metrics(case, model, count, device, seed=20260724):
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    x = sample(case.bounds, count, device, requires_grad=False)
    reference = case.exact(x)
    prediction = model(x)
    relative = torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    torch.random.set_rng_state(generator_state)
    return {"relative_l2": float(relative), "max_abs": float((prediction-reference).abs().max())}


def derivative_unit_test(case, device):
    x = sample(case.bounds, 128, device, requires_grad=True)
    exact = case.exact(x)
    second = second_derivatives(exact, x)
    if case.family == "helmholtz":
        coefficient = (
            case.coefficient(x) if case.coefficient is not None
            else torch.ones_like(exact)
        )
        reconstructed = sum(second) + coefficient * exact
    else:
        reconstructed = second[case.time_axis] - sum(
            value for axis, value in enumerate(second) if axis != case.time_axis
        ) + exact.square()
    error = float((reconstructed - case.forcing(x)).abs().max())
    if error > 2e-3:
        raise RuntimeError(f"manufactured forcing check failed: {error}")
    return error


def load_mapping(model, path, initialization, parameter):
    """Replace random maps by a learned frozen bank and initialize latents."""
    if not path:
        return None
    bank = torch.load(path, map_location="cpu", weights_only=False)
    for axis, branch in enumerate(model.branches):
        key = f"axis{axis}"
        learned = bank["mappings"][key]
        if branch.mapping.shape != learned.shape:
            raise ValueError(
                f"{key} Mapping shape {tuple(learned.shape)} != "
                f"model shape {tuple(branch.mapping.shape)}"
            )
        branch.mapping.copy_(learned.to(branch.mapping))
        if initialization == "mean":
            initial = bank["initial_latents"][key]
        elif initialization == "parameter":
            decoder = bank["parameter_decoders"][key]
            powers = torch.as_tensor(parameter).pow(torch.arange(
                decoder.shape[0], dtype=decoder.dtype
            ))
            initial = powers@decoder
        else:
            continue
        branch.latent.data.copy_(initial.to(branch.latent))
    return bank


def configure_adaptive(model, fraction, count=0):
    for branch in model.branches:
        if count > 0:
            branch.active_latent_size.fill_(min(count, branch.latent.numel()))
        else:
            branch.set_active_latent_fraction(fraction)


def grow_adaptive(model, fraction, count=0):
    grew = False
    for branch in model.branches:
        if count > 0:
            old = int(branch.active_latent_size)
            new = min(branch.latent.numel(), old+count)
            branch.active_latent_size.fill_(new)
        else:
            old, new = branch.grow_active_latent(fraction)
        grew = grew or new > old
    return grew


def active_latent_count(model):
    return sum(branch.active_latent_count for branch in model.branches)


def maximum_latent_count(model):
    return sum(branch.latent.numel() for branch in model.branches
               if branch.latent is not None)


def assert_strict_latent_only(model, mapping_bank):
    """Fail loudly if a main-method target can update anything but latents."""
    if mapping_bank is None:
        return []
    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    illegal = [name for name in trainable if not name.endswith(".latent")]
    if illegal:
        raise RuntimeError(f"strict latent-only contract violated by {illegal}")
    if not trainable:
        raise RuntimeError("mapping target has no trainable latent vectors")
    return trainable


def validation_pde(case, model, x):
    _, equation_residual, operator, forcing = residual(case, model, x)
    scale = (operator.detach().square().mean()
             +forcing.detach().square().mean()).clamp_min(1e-8)
    return equation_residual.square().mean()/scale


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    case, task_value = build_target_case(args)
    if args.cp_rank is not None:
        case = replace(case, rank=args.cp_rank)
    source_error = derivative_unit_test(case, device)
    model = DimensionWiseCP(case, args.variant, args.latent_fraction, args.seed).to(device)
    mapping_bank = load_mapping(
        model, args.mapping_bank, args.mapping_init, task_value
    )
    target_trainable_names = assert_strict_latent_only(model, mapping_bank)
    if args.adaptive_latent_rank:
        if mapping_bank is None or args.variant != "latent_compressed":
            raise ValueError("adaptive latent rank requires a compressed Mapping bank")
        configure_adaptive(
            model, args.initial_latent_fraction, args.initial_latent_count
        )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=args.lr * 0.05)
    run_variant = args.variant
    if args.task_kind != "amplitude" or abs(task_value-1.0) > 1e-12:
        run_variant += f"_{args.task_kind}_{task_value:g}"
    if mapping_bank is not None:
        run_variant += f"_{Path(args.mapping_bank).stem}_{args.mapping_init}init"
    if args.adaptive_latent_rank:
        run_variant += (f"_adaptive_{args.initial_latent_fraction:g}"
                        f"to{args.latent_fraction:g}")
        if args.initial_latent_count > 0:
            run_variant += f"_initcount{args.initial_latent_count}"
        if args.rank_growth_count > 0:
            run_variant += f"_growcount{args.rank_growth_count}"
    if args.tag:
        run_variant += f"_{args.tag}"
    out = OUTPUT / case.family / case.name / run_variant / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.jsonl"
    if history_path.exists() and not args.overwrite:
        raise FileExistsError(out)
    history_path.write_text("")
    config = {**vars(args), "output": str(out), "source_unit_test_max_abs": source_error,
              "trainable_parameters": model.trainable_count, "fixed_parameters": model.fixed_count,
              "target_trainable_names": target_trainable_names,
              "strict_latent_only": mapping_bank is not None,
              "reference_role": "evaluation_only", "dataloss": False}
    (out / "configuration.json").write_text(json.dumps(config, indent=2, default=str))
    best_physics = math.inf
    best_row = None
    best_state = None
    rank_window_best = math.inf
    rank_stale_evaluations = 0
    validation_x = sample(
        case.bounds, args.physics_validation, device, requires_grad=True
    )
    started = time.perf_counter()
    for step in range(args.steps + 1):
        x = sample(case.bounds, args.collocation, device, requires_grad=False)
        prediction, equation_residual, operator, forcing = residual(case, model, x)
        scale = (operator.detach().square().mean() + forcing.detach().square().mean()).clamp_min(1e-8)
        pointwise_pde = equation_residual.square().mean()/scale
        if case.family == "helmholtz" and args.helmholtz_loss != "pointwise":
            spectral_pde = helmholtz_spectral_loss(
                case, model, args.spectral_points_per_axis
            )
            if args.helmholtz_loss == "spectral_preconditioned":
                pde = spectral_pde
            else:
                pde = spectral_pde+args.pointwise_weight*pointwise_pde
        else:
            spectral_pde = None
            pde = pointwise_pde
        constraint = boundary_initial_loss(case, model, args.boundary, device)
        loss = pde + args.constraint_weight * constraint
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            score = metrics(case, model, args.evaluation, device)
            validation_score = validation_pde(case, model, validation_x)
            row = {"step": step, "loss": float(loss.detach()), "pde": float(pde.detach()),
                   "pointwise_pde": float(pointwise_pde.detach()),
                   "spectral_pde": (None if spectral_pde is None
                                    else float(spectral_pde.detach())),
                   "constraint": float(constraint.detach()),
                   "physics_validation": float(validation_score.detach()),
                   "active_latent_rank": active_latent_count(model),
                   "maximum_latent_rank": maximum_latent_count(model), **score,
                   "seconds": time.perf_counter()-started}
            with history_path.open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            physics_score = (row["physics_validation"]
                             +args.constraint_weight*row["constraint"])
            if physics_score < best_physics:
                best_physics, best_row = physics_score, row
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"model": model.state_dict(), "row": row, "configuration": config}, out / "best_physics.pt")
            if (args.adaptive_latent_rank and 0 < step < args.steps
                    and row["physics_validation"] > args.rank_residual_target):
                relative_gain = ((rank_window_best-row["physics_validation"])
                                 /max(abs(rank_window_best), 1e-12))
                if (math.isfinite(rank_window_best)
                        and relative_gain < args.rank_min_relative_gain):
                    rank_stale_evaluations += 1
                else:
                    rank_stale_evaluations = 0
                rank_window_best = min(rank_window_best,
                                       row["physics_validation"])
                if rank_stale_evaluations >= args.rank_patience:
                    grew = grow_adaptive(
                        model, args.rank_growth_fraction, args.rank_growth_count
                    )
                    print(json.dumps({
                        "step": step,
                        "rank_event": "grow" if grew else "at_maximum",
                        "active_latent_rank": active_latent_count(model),
                        "maximum_latent_rank": maximum_latent_count(model),
                        "trigger_physics_validation": row["physics_validation"],
                    }), flush=True)
                    rank_window_best = math.inf
                    rank_stale_evaluations = 0
            physics_score = (row["physics_validation"]
                             +args.constraint_weight*row["constraint"])
            if (args.physics_stop > 0 and step >= args.minimum_steps
                    and physics_score <= args.physics_stop):
                print(json.dumps({
                    "step": step, "stop_event": "physics_target_reached",
                    "physics_score": physics_score,
                    "physics_target": args.physics_stop,
                }), flush=True)
                break
    if args.lbfgs_steps > 0:
        # A fixed physics batch makes the deterministic quasi-Newton closure
        # meaningful and is especially useful after adaptive rank expansion.
        polish_x = sample(case.bounds, args.lbfgs_collocation, device,
                          requires_grad=False)
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=args.lbfgs_lr,
            max_iter=args.lbfgs_steps, history_size=args.lbfgs_history,
            line_search_fn="strong_wolfe",
        )
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            _, equation_residual, operator, forcing = residual(case, model, polish_x)
            scale = (operator.detach().square().mean()
                     +forcing.detach().square().mean()).clamp_min(1e-8)
            pointwise = equation_residual.square().mean()/scale
            if case.family == "helmholtz" and args.helmholtz_loss != "pointwise":
                spectral = helmholtz_spectral_loss(
                    case, model, args.spectral_points_per_axis
                )
                objective = (spectral if args.helmholtz_loss == "spectral_preconditioned"
                             else spectral+args.pointwise_weight*pointwise)
            else:
                objective = pointwise+args.constraint_weight*deterministic_boundary_initial_loss(
                    case, model, args.boundary, device, args.seed+700001
                )
            objective.backward()
            return objective
        lbfgs.step(closure)
        validation_score = validation_pde(case, model, validation_x)
        polish_constraint = deterministic_boundary_initial_loss(
            case, model, args.boundary, device, args.seed+700001
        )
        score = metrics(case, model, args.evaluation, device)
        polish_row = {
            "step": args.steps+args.lbfgs_steps,
            "phase": "lbfgs", "physics_validation": float(validation_score),
            "constraint": float(polish_constraint.detach()),
            "active_latent_rank": active_latent_count(model),
            "maximum_latent_rank": maximum_latent_count(model), **score,
            "seconds": time.perf_counter()-started,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(polish_row)+"\n")
        print(json.dumps(polish_row), flush=True)
        polish_physics = (polish_row["physics_validation"]
                          +args.constraint_weight*polish_row["constraint"])
        if polish_physics < best_physics:
            best_physics = polish_physics
            best_row = {**polish_row, "pde": polish_row["physics_validation"],
                        "constraint": polish_row["constraint"]}
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": model.state_dict(), "row": best_row,
                        "configuration": config}, out/"best_physics.pt")
    model.load_state_dict(best_state)
    summary = {"case": case.name, "family": case.family, "variant": run_variant,
               "seed": args.seed, "best_step": best_row["step"],
               "physics_selected_relative_l2": best_row["relative_l2"],
               "physics_score": best_physics, "pde": best_row["pde"],
               "constraint": best_row["constraint"], "success": best_row["relative_l2"] < 0.05,
               "physics_validation": best_row["physics_validation"],
               "amplitude": (task_value if args.task_kind == "amplitude" else 1.0),
               "task_kind": args.task_kind, "task_value": task_value,
               "adaptive_latent_rank": args.adaptive_latent_rank,
               "selected_latent_rank": active_latent_count(model),
               "maximum_latent_rank": maximum_latent_count(model),
               "trainable_parameters": model.trainable_count, "fixed_parameters": model.fixed_count,
               "strict_latent_only": mapping_bank is not None,
               "target_trainable_names": target_trainable_names,
               "reference_role": "evaluation_only", "dataloss": False,
               "seconds": time.perf_counter()-started, "output": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--variant", choices=(
        "trainable_coefficients", "latent_fullspan", "latent_compressed",
        "latent_fullspan_shared", "latent_compressed_shared",
        "latent_fullspan_shared_random", "latent_compressed_shared_random",
        "latent_fullspan_sharedmap", "latent_compressed_sharedmap",
        "latent_fullspan_aligned", "latent_compressed_aligned",
        "latent_fullspan_implicit",
    ), default="latent_fullspan_shared")
    p.add_argument("--latent-fraction", type=float, default=0.25)
    p.add_argument("--cp-rank", type=int, default=None,
                   help="override the CP rank for fixed/adaptive rank studies")
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--task-kind", choices=(
        "amplitude", "frequency", "coefficient", "structure"
    ), default="amplitude",
                   help="held-out task transformation; amplitude is legacy")
    p.add_argument("--task-value", type=float, default=None,
                   help="frequency, coefficient, structure mask, or amplitude")
    p.add_argument("--mapping-bank", default="")
    p.add_argument("--mapping-init", choices=("zero", "mean", "parameter"),
                   default="zero")
    p.add_argument("--adaptive-latent-rank", action="store_true")
    p.add_argument("--initial-latent-fraction", type=float, default=0.05)
    p.add_argument("--initial-latent-count", type=int, default=0,
                   help="if positive, activate this many bank columns per branch")
    p.add_argument("--rank-growth-fraction", type=float, default=0.05)
    p.add_argument("--rank-growth-count", type=int, default=0,
                   help="if positive, add this many columns per branch per event")
    p.add_argument("--rank-residual-target", type=float, default=1e-4)
    p.add_argument("--rank-patience", type=int, default=1)
    p.add_argument("--rank-min-relative-gain", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--collocation", type=int, default=2048)
    p.add_argument("--boundary", type=int, default=1024)
    p.add_argument("--evaluation", type=int, default=16384)
    p.add_argument("--physics-validation", type=int, default=2048)
    p.add_argument("--constraint-weight", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--tag", default="")
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--helmholtz-loss", choices=(
        "pointwise", "spectral_preconditioned", "hybrid"
    ), default="pointwise")
    p.add_argument("--spectral-points-per-axis", type=int, default=32)
    p.add_argument("--pointwise-weight", type=float, default=0.05)
    p.add_argument("--lbfgs-steps", type=int, default=0)
    p.add_argument("--lbfgs-lr", type=float, default=0.5)
    p.add_argument("--lbfgs-history", type=int, default=50)
    p.add_argument("--lbfgs-collocation", type=int, default=4096)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--physics-stop", type=float, default=0.0,
                   help="physics-only early-stop threshold; zero disables it")
    p.add_argument("--minimum-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
