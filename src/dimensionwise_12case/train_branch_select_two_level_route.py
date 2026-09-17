#!/usr/bin/env python3
"""Automatic branch-profile selection plus two-level CP capacity routing.

The probe compares compact, physics-aligned and rich per-axis dictionaries on
identical physics batches.  The winning profile is randomly reinitialized and
trained once with dictionary-atom gates followed by complete CP-rank gates.
References are evaluation-only and never affect selection or checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cases import get_case, parameterize_case
from model import DimensionWiseCP
from train_manufactured import (
    boundary_initial_loss, deterministic_boundary_initial_loss,
    helmholtz_spectral_loss, metrics, residual, sample,
)

from common.two_level_capacity_routing import RoutedDimensionWiseCP


OUTPUT = ROOT / "results" / "multiphysics_capacity_routing"
PROFILES = ("compact", "aligned", "rich")


def profile_case(case, profile: str):
    modes = case.max_modes
    if profile == "compact":
        selected = tuple(max(2, int(math.ceil(value * 0.5))) for value in modes)
    elif profile == "aligned":
        selected = modes
    elif profile == "rich":
        selected = tuple(max(value, int(math.ceil(value * 1.5))) for value in modes)
    else:
        raise ValueError(profile)
    return replace(case, max_modes=selected)


def fixed_profile_case(case, profile: str):
    """Resolve a predeclared main-table profile without running a probe."""
    if profile == "dimension_specialized":
        return case
    return profile_case(case, profile)


def objective(case, model, x, args, deterministic=False):
    _, equation_residual, operator, forcing = residual(case, model, x)
    scale = (operator.detach().square().mean()
             + forcing.detach().square().mean()).clamp_min(1e-8)
    pointwise = equation_residual.square().mean() / scale
    if (case.family == "helmholtz" and args.helmholtz_loss != "pointwise"
            and tuple(case.feature_kinds) == ("sine_pi", "sine_pi")):
        spectral = helmholtz_spectral_loss(
            case, model, args.spectral_points_per_axis
        )
        pde = (spectral if args.helmholtz_loss == "spectral_preconditioned"
               else spectral + args.pointwise_weight * pointwise)
    else:
        pde = pointwise
    constraint = (deterministic_boundary_initial_loss(
        case, model, args.boundary, x.device, args.seed + 700001,
    ) if deterministic else boundary_initial_loss(
        case, model, args.boundary, x.device
    ))
    return pde + args.constraint_weight * constraint, pde, constraint


def build(case, seed, routed=False, gate_initial_log_alpha=3.0,
          capacity_routing="two_level"):
    base = DimensionWiseCP(
        case, "trainable_coefficients", latent_fraction=1.0, seed=seed
    )
    route_atoms = capacity_routing in {"atom", "two_level"}
    route_ranks = capacity_routing in {"rank", "two_level"}
    return (RoutedDimensionWiseCP(base, gate_initial_log_alpha,
                                  route_atoms, route_ranks)
            if routed else base)


def probe(case, args, device, validation_x):
    candidates = {}
    for profile in PROFILES:
        current_case = profile_case(case, profile)
        model = build(current_case, args.seed + 101, routed=False).to(device)
        candidates[profile] = {
            "case": current_case,
            "model": model,
            "optimizer": torch.optim.Adam(model.parameters(), lr=args.lr),
            "initial": None, "seconds": 0.0,
        }
    for step in range(args.probe_steps + 1):
        # Resetting this generator makes all candidates see identical points.
        generator_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        batches = {}
        for profile in PROFILES:
            torch.random.set_rng_state(generator_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
            batches[profile] = sample(
                candidates[profile]["case"].bounds, args.collocation,
                device, requires_grad=False,
            )
        for profile, candidate in candidates.items():
            started = time.perf_counter()
            loss = objective(candidate["case"], candidate["model"],
                             batches[profile], args)[0]
            if step:
                candidate["optimizer"].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    candidate["model"].parameters(), args.grad_clip
                )
                candidate["optimizer"].step()
            candidate["seconds"] += time.perf_counter() - started
        if step in (0, args.probe_steps):
            for candidate in candidates.values():
                physics = float(objective(
                    candidate["case"], candidate["model"], validation_x,
                    args, True,
                )[0].detach())
                if step == 0:
                    candidate["initial"] = physics
                candidate["physics"] = physics
    median_time = sorted(v["seconds"] for v in candidates.values())[1]
    median_params = sorted(sum(p.numel() for p in v["model"].parameters())
                           for v in candidates.values())[1]
    rows = []
    for profile, candidate in candidates.items():
        parameters = sum(p.numel() for p in candidate["model"].parameters())
        # Unlike the neutron candidates, different fixed dictionaries cannot
        # all represent the same random initial function.  Selection therefore
        # ranks the absolute locked physical residual first and uses cost only
        # as a mild tie breaker; a large relative gain from an unusable initial
        # state must not beat a branch that actually satisfies the PDE.
        cost = (candidate["seconds"] / median_time
                + parameters / median_params)
        score = (-math.log(max(candidate["physics"], 1e-20))
                 - args.probe_parameter_penalty * cost)
        rows.append({
            "profile": profile,
            "modes": candidate["case"].max_modes,
            "initial_physics": candidate["initial"],
            "final_physics": candidate["physics"],
            "seconds": candidate["seconds"],
            "parameters": parameters,
            "selection_score": score,
        })
    winner = max(rows, key=lambda row: row["selection_score"])["profile"]
    return winner, rows


def phase_at(step, args):
    if step < args.route_start:
        return "warmup", 0.0
    if step < args.consolidate_step:
        strength = ((step - args.route_start)
                    / max(1, args.consolidate_step - args.route_start))
        return "route", strength
    return "refine", 1.0


def run(args):
    if args.deterministic_formal:
        # Required by the held-out Auto/oracle audit: otherwise independent
        # CUDA reductions can make two copies of the same selected profile
        # choose different physics checkpoints, obscuring selection regret.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    original = get_case(args.case)
    if args.task_kind != "identity":
        if args.task_value is None:
            raise ValueError("--task-value is required for a parameterized task")
        original = parameterize_case(original, args.task_kind, args.task_value)
    if args.cp_rank is not None:
        if args.cp_rank < 1:
            raise ValueError("--cp-rank must be positive")
        original = replace(original, rank=args.cp_rank)
    validation_x = sample(
        original.bounds, args.physics_validation, device, requires_grad=True
    )
    if args.profile_selection == "auto":
        winner, probe_rows = probe(original, args, device, validation_x)
        case = profile_case(original, winner)
    elif args.profile_selection.startswith("fixed_"):
        fixed = args.profile_selection.removeprefix("fixed_")
        winner, probe_rows = f"fixed_{fixed}", []
        case = fixed_profile_case(original, fixed)
    else:
        winner, probe_rows = "generic_fourier_all_axes", []
        case = replace(
            original,
            feature_kinds=tuple("fourier_periodic" for _ in original.feature_kinds),
        )
    if args.selection_only:
        output = (args.output / case.family / case.name / args.variant /
                  f"seed_{args.seed}")
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "family": case.family, "case": case.name,
            "mode": "physics_only_branch_selection_stability",
            "selected_profile": winner, "selected_modes": case.max_modes,
            "probe": probe_rows, "task_kind": args.task_kind,
            "task_value": args.task_value, "seed": args.seed,
            "reference_role": "evaluation_only", "dataloss": False,
            "output": str(output),
        }
        (output / "configuration.json").write_text(
            json.dumps({**vars(args), **payload}, indent=2, default=str)
        )
        (output / "summary.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2)); return
    # Probing must not change the stochastic formal-training trajectory.  In
    # particular, Auto and the corresponding fixed-profile oracle candidate
    # must see exactly the same collocation sequence under a common formal
    # seed.  Reset after selection (and for fixed profiles at the same point)
    # so probe RNG consumption cannot leak into the comparison.
    formal_seed = args.seed + 900001
    torch.manual_seed(formal_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(formal_seed)
    model = build(case, args.seed, routed=True,
                  gate_initial_log_alpha=args.gate_initial_log_alpha,
                  capacity_routing=args.capacity_routing).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=5, min_lr=args.lr * 0.02
    )
    output = (args.output / case.family / case.name / args.variant
              / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(output)
    history_path = output / "history.jsonl"
    history_path.write_text("")
    configuration = {
        **vars(args), "output": str(output), "selected_profile": winner,
        "selected_modes": case.max_modes, "probe": probe_rows,
        "reference_role": "evaluation_only", "dataloss": False,
        "capacity_routing_ablation": args.capacity_routing,
        "formal_sampling_seed": formal_seed,
        "formal_sampling_seed_reset_after_probe": True,
        "routing_layers": ["per_axis_dictionary_atoms", "CP_ranks"],
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2, default=str)
    )
    best_physics, best_row, best_state = math.inf, None, None
    minimum_rel = math.inf
    consolidated = None
    started = time.perf_counter()
    for step in range(args.steps + 1):
        phase, strength = phase_at(step, args)
        model.set_phase(phase, strength)
        if step == args.consolidate_step:
            consolidated = model.consolidate(
                args.gate_threshold, args.minimum_ranks,
                args.minimum_atoms_per_rank,
            )
            # Refine only surviving physical coefficients; gates are frozen.
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=(args.lr if args.capacity_routing == "none" else args.lr * 0.5),
            )
        model.begin_routing_step()
        x = sample(case.bounds, args.collocation, device, requires_grad=False)
        physical, pde, constraint = objective(case, model, x, args)
        capacity = (model.capacity_penalty()
                    if phase == "route" else physical.new_zeros(()))
        loss = physical + args.capacity_weight * strength * capacity
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        if step in (0, 1) or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            locked = float(objective(
                case, model, validation_x, args, True
            )[0].detach())
            score = metrics(case, model, args.evaluation, device)
            model.train()
            minimum_rel = min(minimum_rel, score["relative_l2"])
            row = {
                "step": step, "phase": phase,
                "loss": float(loss.detach()), "pde": float(pde.detach()),
                "constraint": float(constraint.detach()),
                "capacity_penalty": float(capacity.detach()),
                "physics_validation": locked,
                "relative_l2_evaluation_only": score["relative_l2"],
                "routing": model.routing_statistics(),
                "seconds": time.perf_counter() - started,
            }
            with history_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if step >= args.consolidate_step:
                scheduler.step(locked)
                if locked < best_physics:
                    best_physics, best_row = locked, row
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save({"model": best_state, "row": row,
                                "configuration": configuration},
                               output / "best_physics.pt")
    if best_row is None:
        raise RuntimeError("no post-consolidation checkpoint was evaluated")
    model.load_state_dict(best_state)
    summary = {
        "family": case.family, "case": case.name,
        "mode": ("fixed_dimension_specialized_low_rank"
                 if args.profile_selection.startswith("fixed_")
                 and args.capacity_routing == "none" else
                 "branch_select_two_level_capacity_routing"),
        "selected_profile": winner, "selected_modes": case.max_modes,
        "probe": probe_rows, "routing": consolidated,
        "full_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "main_branch_trainable_parameters": model.base.trainable_count,
        "effective_active_parameters": model.effective_parameter_count(),
        "best_step": best_row["step"],
        "best_physics_validation": best_physics,
        "physics_selected_relative_l2": best_row["relative_l2_evaluation_only"],
        "minimum_monitored_relative_l2": minimum_rel,
        "seconds": time.perf_counter() - started,
        "reference_role": "evaluation_only", "dataloss": False,
        "output": str(output),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("H1", "H2", "H3", "H4", "K1", "K2", "K3", "K4"), required=True)
    p.add_argument("--task-kind", default="identity",
                   choices=("identity", "frequency", "coefficient", "structure"))
    p.add_argument("--task-value", type=float)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument(
        "--cp-rank", type=int, default=None,
        help="override the benchmark CP rank for a controlled fixed-rank run",
    )
    p.add_argument("--probe-steps", type=int, default=300)
    p.add_argument("--deterministic-formal", action="store_true")
    p.add_argument("--collocation", type=int, default=2048)
    p.add_argument("--boundary", type=int, default=1024)
    p.add_argument("--evaluation", type=int, default=16384)
    p.add_argument("--physics-validation", type=int, default=2048)
    p.add_argument("--constraint-weight", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--helmholtz-loss", choices=("pointwise", "spectral_preconditioned", "hybrid"), default="hybrid")
    p.add_argument("--spectral-points-per-axis", type=int, default=32)
    p.add_argument("--pointwise-weight", type=float, default=0.05)
    p.add_argument("--probe-parameter-penalty", type=float, default=0.1)
    p.add_argument("--profile-selection",
                   choices=("auto", "fixed_compact", "fixed_aligned",
                            "fixed_rich", "fixed_dimension_specialized",
                            "generic_fourier"),
                   default="auto")
    p.add_argument("--route-start", type=int, default=600)
    p.add_argument("--consolidate-step", type=int, default=2100)
    p.add_argument("--capacity-weight", type=float, default=1e-3)
    p.add_argument("--capacity-routing", choices=("none", "atom", "rank", "two_level"),
                   default="two_level")
    p.add_argument("--gate-initial-log-alpha", type=float, default=3.0)
    p.add_argument("--gate-threshold", type=float, default=0.95)
    p.add_argument("--minimum-ranks", type=int, default=2)
    p.add_argument("--minimum-atoms-per-rank", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--variant", default="branch_select_route_v1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--selection-only", action="store_true",
                   help="stop after physics-only branch probing")
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.smoke:
        args.steps = 8; args.probe_steps = 2
        args.collocation = 32; args.boundary = 32
        args.evaluation = 64; args.physics_validation = 32
        args.route_start = 2; args.consolidate_step = 5; args.eval_every = 1
    run(args)
