#!/usr/bin/env python3
"""Physics-only formal H2 benchmark and equal-capacity branch-selection audit.

This file is deliberately independent of the paper-table builders.  It serves
two review purposes:

1. run the promoted off-dictionary, variable-coefficient H2 problem (formerly
   the internal HX2 stress test) with the baseline suite used in the
   manuscript;
2. compare complete per-axis branch profiles while every sine branch exposes
   exactly 64 modes.  No compact/aligned/rich capacity labels are used.

Reference fields are evaluated only after physics checkpoint selection.  They
never enter training, profile probing, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from baseline_models import (
    build_baseline, fit_rfm_linear_pde, parameter_counts,
)
from model import DimensionWiseCP, FixedFeatureDictionary
from cases import get_case as get_formal_case
from train_helmholtz_challenges import get_case as get_challenge_case, sample


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/reviewer_followup_20260815/hx2_branch64_audit"
BASELINE_METHODS = {
    "vanilla": "vanilla_mlp_hardbc",
    "ms_fourier": "fourier_mlp_hardbc",
    "spinn": "spinn_style_hardbc",
    "cp_pinn": "cp_pinn_hardbc",
    "piratenet": "piratenet_hardbc",
    "rfm": "rfm_hardbc",
}
BRANCH_PROFILES = {
    "sine_sine": ("sine64", "sine64"),
    "sine_mlp": ("sine64", "mlp"),
    "mlp_sine": ("mlp", "sine64"),
    "mlp_mlp": ("mlp", "mlp"),
}


def get_case(name: str):
    """Resolve the promoted formal H2 while retaining internal HX audits."""
    return get_formal_case(name) if name == "H2" else get_challenge_case(name)


def configured_case(args):
    case = get_case(args.case)
    return replace(case, rank=args.rank) if args.rank is not None else case


def second_derivatives(value: torch.Tensor, points: torch.Tensor):
    first = torch.autograd.grad(value.sum(), points, create_graph=True)[0]
    return [
        torch.autograd.grad(
            first[:, axis].sum(), points, create_graph=True,
            retain_graph=True,
        )[0][:, axis]
        for axis in range(points.shape[1])
    ]


def equation_residual(case, model, points: torch.Tensor):
    if hasattr(model, "forward_with_diagonal_second"):
        prediction, second = model.forward_with_diagonal_second(points)
    else:
        points = points.detach().clone().requires_grad_(True)
        prediction = model(points)
        second = second_derivatives(prediction, points)
    operator = sum(second)+case.coefficient(points)*prediction
    forcing = case.forcing(points)
    equation = operator-forcing
    scale = (operator.detach().square().mean()
             +forcing.detach().square().mean()).clamp_min(1.0e-8)
    return prediction, equation, equation.square().mean()/scale


def inverse_symbol_score(case, model, device, points_per_axis=64):
    """Physics-only solution-scaled score used by branch probing."""
    q = int(points_per_axis)
    coordinate = (torch.arange(q, device=device, dtype=torch.float32)+0.5)/q
    xx, yy = torch.meshgrid(coordinate, coordinate, indexing="ij")
    points = torch.stack((xx, yy), dim=-1).reshape(-1, 2).requires_grad_(True)
    _, equation, _ = equation_residual(case, model, points)
    forcing = case.forcing(points).reshape(q, q)
    modes = torch.arange(1, q+1, device=device, dtype=points.dtype)[None]
    basis = torch.sin(torch.pi*coordinate[:, None]*modes)
    residual_modes = basis.T@equation.reshape(q, q)@basis/float(q*q)
    forcing_modes = basis.T@forcing@basis/float(q*q)
    mode0 = torch.arange(1, q+1, device=device, dtype=points.dtype)[:, None]
    mode1 = torch.arange(1, q+1, device=device, dtype=points.dtype)[None, :]
    symbol = 1.0-(torch.pi*mode0).square()-(torch.pi*mode1).square()
    return ((residual_modes/symbol).square().sum()
            /(forcing_modes.detach()/symbol).square().sum().clamp_min(1e-12))


@torch.no_grad()
def relative_l2(case, model, device, count=16384, seed=2026081501):
    cuda_devices = ([device.index or 0] if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        points = sample(count, device)
    prediction = model(points)
    reference = case.exact(points)
    return float(torch.linalg.vector_norm(prediction-reference)
                 /torch.linalg.vector_norm(reference).clamp_min(1.0e-12))


class SineAxis(nn.Module):
    """A trainable linear head on a locked 64-mode sine dictionary."""

    def __init__(self, bounds, rank: int):
        super().__init__()
        self.dictionary = FixedFeatureDictionary(
            "sine_pi", 64, bounds[0], bounds[1],
        )
        # Match the formal DimensionWiseCP initialization exactly:
        # 0.5/sqrt(dictionary size), followed by the locked 0.05 output-scale
        # contraction used by the Helmholtz runs.  A larger coefficient scale
        # disproportionately amplifies the residual of the highest modes and
        # would turn initialization, rather than branch type, into the probe.
        scale = 0.05*0.5/math.sqrt(self.dictionary.size)
        self.coefficient = nn.Parameter(scale*torch.randn(64, rank))

    def forward(self, coordinate):
        return self.dictionary(coordinate)@self.coefficient


class MLPAxis(nn.Module):
    """Trainable coordinate branch with the same CP rank and hard zero trace."""

    def __init__(self, bounds, rank: int, width=64, depth=3):
        super().__init__()
        self.lower, self.upper = float(bounds[0]), float(bounds[1])
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth-1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, rank))
        self.network = nn.Sequential(*layers)
        with torch.no_grad():
            self.network[-1].weight.mul_(0.05)
            self.network[-1].bias.zero_()

    def forward(self, coordinate):
        unit = (coordinate-self.lower)/(self.upper-self.lower)
        normalized = 2.0*unit-1.0
        # Applying the lift per axis preserves an actual separable CP field.
        return (4.0*unit*(1.0-unit))[:, None]*self.network(normalized[:, None])


class BranchProfileCP(nn.Module):
    """Two-axis CP model whose branch types, but not sine capacity, vary."""

    def __init__(self, case, profile: str):
        super().__init__()
        self.rank = case.rank
        kinds = BRANCH_PROFILES[profile]
        branches = []
        for axis, kind in enumerate(kinds):
            cls = SineAxis if kind == "sine64" else MLPAxis
            branches.append(cls(case.bounds[axis], self.rank))
        self.branches = nn.ModuleList(branches)

    def forward(self, points):
        product = torch.ones(
            points.shape[0], self.rank, device=points.device,
            dtype=points.dtype,
        )
        for axis, branch in enumerate(self.branches):
            product = product*branch(points[:, axis])
        return product.sum(dim=1)/math.sqrt(self.rank)


def build(case, method: str, seed: int, device):
    torch.manual_seed(seed)
    if method == "dimspec_sine64":
        configured = type(case)(
            **{**case.__dict__, "max_modes": (64, 64)}
        )
        model = DimensionWiseCP(
            configured, "trainable_coefficients", 1.0, seed,
        )
        with torch.no_grad():
            for branch in model.branches:
                branch.base.mul_(0.05)
                branch.coefficient.copy_(branch.base)
        return model.to(device), 0.02
    if method in BRANCH_PROFILES:
        return BranchProfileCP(case, method).to(device), 0.02
    if method in BASELINE_METHODS:
        model = build_baseline(BASELINE_METHODS[method], case).to(device)
        lr = 0.001 if method == "piratenet" else 0.01
        return model, lr
    raise ValueError(method)


def train(case, method: str, args, device):
    model, learning_rate = build(case, method, args.seed, device)
    trainable, fixed = parameter_counts(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=learning_rate*0.05,
    )
    prefit, prefit_seconds = None, 0.0
    if method == "rfm":
        started = time.perf_counter()
        torch.manual_seed(args.seed+700001)
        points = sample(max(4096, 2*args.collocation), device, True)
        prefit = fit_rfm_linear_pde(model, case, points)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefit_seconds = time.perf_counter()-started
    with torch.random.fork_rng(
        devices=[device.index or 0] if device.type == "cuda" else []
    ):
        torch.manual_seed(args.seed+991)
        validation_points = sample(args.physics_validation, device)
    best_physics, best_error, best_step, best_state = math.inf, math.inf, 0, None
    minimum_error = math.inf
    history = []
    started = time.perf_counter()
    actual_steps = 0 if prefit is not None else args.steps
    for step in range(actual_steps+1):
        torch.manual_seed(args.seed+10000+step)
        points = sample(args.collocation, device)
        _, _, loss = equation_residual(case, model, points)
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step(); scheduler.step()
        if step == 0 or step % args.eval_every == 0 or step == actual_steps:
            physics = float(equation_residual(
                case, model, validation_points,
            )[2].detach())
            error = relative_l2(case, model, device, args.evaluation)
            minimum_error = min(minimum_error, error)
            row = {
                "step": step, "physics_validation": physics,
                "relative_l2_evaluation_only": error,
                "seconds": prefit_seconds+time.perf_counter()-started,
            }
            history.append(row)
            print(json.dumps({"case": case.name, "method": method, **row}),
                  flush=True)
            if physics < best_physics:
                best_physics, best_error, best_step = physics, error, step
                best_state = copy.deepcopy(model.state_dict())
    return model, best_state, history, {
        "case": case.name, "method": method, "seed": args.seed,
        "best_step": best_step,
        "physics_validation": best_physics,
        "physics_selected_relative_l2": best_error,
        "minimum_monitored_relative_l2": minimum_error,
        "seconds": prefit_seconds+time.perf_counter()-started,
        "trainable_parameters": trainable, "fixed_parameters": fixed,
        "actual_optimizer_steps": actual_steps,
        "rfm_physics_collocation_prefit": prefit,
        "execution_role": "main_timing",
        "reference_role": "evaluation_only", "dataloss": False,
    }


def probe(case, args, device):
    candidates = {}
    with torch.random.fork_rng(
        devices=[device.index or 0] if device.type == "cuda" else []
    ):
        torch.manual_seed(args.seed+991)
        validation_points = sample(args.physics_validation, device)
    for profile in BRANCH_PROFILES:
        model, lr = build(case, profile, args.seed+101, device)
        candidates[profile] = {
            "model": model,
            "optimizer": torch.optim.Adam(model.parameters(), lr=lr),
            "scheduler": None,
            "seconds": 0.0,
        }
        candidates[profile]["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
            candidates[profile]["optimizer"], T_max=max(1, args.probe_steps),
            eta_min=lr*0.05,
        )
    checkpoints = set(args.probe_checkpoints)
    curve = []
    for step in range(1, args.probe_steps+1):
        torch.manual_seed(args.seed+20000+step)
        points = sample(args.probe_collocation, device)
        for item in candidates.values():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss = equation_residual(case, item["model"], points)[2]
            item["optimizer"].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(item["model"].parameters(), 100.0)
            item["optimizer"].step()
            item["scheduler"].step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            item["seconds"] += time.perf_counter()-started
        if step in checkpoints or step == args.probe_steps:
            values = {
                name: float(inverse_symbol_score(
                    case, item["model"], device,
                ).detach())
                for name, item in candidates.items()
            }
            curve.append({
                "step": step, "selected_profile": min(values, key=values.get),
                "physics_validation": values,
            })
    rows = []
    for name, item in candidates.items():
        physics = float(inverse_symbol_score(
            case, item["model"], device,
        ).detach())
        pointwise = float(equation_residual(
            case, item["model"], validation_points,
        )[2].detach())
        rows.append({
            "profile": name, "physics_validation": physics,
            "pointwise_physics_validation": pointwise,
            "probe_seconds": item["seconds"],
            "trainable_parameters": parameter_counts(item["model"])[0],
            "sine_modes_per_spectral_axis": 64,
        })
    return min(rows, key=lambda row: row["physics_validation"])["profile"], rows, curve


def output_path(args, case_name, method):
    return Path(args.output)/case_name/method/f"seed_{args.seed}"


def run_train(args):
    device = torch.device(args.device)
    case = configured_case(args)
    model, state, history, summary = train(case, args.method, args, device)
    out = output_path(args, case.name, args.method)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"summary.json").exists() and not args.overwrite:
        raise FileExistsError(out)
    (out/"history.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in history)
    )
    torch.save({"model": state, "configuration": vars(args)}, out/"best_physics.pt")
    summary["output"] = str(out)
    (out/"configuration.json").write_text(json.dumps({
        **vars(args), "operator": "laplacian(u)+q(x,y)u=f",
        "variable_coefficient": bool(getattr(case, "coefficient", None)),
        "terms": getattr(case, "terms", None), "reference_role": "evaluation_only",
        "dataloss": False,
    }, indent=2, default=str))
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_probe(args):
    device = torch.device(args.device)
    case = configured_case(args)
    selected, rows, curve = probe(case, args, device)
    out = output_path(args, case.name, "auto_branch64")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"summary.json").exists() and not args.overwrite:
        raise FileExistsError(out)
    payload = {
        "case": case.name, "method": "auto_branch64", "seed": args.seed,
        "selected_profile": selected, "probe": rows,
        "probe_budget_curve": curve, "probe_steps": args.probe_steps,
        "candidate_profiles": BRANCH_PROFILES,
        "sine_modes_per_spectral_axis": 64,
        "selection_uses_reference": False,
        "reference_role": "evaluation_only", "dataloss": False,
        "output": str(out),
    }
    (out/"configuration.json").write_text(json.dumps({
        **vars(args), **payload,
    }, indent=2, default=str))
    (out/"summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def run_auto(args):
    """Probe branch types, reinitialize the winner, then train full budget."""
    device = torch.device(args.device)
    case = configured_case(args)
    selected, rows, curve = probe(case, args, device)
    probe_seconds = sum(row["probe_seconds"] for row in rows)
    model, state, history, formal = train(case, selected, args, device)
    out = output_path(args, case.name, "auto_branch64")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"summary.json").exists() and not args.overwrite:
        raise FileExistsError(out)
    (out/"history.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in history)
    )
    torch.save({"model": state, "configuration": vars(args)}, out/"best_physics.pt")
    summary = {
        **formal, "method": "auto_branch64",
        "selected_profile": selected, "probe": rows,
        "probe_budget_curve": curve, "probe_steps": args.probe_steps,
        "probe_compute_seconds": probe_seconds,
        "selected_model_compute_seconds": formal["seconds"],
        "seconds": probe_seconds+formal["seconds"],
        "candidate_profiles": BRANCH_PROFILES,
        "sine_modes_per_spectral_axis": 64,
        "selection_uses_reference": False,
        "reference_role": "evaluation_only", "dataloss": False,
        "output": str(out),
    }
    (out/"configuration.json").write_text(json.dumps({
        **vars(args), **summary,
    }, indent=2, default=str))
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--case", choices=("H2", "HX1", "HX2", "HX3"), required=True)
    value.add_argument("--mode", choices=("train", "probe", "auto"), required=True)
    value.add_argument("--method", choices=(
        "dimspec_sine64", *BASELINE_METHODS, *BRANCH_PROFILES,
    ), default="dimspec_sine64")
    value.add_argument("--steps", type=int, default=3000)
    value.add_argument("--rank", type=int, default=None,
                       help="override the product component count")
    value.add_argument("--probe-steps", type=int, default=2000)
    value.add_argument("--probe-checkpoints", type=int, nargs="+",
                       default=(300, 600, 900, 1200, 1500, 1800, 2000))
    value.add_argument("--collocation", type=int, default=2048)
    value.add_argument("--probe-collocation", type=int, default=512)
    value.add_argument("--physics-validation", type=int, default=2048)
    value.add_argument("--evaluation", type=int, default=16384)
    value.add_argument("--eval-every", type=int, default=100)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--smoke", action="store_true")
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.smoke:
        arguments.steps = 2
        arguments.probe_steps = 2
        arguments.probe_checkpoints = (1, 2)
        arguments.collocation = arguments.probe_collocation = 32
        arguments.physics_validation = arguments.evaluation = 64
        arguments.eval_every = 1
    if arguments.mode == "train":
        run_train(arguments)
    elif arguments.mode == "probe":
        run_probe(arguments)
    else:
        run_auto(arguments)
