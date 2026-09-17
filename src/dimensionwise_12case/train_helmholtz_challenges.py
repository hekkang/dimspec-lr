#!/usr/bin/env python3
"""Physics-only Helmholtz stress tests without dictionary-aligned targets.

The three cases isolate off-grid oscillations, a variable reaction coefficient,
and high effective separation rank.  They are an external-validity diagnostic
only and are not consumed by the current paper-table builder.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from baseline_models import PirateNetPINN, SPINNModified, parameter_counts
from model import DimensionWiseCP


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/reviewer_followup_20260815/helmholtz_challenges"
PROFILES = {"compact": 16, "aligned": 32, "rich": 48}


@dataclass(frozen=True)
class ChallengeCase:
    name: str
    terms: tuple[tuple[float, float, float, bool], ...]
    variable_coefficient: bool
    rank: int
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    # Reuse the established hard-Dirichlet treatment in neural baselines.
    family: str = "helmholtz"
    feature_kinds: tuple[str, ...] = ("sine_pi", "sine_pi")
    max_modes: tuple[int, ...] = (32, 32)
    time_axis: None = None

    @property
    def dimension(self):
        return 2

    @staticmethod
    def axis(x: torch.Tensor, frequency: float, enveloped: bool):
        wave = frequency * math.pi
        sine, cosine = torch.sin(wave*x), torch.cos(wave*x)
        if not enveloped:
            return sine, -(wave**2)*sine
        envelope = x*(1.0-x)
        second = (-2.0*sine + 2.0*(1.0-2.0*x)*wave*cosine
                  -envelope*(wave**2)*sine)
        return envelope*sine, second

    def exact_and_laplacian(self, points: torch.Tensor):
        x, y = points.unbind(dim=1)
        value = torch.zeros_like(x)
        laplacian = torch.zeros_like(x)
        for coefficient, fx, fy, enveloped in self.terms:
            vx, dxx = self.axis(x, fx, enveloped)
            vy, dyy = self.axis(y, fy, enveloped)
            value = value + coefficient*vx*vy
            laplacian = laplacian + coefficient*(dxx*vy+vx*dyy)
        return value, laplacian

    def exact(self, points: torch.Tensor):
        return self.exact_and_laplacian(points)[0]

    def coefficient(self, points: torch.Tensor):
        if not self.variable_coefficient:
            return torch.ones(points.shape[0], device=points.device,
                              dtype=points.dtype)
        x, y = points.unbind(dim=1)
        return 1.0 + 0.75*torch.sin(2.0*math.pi*x)*torch.cos(2.0*math.pi*y)

    def forcing(self, points: torch.Tensor):
        value, laplacian = self.exact_and_laplacian(points)
        return laplacian+self.coefficient(points)*value


def get_case(name: str):
    if name == "HX1":
        return ChallengeCase(
            name, ((1.0, 13.5, 5.25, True),), False, rank=8,
        )
    if name == "HX2":
        return ChallengeCase(
            name,
            ((1.0, 9.5, 3.75, True), (0.35, 4.25, 12.5, True)),
            True, rank=12,
        )
    if name == "HX3":
        pairs = ((1, 24), (2, 21), (3, 18), (4, 15), (5, 12), (6, 9),
                 (7, 7), (9, 6), (12, 5), (15, 4), (18, 3), (21, 2))
        terms = tuple(((-1.0)**index/math.sqrt(len(pairs)), float(a), float(b), False)
                      for index, (a, b) in enumerate(pairs))
        return ChallengeCase(name, terms, False, rank=16)
    raise ValueError(name)


def sample(count: int, device: torch.device, requires_grad=False):
    return torch.rand(count, 2, device=device).requires_grad_(requires_grad)


def second_derivatives(value, points):
    first = torch.autograd.grad(value.sum(), points, create_graph=True)[0]
    return [torch.autograd.grad(
        first[:, axis].sum(), points, create_graph=True, retain_graph=True,
    )[0][:, axis] for axis in range(2)]


class FullSineGalerkin(nn.Module):
    """Mode-aware full tensor-product control for the stress tests.

    Unlike the CP models, this baseline trains a dense coefficient matrix over
    the complete sine-product space.  It is deliberately included here because
    a Helmholtz stress test should not claim an advantage merely by omitting a
    classical spectral representation that already matches homogeneous
    Dirichlet boundaries.
    """

    def __init__(self, modes: int = 48):
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(modes, modes))
        self.register_buffer("frequencies", torch.arange(1, modes+1).float())

    def basis(self, points: torch.Tensor):
        phase_x = math.pi*points[:, 0:1]*self.frequencies[None, :]
        phase_y = math.pi*points[:, 1:2]*self.frequencies[None, :]
        return torch.sin(phase_x), torch.sin(phase_y)

    def forward(self, points: torch.Tensor):
        x, y = self.basis(points)
        return torch.einsum("ni,ij,nj->n", x, self.coefficients, y)

    def forward_with_diagonal_second(self, points: torch.Tensor):
        x, y = self.basis(points)
        symbol = -(math.pi*self.frequencies).square()
        value = torch.einsum("ni,ij,nj->n", x, self.coefficients, y)
        dxx = torch.einsum(
            "ni,ij,nj->n", x*symbol[None, :], self.coefficients, y,
        )
        dyy = torch.einsum(
            "ni,ij,nj->n", x, self.coefficients, y*symbol[None, :],
        )
        return value, (dxx, dyy)


def residual(case, model, points):
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
    return prediction, equation.square().mean()/scale


@torch.no_grad()
def metric(case, model, device):
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(2026081501)
        points = sample(32768, device)
    prediction = model(points)
    reference = case.exact(points)
    return float(torch.linalg.vector_norm(prediction-reference)
                 /torch.linalg.vector_norm(reference).clamp_min(1.0e-12))


def build(case, method: str, seed: int, device):
    torch.manual_seed(seed)
    if method in PROFILES:
        configured = replace(case, max_modes=(PROFILES[method],)*2)
        model = DimensionWiseCP(
            configured, "trainable_coefficients", 1.0, seed,
        )
        with torch.no_grad():
            for branch in model.branches:
                branch.base.mul_(0.05)
                branch.coefficient.copy_(branch.base)
        return configured, model.to(device), 0.02
    if method == "spinn":
        model = SPINNModified(case, rank=32, width=64, depth=4)
        with torch.no_grad():
            for branch in model.branches:
                branch.output_layer.weight.mul_(0.1)
                branch.output_layer.bias.mul_(0.1)
        return case, model.to(device), 0.001
    if method == "piratenet":
        model = PirateNetPINN(case, width=256, blocks=3)
        with torch.no_grad():
            model.output_layer.direction.mul_(0.01)
            model.output_layer.bias.zero_()
        return case, model.to(device), 0.001
    if method == "galerkin":
        # A dense Galerkin matrix has thousands of initially zero modes.  A
        # neural-network learning rate gives every Adam coordinate an O(1e-2)
        # first step and grossly overshoots the O(1e-2--1e-1) spectral
        # coefficients.  The coefficient-space control therefore uses the
        # scale-appropriate rate below.
        return case, FullSineGalerkin(48).to(device), (
            1.0e-2 if case.name == "HX3" else 1.0e-4
        )
    raise ValueError(method)


def validation(case, model, points):
    return float(residual(case, model, points)[1].detach())


def probe(case, args, device, validation_points):
    candidates = {}
    for profile in PROFILES:
        configured, model, lr = build(case, profile, args.seed+101, device)
        candidates[profile] = {
            "case": configured, "model": model,
            "optimizer": torch.optim.Adam(model.parameters(), lr=lr),
            "seconds": 0.0,
        }
    # Exclude one-time kernel/module initialization from candidate cost.  CUDA
    # synchronization is required because host-side perf_counter otherwise
    # measures only asynchronous launch latency.
    for candidate in candidates.values():
        residual(candidate["case"], candidate["model"], validation_points[:32])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    checkpoints = set(args.probe_checkpoints)
    curve = []
    for step in range(1, args.probe_steps+1):
        torch.manual_seed(args.seed+10000+step)
        points = sample(args.collocation, device)
        for candidate in candidates.values():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss = residual(candidate["case"], candidate["model"], points)[1]
            candidate["optimizer"].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate["model"].parameters(), 100.0)
            candidate["optimizer"].step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            candidate["seconds"] += time.perf_counter()-started
        if step in checkpoints or step == args.probe_steps:
            values = {
                name: validation(item["case"], item["model"], validation_points)
                for name, item in candidates.items()
            }
            checkpoint_counts = sorted(
                sum(p.numel() for p in item["model"].parameters())
                for item in candidates.values()
            )
            scores = {}
            for name, item in candidates.items():
                parameters = sum(p.numel() for p in item["model"].parameters())
                cost = parameters/checkpoint_counts[1]
                scores[name] = (math.log(max(values[name], 1.0e-20))
                                +args.parameter_penalty*cost)
            winner = min(scores, key=scores.get)
            curve.append({"probe_steps": step, "selected_profile": winner,
                          "physics_validation": values,
                          "selection_score": scores})
    rows = []
    counts = sorted(sum(p.numel() for p in item["model"].parameters())
                    for item in candidates.values())
    for name, item in candidates.items():
        physics = validation(item["case"], item["model"], validation_points)
        parameters = sum(p.numel() for p in item["model"].parameters())
        # Wall time is reported but not used by this stress-test selector.
        # Shared GPUs make short asynchronous probe timings noisy; the locked
        # selector uses residual plus an architecture-size tie breaker.
        cost = parameters/counts[1]
        rows.append({
            "profile": name, "modes": item["case"].max_modes,
            "physics_validation": physics, "seconds": item["seconds"],
            "parameters": parameters,
            "selection_score": math.log(max(physics, 1.0e-20))
                               +args.parameter_penalty*cost,
        })
    winner = min(rows, key=lambda row: row["selection_score"])["profile"]
    return winner, rows, curve


def train_formal(case, method, args, device, validation_points):
    configured, model, lr = build(case, method, args.seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=lr*0.05,
    )
    best_physics, best_error, best_step, best_state = math.inf, math.inf, 0, None
    history = []
    started = time.perf_counter()
    for step in range(args.steps+1):
        points = sample(args.collocation, device)
        _, loss = residual(configured, model, points)
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step(); scheduler.step()
        if step in (0, 1, args.steps) or step % args.eval_every == 0:
            physics = validation(configured, model, validation_points)
            error = metric(configured, model, device)
            row = {"step": step, "physics_validation": physics,
                   "relative_l2_evaluation_only": error,
                   "seconds": time.perf_counter()-started}
            history.append(row); print(json.dumps(row), flush=True)
            if physics < best_physics:
                best_physics, best_error, best_step = physics, error, step
                best_state = copy.deepcopy(model.state_dict())
    return configured, model, history, best_state, {
        "best_step": best_step, "best_physics_validation": best_physics,
        "physics_selected_relative_l2": best_error,
        "seconds": time.perf_counter()-started,
    }


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    case = get_case(args.case)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(args.seed+991)
        validation_points = sample(args.physics_validation, device)
    probe_rows, probe_curve = [], []
    method = args.method
    if method == "auto":
        method, probe_rows, probe_curve = probe(case, args, device, validation_points)
    configured, model, history, best_state, result = train_formal(
        case, method, args, device, validation_points,
    )
    output = Path(args.output)/case.name/args.method/f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists() and not args.overwrite:
        raise FileExistsError(output)
    (output/"history.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in history)
    )
    torch.save({"model": best_state, "configuration": vars(args)},
               output/"best_physics.pt")
    trainable, fixed = parameter_counts(model)
    summary = {
        "family": "helmholtz_challenge", "case": case.name,
        "requested_method": args.method, "selected_profile": method,
        "probe": probe_rows, "probe_budget_curve": probe_curve,
        "off_dictionary": case.name in {"HX1", "HX2"},
        "variable_coefficient": case.variable_coefficient,
        "effective_term_count": len(case.terms), "cp_rank": case.rank,
        "selected_modes": configured.max_modes,
        "trainable_parameters": trainable, "fixed_parameters": fixed,
        **result, "reference_role": "evaluation_only", "dataloss": False,
        "output": str(output),
    }
    (output/"configuration.json").write_text(json.dumps({
        **vars(args), "case_definition": {
            "terms": case.terms,
            "variable_coefficient": case.variable_coefficient,
            "operator": "laplacian(u)+q(x,y)u=f",
        }, "reference_role": "evaluation_only", "dataloss": False,
    }, indent=2, default=str))
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--case", choices=("HX1", "HX2", "HX3"), required=True)
    value.add_argument("--method", choices=(
        "auto", "compact", "aligned", "rich", "spinn", "piratenet",
        "galerkin",
    ), required=True)
    value.add_argument("--steps", type=int, default=3000)
    value.add_argument("--probe-steps", type=int, default=300)
    value.add_argument("--probe-checkpoints", type=int, nargs="+",
                       default=(50, 100, 200, 300))
    value.add_argument("--collocation", type=int, default=2048)
    value.add_argument("--physics-validation", type=int, default=4096)
    value.add_argument("--parameter-penalty", type=float, default=0.1)
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
        arguments.steps = 2; arguments.probe_steps = 2
        arguments.probe_checkpoints = (1, 2)
        arguments.collocation = 32; arguments.physics_validation = 64
        arguments.eval_every = 1
    run(arguments)
