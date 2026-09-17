#!/usr/bin/env python3
"""Train one method on a non-manufactured H1/A1/A2 replacement.

All methods use the same strong-form operator, hard boundary lift, sampled
collocation points, physics-only checkpoint rule, and numerical reference only
for post-hoc metrics.  Auto probes a locked three-profile bank, reinitializes
the selected profile, and charges the complete probe cost to its summary.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/dimensionwise_12case"))

from antialignment_benchmarks import get_case  # noqa: E402
from antialignment_benchmarks.cases import a1, h1  # noqa: E402
from baseline_models import build_baseline, parameter_counts  # noqa: E402
from model import DimensionWiseCP  # noqa: E402


METHODS = (
    "vanilla_mlp_hardbc", "fourier_mlp_hardbc", "spinn_style_hardbc",
    "cp_pinn_hardbc", "piratenet_hardbc", "rfm_hardbc", "DimSpec-LR-Fixed",
    "DimSpec-LR-Auto",
)
BASELINE_NAMES = set(METHODS[:6])


class PhysicalField(nn.Module):
    """Apply the common hard boundary lift to an unconstrained backbone."""

    def __init__(self, raw: nn.Module, case):
        super().__init__()
        self.raw = raw
        self.case = case

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return (self.case.boundary_lift(points)
                + self.case.boundary_envelope(points) * self.raw(points))


def sample(case, count: int, device: torch.device, seed: int | None = None,
           requires_grad: bool = True) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    unit = torch.rand(count, 2, device=device, generator=generator)
    lower = torch.tensor([item[0] for item in case.bounds], device=device)
    upper = torch.tensor([item[1] for item in case.bounds], device=device)
    return (lower + unit * (upper - lower)).requires_grad_(requires_grad)


def _gradient(value: torch.Tensor, points: torch.Tensor,
              create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(value.sum(), points, create_graph=create_graph)[0]


def operator(case, model: nn.Module, points: torch.Tensor):
    points = points.requires_grad_(True)
    value = model(points)
    if isinstance(model, LiftOnly) and case.boundary_kind == "homogeneous":
        return value, torch.zeros_like(value)
    gradient = _gradient(value, points)
    if case.family == "helmholtz":
        second = []
        for axis in range(2):
            second.append(_gradient(gradient[:, axis], points)[:, axis])
        prediction = second[0] + second[1] + case.coefficient(points) * value
        return value, prediction
    diffusion = case.diffusion(points)
    flux = torch.einsum("nij,nj->ni", diffusion, gradient)
    divergence = torch.zeros_like(value)
    for axis in range(2):
        divergence = divergence + _gradient(flux[:, axis], points)[:, axis]
    prediction = -divergence
    if case.convection is not None:
        prediction = prediction + (case.convection(points) * gradient).sum(dim=1)
    if case.reaction is not None:
        prediction = prediction + case.reaction(points) * value
    return value, prediction


class LiftOnly(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case

    def forward(self, points):
        return self.case.boundary_lift(points)


def locked_scale(case, points: torch.Tensor) -> torch.Tensor:
    forcing = case.forcing(points)
    if float(forcing.square().mean()) > 1e-12:
        return forcing.square().mean().detach().clamp_min(1e-10)
    _, lift_operator = operator(case, LiftOnly(case), points)
    return lift_operator.square().mean().detach().clamp_min(1e-10)


def physics_loss(case, model, points, scale):
    _, prediction = operator(case, model, points)
    residual = prediction - case.forcing(points)
    value = residual.square().mean() / scale
    return value, residual.square().mean().sqrt()


def helmholtz_weak_loss(case, model, grid_size: int = 48,
                        test_modes: int = 32):
    """Inverse-Laplacian-scaled weak residual for the curved-wave case.

    The test space and its normalization are fixed before training.  This is
    a physics loss: neither its construction nor checkpoint score uses the
    numerical reference field.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    (x0, x1), (y0, y1) = case.bounds
    ux = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / grid_size
    uy = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / grid_size
    x = x0 + (x1 - x0) * ux
    y = y0 + (y1 - y0) * uy
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack((xx, yy), dim=-1).reshape(-1, 2).requires_grad_(True)
    value = model(points)
    gradient = _gradient(value, points)
    modes = torch.arange(1, test_modes + 1, device=device, dtype=dtype)
    x_phase = torch.pi * ux[:, None] * modes[None, :]
    y_phase = torch.pi * uy[:, None] * modes[None, :]
    basis_x, basis_y = torch.sin(x_phase), torch.sin(y_phase)
    derivative_x = (torch.pi * modes / (x1 - x0))[None, :] * torch.cos(x_phase)
    derivative_y = (torch.pi * modes / (y1 - y0))[None, :] * torch.cos(y_phase)
    field = value.reshape(grid_size, grid_size)
    grad_x = gradient[:, 0].reshape(grid_size, grid_size)
    grad_y = gradient[:, 1].reshape(grid_size, grid_size)
    reaction_source = (
        -case.coefficient(points) * value + case.forcing(points)
    ).reshape(grid_size, grid_size)
    # Weak form of -(Delta u + q u - s)=0.  A common quadrature factor
    # cancels between residual and source normalization.
    residual_modes = (
        basis_y.T @ grad_x @ derivative_x
        + derivative_y.T @ grad_y @ basis_x
        + basis_y.T @ reaction_source @ basis_x
    ) / float(grid_size * grid_size)
    source_modes = (basis_y.T @ case.forcing(points).reshape(
        grid_size, grid_size
    ) @ basis_x) / float(grid_size * grid_size)
    eig_x = (torch.pi * modes / (x1 - x0)).square()[None, :]
    eig_y = (torch.pi * modes / (y1 - y0)).square()[:, None]
    inverse_scale = 1.0 / (1.0 + eig_x + eig_y)
    residual_scaled = residual_modes * inverse_scale
    source_scaled = source_modes * inverse_scale
    loss = (residual_scaled.square().sum()
            / source_scaled.square().sum().detach().clamp_min(1e-12))
    return loss, residual_scaled.square().mean().sqrt()


def helmholtz_combined_loss(case, model, points, scale):
    weak, _ = helmholtz_weak_loss(case, model)
    strong, strong_rms = physics_loss(case, model, points, scale)
    return weak + 0.01 * strong, strong_rms


@torch.no_grad()
def relative_error(case, model, points):
    prediction = model(points)
    reference = case.reference(points)
    return float(torch.linalg.vector_norm(prediction - reference)
                 / torch.linalg.vector_norm(reference).clamp_min(1e-12))


def proxy_case(case, feature_kinds=None, max_modes=None, rank=None):
    # Baseline CaseModule applies an internal box envelope to names beginning
    # with A or family=helmholtz.  The proxy disables it so PhysicalField is
    # the single, identical boundary implementation for every method.
    return replace(
        case, name="X" + case.name[1:], family="anti_alignment",
        feature_kinds=feature_kinds or case.feature_kinds,
        max_modes=max_modes or case.max_modes,
        rank=rank or case.rank,
    )


def profiles(case):
    return {
        "compact_spectral": {
            "rank": 8, "feature_kinds": ("sine_pi", "sine_pi"),
            "max_modes": (24, 24),
        },
        "balanced_mixed": {
            "rank": 16, "feature_kinds": ("hybrid", "hybrid"),
            "max_modes": (32, 32),
        },
        "rich_mixed": {
            "rank": case.rank, "feature_kinds": case.feature_kinds,
            "max_modes": case.max_modes,
        },
    }


def build_model(case, method: str, seed: int, profile_name: str | None = None,
                cp_rank: int | None = None):
    if method in BASELINE_NAMES:
        raw = build_baseline(method, proxy_case(case))
        return PhysicalField(raw, case), {"profile": "baseline"}
    if profile_name is None:
        profile_name = "rich_mixed"
    config = dict(profiles(case)[profile_name])
    if cp_rank is not None:
        config["rank"] = int(cp_rank)
    configured = proxy_case(case, **config)
    raw = DimensionWiseCP(
        configured, "trainable_coefficients", latent_fraction=1.0, seed=seed,
    )
    # Avoid an O(1) Helmholtz residual at initialization while retaining
    # nonzero products and gradients for every CP branch.
    with torch.no_grad():
        for branch in raw.branches:
            branch.base.mul_(0.05)
            branch.coefficient.copy_(branch.base)
    return PhysicalField(raw, case), {"profile": profile_name, **config}


def _diffusion_divergence(case, points):
    if case.diffusion is None:
        return None
    query = points.detach().clone().requires_grad_(True)
    matrix = case.diffusion(query)
    if not matrix.requires_grad:
        return torch.zeros(query.shape[0], 2, device=query.device,
                           dtype=query.dtype)
    divergence = torch.zeros(query.shape[0], 2, device=query.device,
                             dtype=query.dtype)
    for j in range(2):
        for i in range(2):
            derivative = torch.autograd.grad(
                matrix[:, i, j].sum(), query, create_graph=False,
                retain_graph=True,
            )[0][:, i]
            divergence[:, j] += derivative
    return divergence.detach()


def fit_rfm(case, model: PhysicalField, points: torch.Tensor,
            ridge: float = 1e-7):
    """Physics-only linear collocation for the common hard-lifted RFM."""
    from torch.func import jacfwd, vmap

    raw = model.raw

    def feature_vector(point):
        query = point[None, :]
        return case.boundary_envelope(query)[:, None][0] * raw.linear_features(query)[0]

    values = vmap(feature_vector)(points)
    jacobian = vmap(jacfwd(feature_vector))(points)
    hessian = vmap(jacfwd(jacfwd(feature_vector)))(points)
    if case.family == "helmholtz":
        design = (hessian.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                  + case.coefficient(points)[:, None] * values)
    else:
        diffusion = case.diffusion(points)
        div_diffusion = _diffusion_divergence(case, points)
        design = -(hessian * diffusion[:, None, :, :]).sum(dim=(-2, -1))
        design = design - (jacobian * div_diffusion[:, None, :]).sum(dim=-1)
        if case.convection is not None:
            design = design + (
                jacobian * case.convection(points)[:, None, :]
            ).sum(dim=-1)
        if case.reaction is not None:
            design = design + case.reaction(points)[:, None] * values
    lift_points = points.detach().clone().requires_grad_(True)
    _, lift_operator = operator(case, LiftOnly(case), lift_points)
    target = case.forcing(points) - lift_operator.detach()
    scale = design.square().mean(dim=0).sqrt().clamp_min(1e-9)
    equilibrated = (design / scale).double()
    target64 = target.double()
    gram = equilibrated.T @ equilibrated
    rhs = equilibrated.T @ target64
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    normalized = torch.linalg.solve(
        gram + ridge * gram.diagonal().mean().clamp_min(1e-12) * identity, rhs,
    )
    solution = (normalized / scale.double()).to(points.dtype)
    raw.set_linear_solution(solution)
    residual = design @ solution - target
    return {
        "relative_collocation_residual": float(
            torch.linalg.vector_norm(residual)
            / torch.linalg.vector_norm(target).clamp_min(1e-12)
        ),
        "collocation_points": int(points.shape[0]), "ridge": ridge,
    }


def train_once(case, method, seed, device, steps, lr, batch, validation_points,
               validation_scale, evaluation_points, eval_every, profile_name=None,
               cp_rank=None, save_dir: Path | None = None, save_checkpoint=True,
               evaluate_reference: bool = True,
               evaluate_reference_trajectory: bool = False,
               initial_checkpoint: Path | None = None,
               lr_schedule: str = "cosine", stage_fraction: float = 0.4,
               stage_lr_ratio: float = 0.1,
               helmholtz_loss_kind: str = "combined",
               effective_lr_override: float | None = None):
    torch.manual_seed(seed)
    model, profile = build_model(case, method, seed, profile_name, cp_rank)
    model = model.to(device)
    if initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location=device,
                             weights_only=False)
        model.load_state_dict(payload["model"] if "model" in payload else payload)
    trainable, fixed = parameter_counts(model)
    effective_lr = 1.0e-3 if method == "piratenet_hardbc" else lr
    if effective_lr_override is not None:
        effective_lr = effective_lr_override
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps), eta_min=effective_lr * 0.05,
        )
        first_stage_steps = steps
    elif lr_schedule == "two_stage":
        first_stage_steps = max(1, min(steps - 1, round(steps * stage_fraction)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=first_stage_steps,
            eta_min=effective_lr * stage_lr_ratio,
        )
    else:
        raise ValueError(f"unknown learning-rate schedule {lr_schedule!r}")
    best_score, best_row, best_state = math.inf, None, None
    history = []
    started = time.perf_counter()
    prefit = None
    actual_steps = steps
    if method == "rfm_hardbc":
        fitting = sample(case, max(4096, batch * 2), device, seed + 700001)
        prefit = fit_rfm(case, model, fitting)
        actual_steps = 0
    for step in range(actual_steps + 1):
        if step:
            if case.family == "helmholtz":
                points = sample(case, batch, device)
                scale = locked_scale(case, points)
                loss, _ = (physics_loss(case, model, points, scale)
                           if helmholtz_loss_kind == "strong" else
                           helmholtz_combined_loss(case, model, points, scale))
            else:
                points = sample(case, batch, device)
                scale = locked_scale(case, points)
                loss, _ = physics_loss(case, model, points, scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            scheduler.step()
            if lr_schedule == "two_stage" and step == first_stage_steps:
                second_lr = effective_lr * stage_lr_ratio
                for group in optimizer.param_groups:
                    group["lr"] = second_lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, steps - first_stage_steps),
                    eta_min=effective_lr * 0.005,
                )
        if step not in (0, 1, actual_steps) and step % eval_every:
            continue
        if case.family == "helmholtz":
            validation_loss, residual_rms = (
                physics_loss(case, model, validation_points, validation_scale)
                if helmholtz_loss_kind == "strong" else
                helmholtz_combined_loss(
                    case, model, validation_points, validation_scale,
                )
            )
        else:
            validation_loss, residual_rms = physics_loss(
                case, model, validation_points, validation_scale,
            )
        row = {
            "step": step, "physics_validation": float(validation_loss.detach()),
            "residual_rms": float(residual_rms.detach()),
            "relative_l2_evaluation_only": None,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        if evaluate_reference_trajectory:
            # Evaluation-only diagnostic for multi-budget curves.  It never
            # participates in optimization, scheduling, or checkpoint choice.
            row["relative_l2_evaluation_only"] = relative_error(
                case, model, evaluation_points
            )
        history.append(row)
        if row["physics_validation"] < best_score:
            best_score, best_row = row["physics_validation"], row
            if save_checkpoint:
                best_state = copy.deepcopy(model.state_dict())
    if best_row is None:
        raise RuntimeError("no checkpoint was evaluated")
    if evaluate_reference:
        if best_state is not None:
            model.load_state_dict(best_state)
        posthoc_error = relative_error(case, model, evaluation_points)
        best_row["relative_l2_evaluation_only"] = posthoc_error
        for row in history:
            if row["step"] == best_row["step"]:
                row["relative_l2_evaluation_only"] = posthoc_error
                break
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "history.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in history)
        )
        if best_state is not None:
            torch.save({"model": best_state, "row": best_row,
                        "profile": profile}, save_dir / "best_physics.pt")
    return {
        "best_row": best_row, "history": history, "model": model,
        "profile": profile, "trainable_parameters": trainable,
        "fixed_parameters": fixed, "seconds": time.perf_counter() - started,
        "actual_optimizer_steps": actual_steps, "rfm_prefit": prefit,
    }


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("H1", "A1", "A2"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--validation", type=int, default=4096)
    parser.add_argument("--evaluation", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--effective-lr", type=float, default=None,
                        help="explicit formal-training LR, including PirateNet")
    parser.add_argument("--probe-lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--trajectory-reference", action="store_true",
        help=("record evaluation-only relative error at every checkpoint; "
              "checkpoint selection remains physics-only"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cp-rank", type=int, default=None)
    parser.add_argument("--a1-epsilon", type=float, default=None)
    parser.add_argument("--h1-k0", type=float, default=None)
    parser.add_argument("--reference-file", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-kinds", nargs=2, default=None)
    parser.add_argument("--max-modes", nargs=2, type=int, default=None)
    parser.add_argument("--lr-schedule", choices=("cosine", "two_stage"),
                        default="cosine")
    parser.add_argument("--stage-fraction", type=float, default=0.4)
    parser.add_argument("--stage-lr-ratio", type=float, default=0.1)
    parser.add_argument("--helmholtz-loss", choices=("combined", "strong"),
                        default="combined")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/antialignment_replacements_v1/runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    case = get_case(args.case)
    if args.case == "H1" and args.h1_k0 is not None:
        case = h1(args.h1_k0, args.reference_file)
    elif args.case == "A1" and args.a1_epsilon is not None:
        case = a1(args.a1_epsilon, args.reference_file)
    elif args.reference_file is not None:
        case = replace(case, reference_file=args.reference_file)
    if args.feature_kinds is not None:
        case = replace(case, feature_kinds=tuple(args.feature_kinds))
    if args.max_modes is not None:
        case = replace(case, max_modes=tuple(args.max_modes))
    device = torch.device(args.device)
    output = args.output / case.name / args.method / f"seed_{args.seed}"
    if args.cp_rank is not None:
        output = (args.output / "rank_sweep" / case.name /
                  f"rank_{args.cp_rank}" / f"seed_{args.seed}")
    if (output / "summary.json").exists() and not args.overwrite:
        print(json.dumps({"event": "reuse", "path": str(output / "summary.json")}))
        return
    output.mkdir(parents=True, exist_ok=True)
    validation_points = sample(case, args.validation, device, args.seed + 800001)
    evaluation_points = sample(
        case, args.evaluation, device, args.seed + 800002, requires_grad=False,
    )
    validation_scale = locked_scale(case, validation_points)
    probe_records = []
    selected_profile = None
    probe_seconds = 0.0
    if args.method == "DimSpec-LR-Auto":
        for index, profile_name in enumerate(profiles(case)):
            result = train_once(
                case, "DimSpec-LR-Fixed", args.seed + 10000 + index, device,
                args.probe_steps, args.probe_lr, args.batch, validation_points,
                validation_scale, evaluation_points, args.eval_every,
                profile_name=profile_name, save_checkpoint=False,
                evaluate_reference=False,
                lr_schedule=args.lr_schedule,
                stage_fraction=args.stage_fraction,
                stage_lr_ratio=args.stage_lr_ratio,
                helmholtz_loss_kind=args.helmholtz_loss,
            )
            record = {
                "profile": profile_name,
                "physics_validation": result["best_row"]["physics_validation"],
                "trainable_parameters": result["trainable_parameters"],
                "probe_seconds": result["seconds"],
            }
            probe_records.append(record)
            probe_seconds += result["seconds"]
        selected_profile = min(
            probe_records, key=lambda item: item["physics_validation"]
        )["profile"]
    elif args.method == "DimSpec-LR-Fixed":
        selected_profile = "rich_mixed"
    formal_method = ("DimSpec-LR-Fixed"
                     if args.method == "DimSpec-LR-Auto" else args.method)
    formal = train_once(
        case, formal_method, args.seed, device, args.steps, args.lr, args.batch,
        validation_points, validation_scale, evaluation_points, args.eval_every,
        profile_name=selected_profile, cp_rank=args.cp_rank, save_dir=output,
        initial_checkpoint=args.resume_checkpoint,
        evaluate_reference_trajectory=args.trajectory_reference,
        lr_schedule=args.lr_schedule, stage_fraction=args.stage_fraction,
        stage_lr_ratio=args.stage_lr_ratio,
        helmholtz_loss_kind=args.helmholtz_loss,
        effective_lr_override=args.effective_lr,
    )
    total_seconds = probe_seconds + formal["seconds"]
    peak = formal["trainable_parameters"]
    if probe_records:
        peak = sum(item["trainable_parameters"] for item in probe_records)
    summary = {
        "case": case.name, "family": case.family, "method": args.method,
        "seed": args.seed, "best_step": formal["best_row"]["step"],
        "relative_l2": formal["best_row"]["relative_l2_evaluation_only"],
        "physics_validation": formal["best_row"]["physics_validation"],
        "seconds": total_seconds, "formal_seconds": formal["seconds"],
        "probe_seconds": probe_seconds, "selected_profile": selected_profile,
        "candidate_probes": probe_records,
        "trainable_parameters": formal["trainable_parameters"],
        "fixed_parameters": formal["fixed_parameters"],
        "peak_trainable_parameters": peak,
        "actual_optimizer_steps": formal["actual_optimizer_steps"],
        "formal_optimizer_steps": args.steps,
        "probe_steps_per_candidate": args.probe_steps,
        "cp_rank": args.cp_rank or formal["profile"].get("rank"),
        "profile": formal["profile"], "rfm_prefit": formal["rfm_prefit"],
        "reference_role": "evaluation_only", "dataloss": False,
        "resume_checkpoint": (str(args.resume_checkpoint)
                              if args.resume_checkpoint else None),
        "lr_schedule": args.lr_schedule,
        "stage_fraction": args.stage_fraction,
        "stage_lr_ratio": args.stage_lr_ratio,
        "helmholtz_loss": args.helmholtz_loss,
        "boundary_protocol": "common_hard_lift",
        "reference": str(case.reference_file), "output": str(output),
    }
    configuration = {**vars(args), "output": str(output),
                     "effective_learning_rate": (args.effective_lr if args.effective_lr is not None
                         else 1e-3 if formal_method == "piratenet_hardbc" else args.lr),
                     "reference": str(case.reference_file),
                     "selected_profile": selected_profile,
                     "candidate_profiles": profiles(case)}
    atomic_json(output / "configuration.json", configuration)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
