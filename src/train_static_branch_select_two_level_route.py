#!/usr/bin/env python3
"""Branch selection and two-level capacity routing for static A/S PDEs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "dimensionwise_12case"))

from common.two_level_capacity_routing import RoutedDimensionWiseCP
from dimensionwise_12case.model import DimensionWiseCP
from steady_anisotropic.cases import get_case as get_anisotropic_case
from steady_anisotropic.train import (
    boundary_loss, build_inverse_symbol_target,
    inverse_symbol_loss, physics_loss as anisotropic_physics_loss,
    operator as anisotropic_operator, reference_metric,
    sample as sample_anisotropic,
)
from stationary_transport.cases import get_case as get_transport_case
from stationary_transport.train import (
    inflow_loss, quadrature, reference_metrics, sample_x,
    transport_residual,
)


OUTPUT = ROOT / "results" / "multiphysics_capacity_routing"
PROFILES = ("compact", "aligned", "rich")


def profile_case(case, profile):
    factor = {"compact": 0.5, "aligned": 1.0, "rich": 1.5}[profile]
    modes = tuple(max(2, int(math.ceil(value * factor)))
                  for value in case.max_modes)
    return replace(case, max_modes=modes)


def candidate_cases(case, args):
    """Return physics-routed capacity/basis candidates for the probe stage."""
    if args.family != "transport" or not args.transport_branch_search:
        return {profile: profile_case(case, profile) for profile in PROFILES}
    candidates = {}
    for profile in ("compact", "aligned"):
        sized = profile_case(case, profile)
        for angular in ("collision_eigen", "half_range_chebyshev"):
            label = f"{profile}_{angular}"
            candidates[label] = replace(
                sized,
                feature_kinds=(sized.feature_kinds[0], angular),
            )
    return candidates


def build(case, seed, routed=False, gate_initial=3.0, base_scale=0.01,
          capacity_routing="two_level"):
    base = DimensionWiseCP(case, "trainable_coefficients", 1.0, seed)
    with torch.no_grad():
        for branch in base.branches:
            branch.base.mul_(base_scale)
            branch.coefficient.copy_(branch.base)
    route_atoms = capacity_routing in {"atom", "two_level"}
    route_ranks = capacity_routing in {"rank", "two_level"}
    return (RoutedDimensionWiseCP(base, gate_initial, route_atoms, route_ranks)
            if routed else base)


def physics_projected_cp_initialization(case, model, points_per_axis: int,
                                        ridge: float) -> dict:
    """Initialize A3 factors from a PDE-only tensor-product Galerkin solve.

    The linear PDE is applied to every product of fixed one-dimensional
    dictionary atoms.  A regularized least-squares solve against the known
    forcing produces a coefficient matrix, whose truncated SVD initializes
    the CP factors.  No exact/reference solution enters this calculation.
    """
    if case.name != "A3" or len(model.branches) != 2:
        raise ValueError("physics-projected initialization is currently for A3")
    device = next(model.parameters()).device
    q = int(points_per_axis)
    coordinate = ((torch.arange(q, device=device, dtype=torch.float32) + 0.5)
                  / q).requires_grad_(True)

    def frame(branch):
        values = branch.dictionary(coordinate)
        first = []
        for atom in range(values.shape[1]):
            first.append(torch.autograd.grad(
                values[:, atom].sum(), coordinate, retain_graph=True
            )[0])
        return (values.detach(), torch.stack(first, dim=1).detach(),
                branch.dictionary.second_derivative(coordinate).detach())

    x0, x1, x2 = frame(model.branches[0])
    y0, y1, y2 = frame(model.branches[1])
    probe = torch.zeros((1, 2), device=device)
    diffusion = case.diffusion(probe)[0]
    convection = case.convection(probe)[0]
    reaction = case.reaction(probe)[0]

    def outer(left, right):
        return torch.einsum("ia,jb->ijab", left, right)

    design = (
        -diffusion[0, 0] * outer(x2, y0)
        -2.0 * diffusion[0, 1] * outer(x1, y1)
        -diffusion[1, 1] * outer(x0, y2)
        +convection[0] * outer(x1, y0)
        +convection[1] * outer(x0, y1)
        +reaction * outer(x0, y0)
    ).reshape(q * q, -1)
    xx, yy = torch.meshgrid(coordinate.detach(), coordinate.detach(),
                            indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    forcing = case.forcing(grid).detach()
    gram = design.T @ design
    regularizer = float(ridge) * gram.diagonal().mean().clamp_min(1e-12)
    solution = torch.linalg.solve(
        gram + regularizer * torch.eye(gram.shape[0], device=device),
        design.T @ forcing,
    )
    coefficient = solution.reshape(x0.shape[1], y0.shape[1])
    u, singular, vh = torch.linalg.svd(coefficient, full_matrices=False)
    retained = min(model.rank, singular.numel())
    factor_scale = (singular[:retained] * math.sqrt(model.rank)).sqrt()
    with torch.no_grad():
        for branch in model.branches:
            branch.coefficient.zero_()
        model.branches[0].coefficient[:, :retained].copy_(
            u[:, :retained] * factor_scale[None]
        )
        model.branches[1].coefficient[:, :retained].copy_(
            vh[:retained].T * factor_scale[None]
        )
    residual = design @ coefficient.reshape(-1) - forcing
    return {
        "kind": "PDE_operator_projection_plus_truncated_SVD",
        "points_per_axis": q, "ridge": float(ridge),
        "retained_singular_values": retained,
        "projected_relative_residual": float(
            residual.norm() / forcing.norm().clamp_min(1e-12)
        ),
        "leading_singular_values": singular[:8].detach().cpu().tolist(),
        "reference_used": False,
    }


def transport_projected_cp_initialization(case, model, x_count: int,
                                          ridge: float, mu, weight,
                                          kernel) -> dict:
    """PDE-only Galerkin/SVD initialization for stationary transport.

    The hard inflow lift is kept fixed.  We assemble the linear transport
    operator applied to every tensor product of the spatial and angular
    dictionaries, solve for the raw correction coefficients, and truncate
    that coefficient matrix to the preallocated CP rank.  No reference
    intensity or scalar flux is read.
    """
    if len(model.branches) != 2:
        raise ValueError("transport projection requires x and mu branches")
    device = next(model.parameters()).device
    x = sample_x(
        case, int(x_count), device, 0.35, 0.015, seed=2026080101
    ).sort().values.requires_grad_(True)
    spatial = model.branches[0].dictionary(x)
    spatial_first = []
    for atom in range(spatial.shape[1]):
        spatial_first.append(torch.autograd.grad(
            spatial[:, atom].sum(), x, retain_graph=True
        )[0])
    spatial_first = torch.stack(spatial_first, dim=1)
    angular = model.branches[1].dictionary(mu)

    positive = mu > 0.0
    distance = torch.where(positive[None], x[:, None], 1.0-x[:, None])
    distance_first = torch.where(
        positive[None], torch.ones_like(distance), -torch.ones_like(distance)
    )
    correction = (
        distance[:, :, None, None]
        * spatial[:, None, :, None]
        * angular[None, :, None, :]
    )
    correction_first = (
        (distance_first[:, :, None] * spatial[:, None, :]
         +distance[:, :, None] * spatial_first[:, None, :])[:, :, :, None]
        *angular[None, :, None, :]
    )
    weighted_kernel = kernel*weight[None, :]
    correction_scatter = torch.einsum(
        "xkab,jk->xjab", correction, weighted_kernel
    )
    scattering, absorption = case.coefficients_torch(x)
    total = scattering+absorption
    design = (
        mu[None, :, None, None]*correction_first
        +total[:, None, None, None]*correction
        -0.5*scattering[:, None, None, None]*correction_scatter
    )

    xx = x[:, None].expand(-1, mu.numel())
    mm = mu[None, :].expand(x.numel(), -1)
    safe_mu = mm.abs().clamp_min(1e-6)
    if case.lifting_kind == "linear":
        lifting = torch.where(
            mm > 0.0, torch.full_like(xx, case.left_inflow),
            torch.full_like(xx, case.right_inflow),
        )
        lifting_first = torch.zeros_like(lifting)
    elif case.lifting_kind == "uncollided":
        tau_left = case.optical_depth_torch(x, True)[:, None]
        tau_right = case.optical_depth_torch(x, False)[:, None]
        lifting = torch.where(
            mm > 0.0,
            case.left_inflow*torch.exp(-tau_left/safe_mu),
            case.right_inflow*torch.exp(-tau_right/safe_mu),
        )
        lifting_first = torch.where(
            mm > 0.0, -total[:, None]*lifting/safe_mu,
            total[:, None]*lifting/safe_mu,
        )
    else:
        raise ValueError(case.lifting_kind)
    lifting_scatter = (lifting*weight[None, :])@kernel.T
    lifting_residual = (
        mm*lifting_first+total[:, None]*lifting
        -0.5*scattering[:, None]*lifting_scatter
    )

    matrix = design.reshape(x.numel()*mu.numel(), -1).detach()
    target = -lifting_residual.reshape(-1).detach()
    gram = matrix.T@matrix
    regularizer = float(ridge)*gram.diagonal().mean().clamp_min(1e-12)
    coefficient = torch.linalg.solve(
        gram+regularizer*torch.eye(gram.shape[0], device=device),
        matrix.T@target,
    ).reshape(spatial.shape[1], angular.shape[1])
    left, singular, vh = torch.linalg.svd(coefficient, full_matrices=False)
    retained = min(model.rank, singular.numel())
    factor_scale = (singular[:retained]*math.sqrt(model.rank)).sqrt()
    with torch.no_grad():
        for branch in model.branches:
            branch.coefficient.zero_()
        model.branches[0].coefficient[:, :retained].copy_(
            left[:, :retained]*factor_scale[None]
        )
        model.branches[1].coefficient[:, :retained].copy_(
            vh[:retained].T*factor_scale[None]
        )
    projected = matrix@coefficient.reshape(-1)-target
    return {
        "kind": "transport_operator_projection_plus_truncated_SVD",
        "x_points": int(x.numel()), "angular_quadrature": int(mu.numel()),
        "ridge": float(ridge), "retained_singular_values": retained,
        "projected_relative_residual": float(
            projected.norm()/target.norm().clamp_min(1e-12)
        ),
        "leading_singular_values": singular[:8].detach().cpu().tolist(),
        "reference_used": False,
    }


class StaticProblem:
    def __init__(self, args, device):
        self.args, self.device = args, device
        if args.family == "anisotropic":
            original = get_anisotropic_case(args.case, args.task_parameter)
            self.case = replace(
                original,
                rank=max(1, int(round(original.rank * args.rank_multiplier))),
            )
            self.validation = sample_anisotropic(
                args.validation, device, 2026072801
            )
            self.evaluation = sample_anisotropic(
                args.evaluation, device, 2026072802, requires_grad=False
            )
            self.mu = self.weight = self.kernel = None
            self.inverse_targets = {}
        else:
            original = get_transport_case(args.case, args.task_parameter)
            self.case = replace(
                original,
                rank=max(1, int(round(original.rank * args.rank_multiplier))),
            )
            self.mu, self.weight, self.kernel = quadrature(
                self.case, args.quadrature_order, device
            )
            self.validation = sample_x(
                self.case, args.validation_x, device,
                args.interface_fraction, args.interface_width,
                seed=2026072811,
            )
            self.evaluation = None

    def anisotropic_hminus1_loss(self, case, model):
        """Inverse-symbol weighted residual on a fixed sine test frame.

        This is a negative-Sobolev physics norm, not a solution-data term.
        It prevents large eigenvalues of a rotated anisotropic operator from
        dominating the objective and restores sensitivity to flux/solution
        shape.  The reference solution is never evaluated here.
        """
        q = int(self.args.spectral_points_per_axis)
        key = ("hminus1", q)
        if key not in self.inverse_targets:
            coordinate = ((torch.arange(q, device=self.device,
                                        dtype=torch.float32) + 0.5) / q)
            xx, yy = torch.meshgrid(coordinate, coordinate, indexing="ij")
            grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
            modes = torch.arange(1, q + 1, device=self.device,
                                 dtype=torch.float32)
            basis = torch.sin(torch.pi * coordinate[:, None] * modes[None])
            diffusion = case.diffusion(grid[:1])[0]
            convection = case.convection(grid[:1])[0]
            reaction = case.reaction(grid[:1])[0]
            kx = torch.pi * modes[:, None]
            ky = torch.pi * modes[None, :]
            # Positive symbol majorant for both signs generated by a sine
            # test frame.  It includes rotated diffusion and convection.
            elliptic = (diffusion[0, 0] * kx.square()
                        + diffusion[1, 1] * ky.square()
                        + 2.0 * diffusion[0, 1].abs() * kx * ky
                        + reaction)
            symbol = torch.sqrt(
                elliptic.square()
                + (convection[0].abs() * kx
                   + convection[1].abs() * ky).square()
            ).clamp_min(1e-6)
            forcing = case.forcing(grid).reshape(q, q)
            forcing_coeff = basis.T @ forcing @ basis
            scale = (forcing_coeff / symbol).square().sum().clamp_min(1e-12)
            self.inverse_targets[key] = {
                "grid": grid, "basis": basis, "symbol": symbol,
                "scale": scale.detach(),
            }
        target = self.inverse_targets[key]
        grid = target["grid"].detach().clone().requires_grad_(True)
        _, prediction = anisotropic_operator(case, model, grid)
        residual = (prediction - case.forcing(grid)).reshape(q, q)
        coefficients = target["basis"].T @ residual @ target["basis"]
        return (coefficients / target["symbol"]).square().sum() / target["scale"]

    def sample(self, case, count=None):
        if self.args.family == "anisotropic":
            return sample_anisotropic(count or self.args.batch,
                                      self.device)
        return sample_x(
            case, count or self.args.batch_x, self.device,
            self.args.interface_fraction, self.args.interface_width,
        )

    def loss(self, case, model, points, deterministic=False):
        if self.args.family == "anisotropic":
            if case.feature_kinds == ("sine_pi", "sine_pi"):
                key = tuple(case.max_modes)
                if key not in self.inverse_targets:
                    self.inverse_targets[key] = build_inverse_symbol_target(
                        case, self.args.spectral_points_per_axis, self.device
                    )
                target = self.inverse_targets[key]
                if isinstance(model, RoutedDimensionWiseCP):
                    left, right = model.effective_coefficients_by_axis()
                    coefficients = left @ right.T / math.sqrt(model.rank)
                    dx, dy = target["derivative_x"], target["derivative_y"]
                    diffusion = target["diffusion"]
                    operator_coefficients = target["diagonal"] * coefficients
                    operator_coefficients = operator_coefficients - 2.0 * diffusion[0, 1] * (
                        dx @ coefficients @ dy.T
                    )
                    operator_coefficients = operator_coefficients + target["convection"][0] * (
                        dx @ coefficients
                    ) + target["convection"][1] * (coefficients @ dy.T)
                    spectral_residual = operator_coefficients - target["forcing"]
                    preconditioned = spectral_residual / target["diagonal"]
                    scale = (target["forcing"] / target["diagonal"]).square().sum().clamp_min(1e-12)
                    pde = preconditioned.square().sum() / scale
                else:
                    pde, _ = inverse_symbol_loss(model, target)
            elif (case.name == "A3"
                  and self.args.anisotropic_loss != "pointwise"):
                hminus = self.anisotropic_hminus1_loss(case, model)
                if self.args.anisotropic_loss == "hminus1":
                    pde = hminus
                else:
                    pointwise, _ = anisotropic_physics_loss(case, model, points)
                    pde = (hminus
                           + self.args.anisotropic_pointwise_weight * pointwise)
            else:
                pde, _ = anisotropic_physics_loss(case, model, points)
            boundary = boundary_loss(
                case, model, self.args.boundary_batch, self.device,
                2026072803 if deterministic else None,
            )
        else:
            pde, _, _ = transport_residual(
                case, model, points, self.mu, self.weight, self.kernel
            )
            boundary = inflow_loss(case, model, self.mu)
        return pde + self.args.boundary_weight * boundary, pde, boundary

    def metric(self, case, model):
        if self.args.family == "anisotropic":
            relative = reference_metric(case, model, self.evaluation)
            return {"relative_l2": relative}
        relative, flux = reference_metrics(
            case, model, self.args.reference, self.device
        )
        return {"relative_l2": relative, "scalar_flux_relative_l2": flux}


def probe(problem, args):
    candidates = {}
    cases = candidate_cases(problem.case, args)
    for profile, case in cases.items():
        model = build(case, args.seed + 101, base_scale=args.base_scale).to(
            problem.device
        )
        candidates[profile] = {
            "case": case, "model": model,
            "optimizer": torch.optim.Adam(
                model.parameters(),
                lr=(args.probe_lr if args.probe_lr is not None else args.lr),
            ),
            "seconds": 0.0,
        }
    for step in range(args.probe_steps + 1):
        torch.manual_seed(args.seed + 10000 + step)
        for profile, candidate in candidates.items():
            points = problem.sample(candidate["case"], args.probe_batch)
            started = time.perf_counter()
            loss = problem.loss(candidate["case"], candidate["model"], points)[0]
            if step:
                candidate["optimizer"].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate["model"].parameters(),
                                               args.grad_clip)
                candidate["optimizer"].step()
            candidate["seconds"] += time.perf_counter() - started
    median_time = sorted(item["seconds"] for item in candidates.values())[1]
    median_parameters = sorted(sum(p.numel() for p in item["model"].parameters())
                               for item in candidates.values())[1]
    rows = []
    for profile, candidate in candidates.items():
        value = float(problem.loss(
            candidate["case"], candidate["model"], problem.validation, True
        )[0].detach())
        parameters = sum(p.numel() for p in candidate["model"].parameters())
        cost = candidate["seconds"] / median_time + parameters / median_parameters
        rows.append({
            "profile": profile, "modes": candidate["case"].max_modes,
            "physics_validation": value,
            "seconds": candidate["seconds"], "parameters": parameters,
            "selection_score": (-math.log(max(value, 1e-20))
                                - args.probe_parameter_penalty * cost),
        })
    winner = max(rows, key=lambda row: row["selection_score"])["profile"]
    # Preserve the winner state so continuation can be tested explicitly.
    # The default remains an independent retraining protocol because a short
    # physics-residual winner need not be the best long-horizon optimizer.
    winner_state = copy.deepcopy(candidates[winner]["model"].state_dict())
    return winner, rows, cases[winner], winner_state


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = StaticProblem(args, device)
    if args.profile_selection == "auto":
        winner, probe_rows, case, probe_state = probe(problem, args)
    elif args.profile_selection.startswith("fixed_"):
        fixed = args.profile_selection.removeprefix("fixed_")
        if fixed == "dimension_specialized":
            case = problem.case
        else:
            case = profile_case(problem.case, fixed)
        winner, probe_rows, probe_state = f"fixed_{fixed}", [], None
    else:
        winner, probe_rows, probe_state = "generic_fourier_all_axes", [], None
        changes = {"feature_kinds": tuple(
            "fourier_periodic" for _ in problem.case.feature_kinds
        )}
        if hasattr(problem.case, "dictionary_parameters"):
            changes["dictionary_parameters"] = tuple(
                None for _ in problem.case.feature_kinds
            )
        case = replace(problem.case, **changes)
    if args.selection_only:
        family_name = ("steady_anisotropic_cdr" if args.family == "anisotropic"
                       else "stationary_transport")
        output = (args.output/family_name/case.name/args.variant/f"seed_{args.seed}")
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "family": family_name, "case": case.name,
            "mode": "physics_only_branch_selection_stability",
            "selected_profile": winner, "selected_modes": case.max_modes,
            "probe": probe_rows, "task_parameter": args.task_parameter,
            "seed": args.seed, "reference_role": "evaluation_only",
            "dataloss": False, "output": str(output),
        }
        (output/"configuration.json").write_text(
            json.dumps({**vars(args), **payload}, indent=2, default=str)
        )
        (output/"summary.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2)); return
    torch.manual_seed(args.seed)
    model = build(case, args.seed, routed=True,
                  gate_initial=args.gate_initial_log_alpha,
                  base_scale=args.base_scale,
                  capacity_routing=args.capacity_routing).to(device)
    if args.continue_probe_state:
        if probe_state is None:
            raise ValueError("--continue-probe-state requires automatic probing")
        model.base.load_state_dict(probe_state)
    initialization = None
    initialization_started = time.perf_counter()
    if args.physics_projected_warm_start:
        if args.family == "transport":
            initialization = transport_projected_cp_initialization(
                case, model, args.warm_start_points_per_axis,
                args.warm_start_ridge, problem.mu, problem.weight,
                problem.kernel,
            )
        else:
            initialization = physics_projected_cp_initialization(
                case, model, args.warm_start_points_per_axis,
                args.warm_start_ridge,
            )
        print(json.dumps({"event": "physics_projected_warm_start",
                          **initialization}), flush=True)
    initialization_seconds = time.perf_counter() - initialization_started
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    family_name = ("steady_anisotropic_cdr" if args.family == "anisotropic"
                   else "stationary_transport")
    output = (args.output / family_name / case.name / args.variant
              / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(output)
    history_path = output / "history.jsonl"
    history_path.write_text("")
    config = {
        **vars(args), "output": str(output), "selected_profile": winner,
        "selected_modes": case.max_modes, "probe": probe_rows,
        "base_case_rank": int(round(case.rank / args.rank_multiplier)),
        "maximum_cp_rank": case.rank,
        "reference_role": "evaluation_only", "dataloss": False,
        "initialization": initialization,
        "initialization_seconds": initialization_seconds,
        "routing_layers": ["per_axis_dictionary_atoms", "CP_ranks"],
        "capacity_routing_ablation": args.capacity_routing,
    }
    (output / "configuration.json").write_text(
        json.dumps(config, indent=2, default=str)
    )
    best_physics, best_row, best_state = math.inf, None, None
    pre_prune_best, pre_prune_state = math.inf, None
    minima = {"relative_l2": math.inf, "scalar_flux_relative_l2": math.inf}
    consolidated = None
    started = time.perf_counter()
    for step in range(args.steps + 1):
        if args.rank_routing_mode == "physics_greedy":
            phase = ("warmup" if step < args.rank_prune_step else "refine")
            strength = 0.0
        elif step < args.route_start:
            phase, strength = "warmup", 0.0
        elif step < args.consolidate_step:
            phase = "route"
            strength = ((step - args.route_start)
                        / max(1, args.consolidate_step - args.route_start))
        else:
            phase, strength = "refine", 1.0
        model.set_phase(phase, strength)
        if (args.rank_routing_mode == "physics_greedy"
                and step == args.rank_prune_step):
            if pre_prune_state is None:
                raise RuntimeError("no pre-prune physics checkpoint")
            model.load_state_dict(pre_prune_state)
            model.set_phase("warmup", 0.0)
            model.eval()
            consolidated = model.greedy_physics_rank_prune(
                lambda: problem.loss(
                    case, model, problem.validation, True
                )[0],
                args.rank_prune_relative_tolerance,
                args.rank_prune_absolute_target,
                args.minimum_ranks,
            )
            model.train()
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=(args.lr if args.capacity_routing == "none" else args.lr * 0.5),
            )
            print(json.dumps({"event": "physics_rank_prune",
                              **consolidated}), flush=True)
        elif (args.rank_routing_mode == "l0_two_level"
              and step == args.consolidate_step):
            consolidated = model.consolidate(
                args.gate_threshold, args.minimum_ranks,
                args.minimum_atoms_per_rank,
            )
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr * 0.5,
            )
        model.begin_routing_step()
        points = problem.sample(case)
        physical, pde, boundary = problem.loss(case, model, points)
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
            locked = float(problem.loss(
                case, model, problem.validation, True
            )[0].detach())
            metric = problem.metric(case, model)
            model.train()
            for name, value in metric.items():
                minima[name] = min(minima[name], value)
            row = {
                "step": step, "phase": phase,
                "loss": float(loss.detach()), "pde": float(pde.detach()),
                "boundary": float(boundary.detach()),
                "capacity_penalty": float(capacity.detach()),
                "physics_validation": locked,
                "routing": model.routing_statistics(),
                "seconds": time.perf_counter() - started,
                **metric,
            }
            with history_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            selection_start = (args.rank_prune_step
                               if args.rank_routing_mode == "physics_greedy"
                               else args.consolidate_step)
            if step < selection_start and locked < pre_prune_best:
                pre_prune_best = locked
                pre_prune_state = copy.deepcopy(model.state_dict())
            if step >= selection_start and locked < best_physics:
                best_physics, best_row = locked, row
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"model": best_state, "row": row,
                            "configuration": config}, output / "best_physics.pt")
    if best_state is None:
        raise RuntimeError("no post-consolidation checkpoint")
    if args.lbfgs_steps:
        # Deterministic full-batch physics refinement.  Starting from the best
        # Adam checkpoint avoids stochastic late-stage drift while retaining
        # reference-free model selection.
        model.load_state_dict(best_state)
        model.set_phase("refine", 1.0)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer_lbfgs = torch.optim.LBFGS(
            trainable, lr=args.lbfgs_lr, max_iter=1,
            history_size=50, line_search_fn="strong_wolfe",
        )
        for iteration in range(1, args.lbfgs_steps + 1):
            def closure():
                optimizer_lbfgs.zero_grad(set_to_none=True)
                objective = problem.loss(
                    case, model, problem.validation, True
                )[0]
                objective.backward()
                return objective
            optimizer_lbfgs.step(closure)
            if iteration % args.lbfgs_eval_every == 0 or iteration == args.lbfgs_steps:
                model.eval()
                locked, pde, boundary = problem.loss(
                    case, model, problem.validation, True
                )
                locked_value = float(locked.detach())
                metric = problem.metric(case, model)
                model.train()
                for name, value in metric.items():
                    minima[name] = min(minima[name], value)
                row = {
                    "step": args.steps + iteration,
                    "phase": "lbfgs_refine", "loss": locked_value,
                    "pde": float(pde.detach()),
                    "boundary": float(boundary.detach()),
                    "capacity_penalty": 0.0,
                    "physics_validation": locked_value,
                    "routing": model.routing_statistics(),
                    "seconds": time.perf_counter() - started, **metric,
                }
                with history_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if locked_value < best_physics:
                    best_physics, best_row = locked_value, row
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save({"model": best_state, "row": row,
                                "configuration": config},
                               output / "best_physics.pt")
    model.load_state_dict(best_state)
    formal_optimization_seconds = time.perf_counter() - started
    summary = {
        "family": family_name, "case": case.name,
        "mode": ("fixed_dimension_specialized_low_rank"
                 if args.profile_selection.startswith("fixed_")
                 and args.capacity_routing == "none" else
                 "branch_select_physics_validated_rank_pruning"
                 if args.rank_routing_mode == "physics_greedy" else
                 "branch_select_two_level_capacity_routing"),
        "selected_profile": winner, "selected_modes": case.max_modes,
        "probe": probe_rows, "routing": consolidated,
        "full_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "main_branch_trainable_parameters": model.base.trainable_count,
        "effective_active_parameters": model.effective_parameter_count(),
        "best_step": best_row["step"],
        "best_physics_validation": best_physics,
        "physics_selected_relative_l2": best_row["relative_l2"],
        "minimum_monitored_relative_l2": minima["relative_l2"],
        "seconds": formal_optimization_seconds,
        "formal_optimization_seconds": formal_optimization_seconds,
        "initialization_seconds": initialization_seconds,
        "end_to_end_seconds": formal_optimization_seconds + initialization_seconds,
        "reference_role": "evaluation_only", "dataloss": False,
        "output": str(output),
    }
    if args.family == "transport":
        summary.update({
            "physics_selected_scalar_flux_relative_l2":
                best_row["scalar_flux_relative_l2"],
            "minimum_monitored_scalar_flux_relative_l2":
                minima["scalar_flux_relative_l2"],
        })
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("anisotropic", "transport"), required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--task-parameter", type=float, default=None,
                   help="operator/geometry parameter used to build the target PDE")
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--rank-multiplier", type=float, default=1.0,
                   help="multiply the case's preallocated maximum CP rank")
    p.add_argument("--rank-routing-mode",
                   choices=("l0_two_level", "physics_greedy"),
                   default="l0_two_level")
    p.add_argument("--rank-prune-step", type=int, default=1200)
    p.add_argument("--rank-prune-relative-tolerance", type=float, default=0.10)
    p.add_argument("--rank-prune-absolute-target", type=float, default=0.0)
    p.add_argument("--probe-steps", type=int, default=200)
    p.add_argument("--probe-batch", type=int, default=256)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--validation", type=int, default=4096)
    p.add_argument("--evaluation", type=int, default=16384)
    p.add_argument("--boundary-batch", type=int, default=256)
    p.add_argument("--spectral-points-per-axis", type=int, default=128)
    p.add_argument("--anisotropic-loss",
                   choices=("pointwise", "hminus1", "hybrid"),
                   default="pointwise")
    p.add_argument("--anisotropic-pointwise-weight", type=float, default=0.05)
    p.add_argument("--physics-projected-warm-start", action="store_true")
    p.add_argument("--warm-start-points-per-axis", type=int, default=40)
    p.add_argument("--warm-start-ridge", type=float, default=1e-7)
    p.add_argument("--batch-x", type=int, default=64)
    p.add_argument("--validation-x", type=int, default=128)
    p.add_argument("--quadrature-order", type=int, default=32)
    p.add_argument("--transport-branch-search", action="store_true",
                   help="probe collision-eigen versus half-range angular branches")
    p.add_argument("--interface-fraction", type=float, default=0.25)
    p.add_argument("--interface-width", type=float, default=0.02)
    p.add_argument("--boundary-weight", type=float, default=10.0)
    p.add_argument("--base-scale", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--probe-lr", type=float, default=None,
                   help="optional learning rate used only by parallel branch probes")
    p.add_argument("--continue-probe-state", action="store_true",
                   help="continue from winner probe weights instead of retraining it")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lbfgs-steps", type=int, default=0)
    p.add_argument("--lbfgs-lr", type=float, default=0.5)
    p.add_argument("--lbfgs-eval-every", type=int, default=10)
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
    p.add_argument("--minimum-ranks", type=int, default=4)
    p.add_argument("--minimum-atoms-per-rank", type=int, default=4)
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
    if args.family == "transport" and args.reference is None:
        args.reference = (ROOT / "results" / "iclr2027" / "references"
                          / "stationary_transport" / args.case.upper()
                          / "reference.npz")
        args.base_scale = 0.05
    if args.smoke:
        args.steps = 6; args.probe_steps = 1
        args.probe_batch = 8; args.batch = 16; args.validation = 16
        args.evaluation = 32; args.boundary_batch = 8
        args.batch_x = 4; args.validation_x = 4; args.quadrature_order = 4
        args.route_start = 2; args.consolidate_step = 4; args.eval_every = 1
        args.rank_prune_step = 4
    run(args)
