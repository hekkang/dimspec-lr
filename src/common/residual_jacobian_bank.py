"""Residual-Jacobian growth for dimension-wise frozen Mapping banks.

The target optimizer remains latent-only.  At a rank-growth event this module
uses a physics objective to obtain ``-J^T r`` in each branch's *full fixed
coefficient space*, orthogonalizes that direction against the active Mapping,
and writes it into an inactive Mapping column.  The new latent coordinate is
initialized to zero, so changing the bank does not change the represented
field until the latent-only optimizer takes another step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


ObjectiveBuilder = Callable[[], torch.Tensor]


def _coefficient_gradients(model: torch.nn.Module,
                           objective: ObjectiveBuilder) -> tuple[float, dict[int, torch.Tensor]]:
    branches = list(getattr(model, "branches"))
    for branch in branches:
        branch.base.requires_grad_(True)
        branch.base.grad = None
    model.zero_grad(set_to_none=True)
    value = objective()
    value.backward()
    gradients = {
        axis: -branch.base.grad.detach().reshape(-1).clone()
        for axis, branch in enumerate(branches)
        if branch.base.grad is not None
    }
    for branch in branches:
        branch.base.requires_grad_(False)
        branch.base.grad = None
        if branch.latent is not None:
            branch.latent.grad = None
    return float(value.detach()), gradients


def _append(branch: torch.nn.Module, direction: torch.Tensor,
            tolerance: float) -> tuple[bool, int, float]:
    if branch.latent is None or branch.mapping.numel() == 0:
        return False, branch.active_latent_count, 0.0
    column = branch.active_latent_count
    if column >= branch.mapping.shape[1]:
        return False, column, 0.0
    vector = direction.to(branch.mapping).reshape(-1)
    if vector.numel() != branch.mapping.shape[0]:
        raise ValueError("residual direction and branch coefficient sizes differ")
    if column:
        active = branch.mapping[:, :column]
        # Two passes make modified Gram--Schmidt robust for nearly dependent
        # residual directions accumulated over a long nonlinear trajectory.
        for _ in range(2):
            vector = vector-active@(active.T@vector)
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm) <= tolerance:
        return False, column, float(norm) if torch.isfinite(norm) else 0.0
    with torch.no_grad():
        branch.mapping[:, column].copy_(vector/norm)
        branch.latent[column].zero_()
        branch.active_latent_size.fill_(column+1)
    return True, column+1, float(norm)


def grow_from_residual_jacobian(
    model: torch.nn.Module,
    objectives: Sequence[tuple[str, ObjectiveBuilder]],
    *,
    directions_per_event: int = 1,
    selected_axis: int | None = None,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Append target-physics directions without making weights trainable.

    If ``selected_axis`` is omitted, the dimension with the largest normalized
    full-coefficient gradient is selected independently for each objective.
    This makes rank growth dimension selective while preserving one named
    latent vector per physical input.
    """
    records, accepted = [], 0
    for label, objective in objectives:
        if accepted >= directions_per_event:
            break
        loss, gradients = _coefficient_gradients(model, objective)
        scores = {
            axis: float(torch.linalg.vector_norm(gradient)
                        / max(1, gradient.numel())**0.5)
            for axis, gradient in gradients.items()
        }
        candidates = ([selected_axis] if selected_axis is not None else
                      sorted(scores, key=scores.get, reverse=True))
        record = {"label": label, "physics_objective": loss,
                  "dimension_scores": scores, "accepted": False}
        for axis in candidates:
            gradient = gradients.get(axis)
            if gradient is None:
                continue
            grew, active, raw_norm = _append(
                model.branches[axis], gradient, tolerance,
            )
            if grew:
                record.update({"accepted": True, "selected_axis": axis,
                               "active_after": active,
                               "raw_direction_norm": raw_norm})
                accepted += 1
                break
        records.append(record)
    return {"accepted_directions": accepted, "directions": records}
