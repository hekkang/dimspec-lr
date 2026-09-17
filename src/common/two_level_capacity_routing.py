"""Generic two-level capacity routing for dimension-wise CP models.

Layer one gates fixed dictionary atoms independently in every input branch and
CP component.  Layer two gates complete CP components.  The gates are trained
with physics losses and are consolidated before the final full-weight refine.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from common.step_locked_gate import StepLockedHardConcrete


class RoutedDimensionWiseCP(nn.Module):
    """Wrap a ``DimensionWiseCP`` with atom and separation-rank gates."""

    def __init__(self, base: nn.Module, initial_log_alpha: float = 3.0,
                 route_atoms: bool = True, route_ranks: bool = True):
        super().__init__()
        self.base = base
        self.route_atoms = bool(route_atoms)
        self.route_ranks = bool(route_ranks)
        self.rank_router = StepLockedHardConcrete(
            (base.rank,), initial_log_alpha
        )
        self.atom_routers = nn.ModuleList([
            StepLockedHardConcrete(
                (branch.dictionary.size, base.rank), initial_log_alpha
            ) for branch in base.branches
        ])

    @property
    def rank(self):
        return self.base.rank

    @property
    def branches(self):
        return self.base.branches

    def set_phase(self, phase: str, route_strength: float = 0.0) -> None:
        routed = phase == "route"
        strength = min(max(float(route_strength), 0.0), 1.0)
        self.rank_router.routing = routed and self.route_ranks
        for gate in self.atom_routers:
            gate.routing = routed and self.route_atoms
        for gate in [self.rank_router, *self.atom_routers]:
            gate.route_strength = strength

    def begin_routing_step(self) -> None:
        self.rank_router.begin_step()
        for gate in self.atom_routers:
            gate.begin_step()

    def _values(self, x: torch.Tensor, second_axis: int | None = None):
        rank_gate = self.rank_router()[None, :]
        values = []
        for axis, (branch, atom_gate) in enumerate(
                zip(self.branches, self.atom_routers)):
            coefficient = branch.effective_coefficient() * atom_gate()
            dictionary = (branch.dictionary.second_derivative(x[:, axis])
                          if axis == second_axis else
                          branch.dictionary(x[:, axis]))
            values.append((dictionary @ coefficient) * rank_gate)
        return values

    def effective_coefficients_by_axis(self):
        """Return gated coefficient matrices for projected-operator losses."""
        rank_gate = self.rank_router()[None, :]
        return [
            branch.effective_coefficient() * atom_gate() * rank_gate
            for branch, atom_gate in zip(self.branches, self.atom_routers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self._values(x)
        product = torch.ones_like(values[0])
        for value in values:
            product = product * value
        return product.sum(dim=1) / math.sqrt(self.rank)

    def forward_with_diagonal_second(self, x: torch.Tensor):
        values = self._values(x)
        product = torch.ones_like(values[0])
        for value in values:
            product = product * value
        prediction = product.sum(dim=1) / math.sqrt(self.rank)
        diagonal = []
        for differentiated_axis in range(len(values)):
            second_values = self._values(x, second_axis=differentiated_axis)
            term = torch.ones_like(values[0])
            for value in second_values:
                term = term * value
            diagonal.append(term.sum(dim=1) / math.sqrt(self.rank))
        return prediction, diagonal

    def capacity_penalty(self) -> torch.Tensor:
        rank_probability = self.rank_router.expected_l0()
        terms = []
        if self.route_ranks:
            terms.append(rank_probability.mean())
        # atom-only 消融中 rank 是固定容量，不能因原子开销项被
        # 间接优化。仅 two-level 模式才使用条件原子开销。
        rank_condition = (rank_probability[None, :]
                          if self.route_ranks else
                          torch.ones_like(rank_probability[None, :]))
        atom_costs = [
            (gate.expected_l0() * rank_condition).mean()
            for gate in self.atom_routers
        ]
        if self.route_atoms:
            terms.append(torch.stack(atom_costs).mean())
        return (torch.stack(terms).sum() if terms else
                rank_probability.new_zeros(()))

    @torch.no_grad()
    def consolidate(self, threshold: float, minimum_ranks: int,
                    minimum_atoms_per_rank: int) -> dict:
        if self.route_ranks:
            rank_mask = self.rank_router.consolidate(
                threshold, minimum_ranks
            ).bool()
        else:
            self.rank_router.fixed_mask.fill_(1.0)
            self.rank_router.consolidated = True
            self.rank_router.routing = False
            self.rank_router.log_alpha.requires_grad_(False)
            rank_mask = self.rank_router.fixed_mask.bool()
        for gate in self.atom_routers:
            probability = gate.expected_l0()
            if self.route_atoms:
                mask = (probability >= threshold) & rank_mask[None, :]
                for rank in torch.nonzero(rank_mask, as_tuple=False).flatten():
                    required = min(int(minimum_atoms_per_rank), mask.shape[0])
                    if int(mask[:, rank].sum()) < required:
                        chosen = torch.topk(probability[:, rank], required).indices
                        mask[chosen, rank] = True
            else:
                mask = torch.ones_like(probability, dtype=torch.bool)
                mask &= rank_mask[None, :]
            gate.fixed_mask.copy_(mask.to(gate.fixed_mask))
            gate.consolidated = True
            gate.routing = False
            gate._cached_sample = None
            gate.log_alpha.requires_grad_(False)
        return self.routing_statistics()

    def greedy_physics_rank_prune(self, physics_closure,
                                  relative_tolerance: float,
                                  absolute_target: float,
                                  minimum_ranks: int) -> dict:
        """Prune CP components by their locked physical-residual increase.

        A one-at-a-time ablation ranks candidates.  Deleting one component
        also redistributes its mean amplitude over the surviving components
        in exactly one branch.  This matters when optimization has represented
        one physical mode by several nearly identical CP components: a raw
        mask changes the total amplitude and falsely makes every duplicate
        appear indispensable.  Cumulative removals are accepted only while
        the independently evaluated physics loss remains below the declared
        target.  No reference solution is read.  Atom gates stay fully open
        so this experiment isolates rank routing.
        """
        self.rank_router.routing = False
        self.rank_router.consolidated = True
        self.rank_router.fixed_mask.fill_(1.0)
        self.rank_router.log_alpha.requires_grad_(False)
        for gate in self.atom_routers:
            gate.routing = False
            gate.consolidated = True
            gate.fixed_mask.fill_(1.0)
            gate.log_alpha.requires_grad_(False)

        baseline = float(physics_closure().detach())
        target = max(
            baseline * (1.0 + float(relative_tolerance)),
            float(absolute_target),
        )
        individual = []
        compensation_branch = self.branches[0]
        if compensation_branch.coefficient is None:
            raise RuntimeError(
                "physics-greedy rank compensation requires direct trainable "
                "branch coefficients"
            )

        def remove_with_compensation(rank: int) -> torch.Tensor:
            """Mask one rank and preserve its mean amplitude in branch zero."""
            active_count = int((active > 0.5).sum())
            snapshot = compensation_branch.coefficient.detach().clone()
            active[rank] = 0.0
            remaining = active > 0.5
            if active_count > 1:
                compensation_branch.coefficient[:, remaining].mul_(
                    active_count / (active_count - 1)
                )
            return snapshot

        def restore_removal(rank: int, snapshot: torch.Tensor) -> None:
            compensation_branch.coefficient.copy_(snapshot)
            active[rank] = 1.0

        with torch.no_grad():
            active = self.rank_router.fixed_mask
        for rank in range(self.rank):
            with torch.no_grad():
                snapshot = remove_with_compensation(rank)
            value = float(physics_closure().detach())
            with torch.no_grad():
                restore_removal(rank, snapshot)
            individual.append({
                "rank": rank, "ablated_physics": value,
                "increase": value - baseline,
            })

        accepted, rejected = [], []
        current = baseline
        for candidate in sorted(individual, key=lambda item: item["increase"]):
            if int((active > 0.5).sum()) <= int(minimum_ranks):
                break
            rank = candidate["rank"]
            with torch.no_grad():
                snapshot = remove_with_compensation(rank)
            trial = float(physics_closure().detach())
            if trial <= target:
                accepted.append({"rank": rank, "physics": trial})
                current = trial
            else:
                with torch.no_grad():
                    restore_removal(rank, snapshot)
                rejected.append({"rank": rank, "physics": trial})
        return {
            "amplitude_compensation": "survivor_scale_R_over_R_minus_1",
            "compensation_axis": 0,
            "baseline_physics": baseline,
            "allowed_physics": target,
            "post_prune_physics": current,
            "accepted_removals": accepted,
            "rejected_removals": rejected,
            "individual_ablation": individual,
            **self.routing_statistics(),
        }

    @torch.no_grad()
    def routing_statistics(self) -> dict:
        rank_values = (self.rank_router.fixed_mask
                       if self.rank_router.consolidated
                       else self.rank_router.expected_l0())
        rank_mask = rank_values >= 0.5
        atom_counts, maxima = [], []
        for gate in self.atom_routers:
            values = (gate.fixed_mask if gate.consolidated
                      else gate.expected_l0())
            mask = (values >= 0.5) & rank_mask[None, :]
            atom_counts.append(int(mask.sum()))
            maxima.append(mask.numel())
        return {
            "active_ranks": int(rank_mask.sum()),
            "maximum_ranks": int(rank_mask.numel()),
            "active_atoms_by_axis": atom_counts,
            "maximum_atoms_by_axis": maxima,
            "active_atoms": sum(atom_counts),
            "maximum_atoms": sum(maxima),
        }

    @property
    def trainable_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def fixed_count(self):
        return (sum(p.numel() for p in self.parameters()
                    if not p.requires_grad)
                + sum(b.numel() for b in self.buffers()))

    @torch.no_grad()
    def effective_parameter_count(self) -> int:
        statistics = self.routing_statistics()
        # After consolidation each surviving coefficient is an effective
        # degree of freedom; router logits are no longer counted.
        return statistics["active_atoms"]
