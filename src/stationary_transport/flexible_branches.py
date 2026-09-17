"""Flexible one-dimensional branch families for low-rank transport PINNs."""

from __future__ import annotations

import math

import torch
from torch import nn

from dimensionwise_12case.model import FixedFeatureDictionary


def small_mlp(rank: int, width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend((nn.Linear(width, width), nn.Tanh()))
    layers.append(nn.Linear(width, rank))
    network = nn.Sequential(*layers)
    with torch.no_grad():
        network[-1].weight.mul_(0.05)
        network[-1].bias.zero_()
    return network


class FourierAxis(nn.Module):
    def __init__(self, modes: int, bounds, rank: int):
        super().__init__()
        self.dictionary = FixedFeatureDictionary(
            "fourier_periodic", modes, bounds[0], bounds[1]
        )
        self.coefficient = nn.Parameter(
            0.02 * torch.randn(self.dictionary.size, rank)
        )

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        return self.dictionary(coordinate) @ self.coefficient


class MLPAxis(nn.Module):
    def __init__(self, rank: int, width: int, depth: int):
        super().__init__()
        self.network = small_mlp(rank, width, depth)

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        return self.network(coordinate.reshape(-1, 1))


class ModifiedMLPAxis(nn.Module):
    """SPINN-style gated one-dimensional body network."""

    def __init__(self, rank: int, width: int, depth: int, bounds):
        super().__init__()
        self.register_buffer("lower", torch.tensor(float(bounds[0])))
        self.register_buffer("upper", torch.tensor(float(bounds[1])))
        self.encoder_u = nn.Linear(1, width)
        self.encoder_v = nn.Linear(1, width)
        self.input_layer = nn.Linear(1, width)
        self.gates = nn.ModuleList(
            nn.Linear(width, width) for _ in range(depth)
        )
        self.output_layer = nn.Linear(width, rank)

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        coordinate = (
            2.0*(coordinate-self.lower)/(self.upper-self.lower)-1.0
        ).reshape(-1, 1)
        encoded_u = torch.tanh(self.encoder_u(coordinate))
        encoded_v = torch.tanh(self.encoder_v(coordinate))
        hidden = torch.tanh(self.input_layer(coordinate))
        for layer in self.gates:
            gate = torch.tanh(layer(hidden))
            hidden = (1.0-gate)*encoded_u+gate*encoded_v
        return self.output_layer(hidden)


class FixedPhysicsAxis(nn.Module):
    def __init__(self, kind: str, modes: int, bounds, rank: int):
        super().__init__()
        self.dictionary = FixedFeatureDictionary(
            kind, modes, bounds[0], bounds[1]
        )
        self.coefficient = nn.Parameter(
            0.02 * torch.randn(self.dictionary.size, rank)
        )

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        return self.dictionary(coordinate) @ self.coefficient


class PhysicsCorrectionAxis(nn.Module):
    def __init__(self, kind: str, modes: int, bounds, rank: int,
                 width: int, depth: int):
        super().__init__()
        self.dictionary = FixedFeatureDictionary(
            kind, modes, bounds[0], bounds[1]
        )
        self.coefficient = nn.Parameter(
            0.02 * torch.randn(self.dictionary.size, rank)
        )
        self.correction = small_mlp(rank, width, depth)
        self.correction_log_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        fixed = self.dictionary(coordinate) @ self.coefficient
        scale = torch.nn.functional.softplus(self.correction_log_scale)
        return fixed + scale * self.correction(coordinate.reshape(-1, 1))


class FlexibleTransportCP(nn.Module):
    """Two one-dimensional branches coupled only through a CP rank sum."""

    def __init__(self, case, family: str, rank: int | None = None,
                 width: int = 48, depth: int = 2,
                 physics_modes: int | None = None):
        super().__init__()
        self.rank = int(rank or case.rank)
        mode0 = int(physics_modes or case.max_modes[0])
        mode1 = int(physics_modes or case.max_modes[1])
        if family in {"fixed_physics", "physics_physics"}:
            self.branches = nn.ModuleList([
                FixedPhysicsAxis(
                    case.feature_kinds[0], mode0, case.bounds[0],
                    self.rank,
                ),
                FixedPhysicsAxis(
                    "collision_eigen", mode1, case.bounds[1],
                    self.rank,
                ),
            ])
        elif family == "fourier":
            self.branches = nn.ModuleList([
                FourierAxis(case.max_modes[0], case.bounds[0], self.rank),
                FourierAxis(case.max_modes[1], case.bounds[1], self.rank),
            ])
        elif family in {"mlp", "mlp_mlp"}:
            self.branches = nn.ModuleList([
                MLPAxis(self.rank, width, depth),
                MLPAxis(self.rank, width, depth),
            ])
        elif family == "physics_mlp":
            self.branches = nn.ModuleList([
                FixedPhysicsAxis(
                    case.feature_kinds[0], mode0, case.bounds[0],
                    self.rank,
                ),
                MLPAxis(self.rank, width, depth),
            ])
        elif family == "mlp_physics":
            self.branches = nn.ModuleList([
                MLPAxis(self.rank, width, depth),
                FixedPhysicsAxis(
                    "collision_eigen", mode1, case.bounds[1],
                    self.rank,
                ),
            ])
        elif family == "modified_mlp":
            self.branches = nn.ModuleList([
                ModifiedMLPAxis(self.rank, width, depth, case.bounds[0]),
                ModifiedMLPAxis(self.rank, width, depth, case.bounds[1]),
            ])
        elif family == "physics_modified":
            # Preserve the material/interface-aware spatial branch while
            # giving the angular axis the well-conditioned gated SPINN body.
            self.branches = nn.ModuleList([
                PhysicsCorrectionAxis(
                    case.feature_kinds[0], mode0, case.bounds[0],
                    self.rank, width, depth,
                ),
                ModifiedMLPAxis(self.rank, width, depth, case.bounds[1]),
            ])
        elif family == "physics_correction":
            # The spatial basis follows geometry/material structure.  The
            # angular basis follows collision eigenmodes even for S1; its
            # trainable correction can represent the remaining micro part.
            self.branches = nn.ModuleList([
                PhysicsCorrectionAxis(
                    case.feature_kinds[0], mode0, case.bounds[0],
                    self.rank, width, depth,
                ),
                PhysicsCorrectionAxis(
                    "collision_eigen", mode1, case.bounds[1],
                    self.rank, width, depth,
                ),
            ])
        elif family == "homogeneous_uniform_correction":
            # Parameter-matched homogeneous control for S3.  Both coordinates
            # use the same hybrid dictionary, correction network, and atom
            # count.  With rank 16 and a width-48, depth-2 correction, M=36
            # yields 11,202 trainable parameters (0.4% above DimSpec S3).
            uniform_modes = 36
            self.branches = nn.ModuleList([
                PhysicsCorrectionAxis(
                    "hybrid", uniform_modes, case.bounds[0],
                    self.rank, width, depth,
                ),
                PhysicsCorrectionAxis(
                    "hybrid", uniform_modes, case.bounds[1],
                    self.rank, width, depth,
                ),
            ])
        elif family == "homogeneous_routed_correction":
            # Capacity-routed homogeneous control for S3.  Both coordinates
            # still use the same hybrid-plus-correction function class, while
            # the spatial axis receives more atoms.  The 64/7 split yields
            # 11,138 trainable parameters (0.1% below DimSpec S3).
            self.branches = nn.ModuleList([
                PhysicsCorrectionAxis(
                    "hybrid", 64, case.bounds[0],
                    self.rank, width, depth,
                ),
                PhysicsCorrectionAxis(
                    "hybrid", 7, case.bounds[1],
                    self.rank, width, depth,
                ),
            ])
        elif family in {"collision_width_uniform_correction", "collision_width_routed_correction"}:
            # Match parameters using neural capacity rather than arbitrarily
            # raising polynomial degree far beyond the angular quadrature.
            # Both coordinates retain C32; the width-62/62 and 74/48 choices
            # give 11,134 and 11,188 parameters at rank 16 (target 11,154).
            widths = ((62, 62) if family == "collision_width_uniform_correction"
                      else (74, 48))
            self.branches = nn.ModuleList([
                PhysicsCorrectionAxis("collision_eigen", 32, bounds,
                                      self.rank, axis_width, depth)
                for axis_width, bounds in zip(widths, case.bounds)
            ])
        elif family in {"collision_uniform_correction", "collision_routed_correction"}:
            # Match the H64/C32 dictionary's 260+33 features. Corrections,
            # rank, optimizer and inflow loss remain unchanged. C uses the
            # same raw-coordinate Legendre family on both axes; no change
            # of polynomial family or physics source is hidden in the control.
            counts = ((146, 145) if family == "collision_uniform_correction"
                      else (259, 32))
            self.branches = nn.ModuleList([
                PhysicsCorrectionAxis("collision_eigen", count, bounds,
                                      self.rank, width, depth)
                for count, bounds in zip(counts, case.bounds)
            ])
        elif family == "permuted_physics_correction":
            # Same rank, atom counts, neural corrections, and parameter count
            # as ``physics_correction``, but with the two dictionaries placed
            # on the wrong coordinates.  This isolates assignment quality
            # without introducing a capacity difference.
            self.branches = nn.ModuleList([
                PhysicsCorrectionAxis(
                    "collision_eigen", mode1, case.bounds[0],
                    self.rank, width, depth,
                ),
                PhysicsCorrectionAxis(
                    case.feature_kinds[0], mode0, case.bounds[1],
                    self.rank, width, depth,
                ),
            ])
        else:
            raise ValueError(f"unknown branch family {family!r}")
        self.family = family
        self.register_buffer("active_rank_mask", torch.ones(self.rank))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        left = self.branches[0](points[:, 0])
        right = self.branches[1](points[:, 1])
        return (left * right * self.active_rank_mask[None]).sum(dim=1) / math.sqrt(self.rank)

    def separable_factors(
        self, x: torch.Tensor, mu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the two CP axes without repeating the tensor grid.

        The pointwise ``forward`` method remains available for arbitrary
        coordinates and reference evaluation.  The transport residual is
        evaluated on a tensor-product quadrature, where evaluating the
        spatial branch once per angle (and vice versa) is unnecessary.
        """
        left = self.branches[0](x) * self.active_rank_mask[None]
        right = self.branches[1](mu)
        return left, right

    @property
    def trainable_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)
