"""Fixed-mapping latent coefficient and fully trainable separable controls."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn


def _seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


class FixedFeatureDictionary(nn.Module):
    """One-dimensional fixed polynomial/Fourier dictionary."""

    def __init__(self, kind: str, maximum_mode: int,
                 lower: float, upper: float,
                 physics_parameter: float | None = None):
        super().__init__()
        self.kind = kind
        self.maximum_mode = int(maximum_mode)
        if kind == "sine_pi":
            self.size = self.maximum_mode
        elif kind == "fourier_periodic":
            # Constant plus periodic sine/cosine atoms.  Unlike ``hybrid``
            # this dictionary cannot leak non-periodic polynomial terms into
            # the spatial branch of a periodic kinetic equation.
            self.size = 1 + 2 * self.maximum_mode
        elif kind == "half_range_chebyshev":
            # Independent degree-0..M polynomial systems on (-1,0) and
            # (0,1).  Half-range transport solutions need not be continuous
            # through mu=0 because their inflow traces originate at opposite
            # boundaries.
            self.size = 2*(self.maximum_mode+1)
        elif kind in {"collision_eigen", "legendre"}:
            # Legendre collision eigenfunctions P_l(mu).  For a kernel
            # represented by Legendre moments, these separate the slowly
            # damped macro modes from the micro angular complement.
            self.size = self.maximum_mode + 1
        elif kind in {"steerable_x", "steerable_y"}:
            # A fixed steerable frame.  Every atom remains one-dimensional;
            # oblique waves are recovered by low-rank sine/cosine products.
            # ``maximum_mode`` controls the number of angular intervals.  The
            # old implementation hard-coded 19 directions, so compact,
            # aligned, and rich A3 profiles instantiated the same dictionary
            # even though their recorded capacities differed.  Including both
            # endpoints gives M+1 directions and makes the profile contract
            # effective without changing the aligned M=18 dictionary.
            self.steerable_angles = tuple(
                90.0 * index / self.maximum_mode
                for index in range(self.maximum_mode + 1)
            )
            self.steerable_carrier = 12.0 * math.pi
            self.size = 2 * len(self.steerable_angles)
        elif kind in {"hybrid", "boundary_layer", "interface_hybrid"}:
            # 1,x,x^2,x^3 plus sin/cos(n*x) and sin/cos(n*pi*x).
            self.size = 4 + 4 * self.maximum_mode
            self.boundary_layer_scales = (
                0.00125, 0.0025, 0.005, 0.01, 0.02,
                0.04, 0.08, 0.16, 0.32,
            ) if kind == "boundary_layer" else ()
            if kind == "boundary_layer" and physics_parameter is not None:
                # The final coordinate is an operator-conditioned atom.  Its
                # semantic role (the current diffusion length) is identical
                # across source and target tasks even though its shape changes
                # continuously with epsilon.  It is fixed, not trainable.
                self.boundary_layer_scales += (float(physics_parameter),)
            self.size += 3*len(self.boundary_layer_scales)
            self.interface_edges = (0.25, 0.75) if kind == "interface_hybrid" else ()
            self.size += 3*len(self.interface_edges)
        else:
            raise ValueError(kind)
        self.lower = float(lower)
        self.upper = float(upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        modes = torch.arange(1, self.maximum_mode + 1, device=x.device, dtype=x.dtype)[None]
        if self.kind == "sine_pi":
            return torch.sin(torch.pi * x * modes)
        if self.kind == "fourier_periodic":
            phase = 2.0 * torch.pi * (x - self.lower) / (self.upper - self.lower)
            return torch.cat((
                torch.ones_like(x), torch.sin(phase * modes),
                torch.cos(phase * modes),
            ), dim=1)
        if self.kind == "half_range_chebyshev":
            positive = x > 0.0
            # Map each open half range onto [-1,1].  Gauss--Legendre angular
            # nodes never contain zero, so the arbitrary convention at zero
            # cannot enter either the PDE quadrature or evaluation metric.
            coordinate = torch.where(positive, 2.0*x-1.0, 2.0*x+1.0)
            values = [torch.ones_like(coordinate)]
            if self.maximum_mode:
                values.append(coordinate)
            for _ in range(2, self.maximum_mode+1):
                values.append(2.0*coordinate*values[-1]-values[-2])
            polynomial = torch.cat(values, dim=1)
            return torch.cat((
                polynomial*positive.to(x.dtype),
                polynomial*(~positive).to(x.dtype),
            ), dim=1)
        if self.kind in {"collision_eigen", "legendre"}:
            coordinate = (x if self.kind == "collision_eigen" else
                          2.0*(x-self.lower)/(self.upper-self.lower)-1.0)
            values = [torch.ones_like(x)]
            if self.maximum_mode:
                values.append(coordinate)
            for degree in range(2, self.maximum_mode+1):
                values.append(((2*degree-1)*coordinate*values[-1]
                               -(degree-1)*values[-2])/degree)
            return torch.cat(values, dim=1)
        if self.kind in {"steerable_x", "steerable_y"}:
            angles = torch.tensor(
                self.steerable_angles, device=x.device, dtype=x.dtype,
            )[None, :] * (torch.pi/180.0)
            projection = (torch.cos(angles) if self.kind == "steerable_x"
                          else torch.sin(angles))
            phase = self.steerable_carrier*x*projection
            envelope = ((x-self.lower)*(self.upper-x)
                        /(self.upper-self.lower)**2)
            return torch.cat((envelope*torch.sin(phase),
                              envelope*torch.cos(phase)), dim=1)
        # Polynomial features use a bounded coordinate even when t spans
        # [0,10]; trigonometric features retain physical coordinates so
        # published frequencies such as cos(2t) remain exact dictionary atoms.
        normalized = 2.0 * (x - self.lower) / (self.upper - self.lower) - 1.0
        polynomial = torch.cat((
            torch.ones_like(x), normalized, normalized.square(), normalized.pow(3)
        ), dim=1)
        base = torch.cat((
            polynomial,
            torch.sin(x * modes), torch.cos(x * modes),
            torch.sin(torch.pi * x * modes), torch.cos(torch.pi * x * modes),
        ), dim=1)
        if self.kind == "interface_hybrid":
            unit = (x-self.lower)/(self.upper-self.lower)
            width = 0.04
            local = []
            for edge in self.interface_edges:
                offset = unit-edge
                local.extend((torch.relu(offset),
                              torch.exp(-offset.abs()/width),
                              torch.tanh(offset/width)))
            return torch.cat((base, *local), dim=1)
        if self.kind != "boundary_layer":
            return base
        unit = (x-self.lower)/(self.upper-self.lower)
        scales = x.new_tensor(self.boundary_layer_scales)[None, :]
        exponential = torch.exp((unit-1.0)/scales)
        layer_atoms = torch.cat((
            exponential,
            unit*exponential,
            unit*(1.0-exponential),
        ), dim=1)
        return torch.cat((base, layer_atoms), dim=1)

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Return analytic second derivatives of all fixed atoms.

        The dictionaries never change during training, so differentiating the
        complete CP graph twice with autograd is unnecessary.  Besides being
        faster, the analytic form avoids noisy high-order graphs for the
        highest Fourier modes.
        """
        x = x.reshape(-1, 1)
        modes = torch.arange(
            1, self.maximum_mode + 1, device=x.device, dtype=x.dtype
        )[None]
        if self.kind == "sine_pi":
            atoms = torch.sin(torch.pi*x*modes)
            return -(torch.pi*modes).square()*atoms
        if self.kind == "fourier_periodic":
            phase = 2.0*torch.pi*(x-self.lower)/(self.upper-self.lower)
            frequencies = 2.0*torch.pi*modes/(self.upper-self.lower)
            zeros = torch.zeros_like(x)
            return torch.cat((
                zeros,
                -frequencies.square()*torch.sin(phase*modes),
                -frequencies.square()*torch.cos(phase*modes),
            ), dim=1)
        if self.kind == "half_range_chebyshev":
            raise NotImplementedError(
                "half-range angular atoms are not used with spatial Hessians"
            )
        if self.kind in {"collision_eigen", "legendre"}:
            raise NotImplementedError(
                "Legendre dictionaries are differentiated through autograd"
            )
        if self.kind in {"steerable_x", "steerable_y"}:
            angles = torch.tensor(
                self.steerable_angles, device=x.device, dtype=x.dtype,
            )[None, :] * (torch.pi/180.0)
            projection = (torch.cos(angles) if self.kind == "steerable_x"
                          else torch.sin(angles))
            frequency = self.steerable_carrier*projection
            length = self.upper-self.lower
            envelope = (x-self.lower)*(self.upper-x)/length**2
            envelope_first = (self.lower+self.upper-2*x)/length**2
            envelope_second = -2.0/length**2
            phase = frequency*x
            sine_second = (envelope_second*torch.sin(phase)
                           +2*envelope_first*frequency*torch.cos(phase)
                           -envelope*frequency.square()*torch.sin(phase))
            cosine_second = (envelope_second*torch.cos(phase)
                             -2*envelope_first*frequency*torch.sin(phase)
                             -envelope*frequency.square()*torch.cos(phase))
            return torch.cat((sine_second, cosine_second), dim=1)
        scale = 2.0/(self.upper-self.lower)
        normalized = scale*(x-self.lower)-1.0
        polynomial_second = torch.cat((
            torch.zeros_like(x), torch.zeros_like(x),
            torch.full_like(x, 2.0*scale**2),
            6.0*scale**2*normalized,
        ), dim=1)
        base = torch.cat((
            polynomial_second,
            -modes.square()*torch.sin(x*modes),
            -modes.square()*torch.cos(x*modes),
            -(torch.pi*modes).square()*torch.sin(torch.pi*x*modes),
            -(torch.pi*modes).square()*torch.cos(torch.pi*x*modes),
        ), dim=1)
        if self.kind == "interface_hybrid":
            # The transport solver differentiates this branch with autograd;
            # no elliptic case requests its classical second derivative.
            raise NotImplementedError(
                "interface_hybrid has a weak interface derivative only"
            )
        if self.kind != "boundary_layer":
            return base
        unit = (x-self.lower)/(self.upper-self.lower)
        scales = x.new_tensor(self.boundary_layer_scales)[None, :]
        inverse_length_squared = 1.0/(self.upper-self.lower)**2
        exponential = torch.exp((unit-1.0)/scales)
        exponential_second = exponential/scales.square()*inverse_length_squared
        unit_exponential_second = exponential*(
            unit/scales.square()+2.0/scales
        )*inverse_length_squared
        layer_second = torch.cat((
            exponential_second,
            unit_exponential_second,
            -unit_exponential_second,
        ), dim=1)
        return torch.cat((base, layer_second), dim=1)


class DimensionBranch(nn.Module):
    """A branch whose coefficient tensor is direct or a fixed map of a latent.

    `latent_fullspan` uses a fixed square orthogonal map.  It trains only the
    latent vector while retaining the complete coefficient space, making it a
    clean implementation validation before compressing latent dimensionality.
    """

    def __init__(self, name: str, kind: str, maximum_mode: int,
                 bounds, rank: int, variant: str,
                 latent_fraction: float, seed: int,
                 latent_initialization_seed: int | None = None,
                 mapping_name: str | None = None,
                 physics_parameter: float | None = None):
        super().__init__()
        self.dictionary = FixedFeatureDictionary(
            kind, maximum_mode, *bounds,
            physics_parameter=physics_parameter,
        )
        coefficient_size = self.dictionary.size * rank
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed(name, seed))
        # Each branch starts with O(1) rather than near-zero output.  A tiny
        # branch scale is multiplied once per physical dimension and otherwise
        # produces vanishing gradients in 3-D/4-D CP products.
        base = torch.randn(coefficient_size, generator=generator) * (0.5 / math.sqrt(self.dictionary.size))
        self.register_buffer("base", base.reshape(self.dictionary.size, rank))
        self.variant = variant
        base_variant = (variant.removesuffix("_aligned")
                        .removesuffix("_implicit")
                        .removesuffix("_shared_random")
                        .removesuffix("_sharedmap").removesuffix("_shared"))
        if base_variant == "trainable_coefficients":
            self.coefficient = nn.Parameter(self.base.clone())
            self.latent = None
        elif base_variant in {"latent_fullspan", "latent_compressed"}:
            latent_size = coefficient_size if base_variant == "latent_fullspan" else max(
                1, int(round(coefficient_size * latent_fraction))
            )
            if variant.endswith("_implicit"):
                # The fixed identity map is represented algebraically instead
                # of materializing an O(n^2) eye matrix.  The trainable object
                # is still the latent displacement, not the frozen base.
                mapping = torch.empty(0, dtype=torch.float64)
            elif variant.endswith("_aligned"):
                # Fixed optimizer-aligned map used to separate basic
                # latent-only solvability from random-manifold conditioning.
                mapping = torch.eye(coefficient_size, latent_size, dtype=torch.float64)
            else:
                mapping_generator = torch.Generator(device="cpu")
                mapping_generator.manual_seed(_seed(mapping_name or name, seed))
                raw = torch.randn(
                    coefficient_size, latent_size,
                    generator=mapping_generator, dtype=torch.float64,
                )
                mapping, _ = torch.linalg.qr(raw, mode="reduced")
            self.register_buffer("mapping", mapping.float())
            self.latent = nn.Parameter(torch.zeros(latent_size))
            if variant.endswith("_shared_random"):
                latent_generator = torch.Generator(device="cpu")
                latent_generator.manual_seed(int(latent_initialization_seed or seed))
                with torch.no_grad():
                    self.latent.copy_(0.05 * torch.randn(latent_size, generator=latent_generator))
            self.coefficient = None
        else:
            raise ValueError(variant)
        self.rank = rank
        # A compressed Mapping defines a nested maximum subspace.  Keeping the
        # active prefix in the checkpoint lets a residual controller expand
        # capacity without rebuilding the module or optimizer.
        if self.latent is not None:
            self.register_buffer(
                "active_latent_size",
                torch.tensor(self.latent.numel(), dtype=torch.long),
            )

    def set_active_latent_fraction(self, fraction: float) -> int:
        """Activate a prefix sized as a fraction of the full coefficients."""
        if self.latent is None:
            return 0
        requested = max(1, int(round(self.base.numel()*fraction)))
        self.active_latent_size.fill_(min(requested, self.latent.numel()))
        return int(self.active_latent_size)

    def grow_active_latent(self, coefficient_fraction: float) -> tuple[int, int]:
        """Grow the nested active prefix and return old/new dimensions."""
        if self.latent is None:
            return 0, 0
        old = int(self.active_latent_size)
        increment = max(1, int(round(self.base.numel()*coefficient_fraction)))
        new = min(self.latent.numel(), old+increment)
        self.active_latent_size.fill_(new)
        return old, new

    @property
    def active_latent_count(self) -> int:
        return 0 if self.latent is None else int(self.active_latent_size)

    def effective_coefficient(self):
        if self.coefficient is not None:
            return self.coefficient
        if self.variant.endswith("_implicit"):
            displacement = self.latent
        else:
            active = int(self.active_latent_size)
            displacement = (self.mapping[:, :active].to(self.latent)
                            @ self.latent[:active])
        return self.base + displacement.reshape_as(self.base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dictionary(x) @ self.effective_coefficient()

    def forward_second(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the branch's analytic second spatial derivative."""
        return self.dictionary.second_derivative(x) @ self.effective_coefficient()


class DimensionWiseCP(nn.Module):
    """CP product of dimension-specific fixed dictionaries."""

    def __init__(self, case, variant: str, latent_fraction: float, seed: int):
        super().__init__()
        self.rank = case.rank
        # ``rank`` is the preallocated maximum.  Existing experiments keep the
        # default full prefix; residual-Jacobian GFR experiments reduce this
        # buffer and therefore grow the actual CP separation rank, not merely
        # the number of active Mapping coordinates.
        self.register_buffer("active_separation_rank",
                             torch.tensor(case.rank, dtype=torch.long),
                             persistent=False)
        shared_target = variant.endswith("_shared") or variant.endswith("_shared_random")
        shared_map_only = variant.endswith("_sharedmap")
        self.branches = nn.ModuleList([
            DimensionBranch(
                (f"type_{kind}_modes_{modes}_rank_{case.rank}"
                 if shared_target else f"axis_{axis}"),
                kind, modes, case.bounds[axis], case.rank, variant,
                latent_fraction, seed,
                latent_initialization_seed=_seed(f"latent_axis_{axis}", seed),
                mapping_name=(f"type_{kind}_modes_{modes}_rank_{case.rank}"
                              if shared_map_only else None),
                physics_parameter=(getattr(case, "dictionary_parameters", (None,)*case.dimension)[axis]),
            )
            for axis, (kind, modes) in enumerate(zip(case.feature_kinds, case.max_modes))
        ])

    @property
    def active_rank(self) -> int:
        return int(self.active_separation_rank)

    def set_active_rank(self, value: int) -> None:
        if not 1 <= int(value) <= self.rank:
            raise ValueError("active separation rank outside preallocated range")
        self.active_separation_rank.fill_(int(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.active_rank
        product = torch.ones(x.shape[0], active, device=x.device, dtype=x.dtype)
        for axis, branch in enumerate(self.branches):
            product = product * branch(x[:, axis])[:, :active]
        return product.sum(dim=1) / math.sqrt(self.rank)

    def forward_with_diagonal_second(self, x: torch.Tensor):
        """Evaluate the field and every diagonal Hessian entry analytically."""
        active = self.active_rank
        values = [branch(x[:, axis])[:, :active]
                  for axis, branch in enumerate(self.branches)]
        seconds = [
            branch.forward_second(x[:, axis])[:, :active]
            for axis, branch in enumerate(self.branches)
        ]
        product = torch.ones_like(values[0])
        for value in values:
            product = product*value
        prediction = product.sum(dim=1)/math.sqrt(self.rank)
        diagonal = []
        for differentiated_axis in range(len(values)):
            term = torch.ones_like(values[0])
            for axis, value in enumerate(values):
                term = term*(seconds[axis] if axis == differentiated_axis else value)
            diagonal.append(term.sum(dim=1)/math.sqrt(self.rank))
        return prediction, diagonal

    @property
    def trainable_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def fixed_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad) + sum(
            buffer.numel() for buffer in self.buffers()
        )
