"""Locked A1--A4 steady anisotropic convection--diffusion cases.

The exact fields are used only to manufacture the prescribed source and to
evaluate a trained solver.  No interior exact value is part of the training
loss.  Every case is defined on the unit square and has homogeneous Dirichlet
data, so the field can be hard constrained by a boundary envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class AnisotropicCase:
    name: str
    task_parameter: float
    task_kind: str
    exact: Callable[[Tensor], Tensor]
    diffusion: Callable[[Tensor], Tensor]
    convection: Callable[[Tensor], Tensor]
    reaction: Callable[[Tensor], Tensor]
    rank: int = 16
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    feature_kinds: tuple[str, ...] = ("hybrid", "hybrid")
    max_modes: tuple[int, ...] = (96, 96)
    dictionary_parameters: tuple[float | None, ...] = (None, None)

    @property
    def dimension(self) -> int:
        return 2

    def forcing(self, points: Tensor) -> Tensor:
        """Manufacture ``f=-div(D grad u)+b.grad(u)+c*u`` by autograd."""
        x = points.detach().clone().requires_grad_(True)
        value = self.exact(x)
        gradient = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
        hessian_columns = []
        for axis in range(2):
            hessian_columns.append(torch.autograd.grad(
                gradient[:, axis].sum(), x, create_graph=True,
            )[0])
        # H[:, i, j] = d_j d_i u; symmetrize to suppress roundoff only.
        hessian = torch.stack(hessian_columns, dim=1)
        hessian = 0.5 * (hessian + hessian.transpose(1, 2))
        diffusion = self.diffusion(x)
        convection = self.convection(x)
        reaction = self.reaction(x)
        operator = -(diffusion * hessian).sum(dim=(1, 2))
        operator = operator + (convection * gradient).sum(dim=1)
        operator = operator + reaction * value
        return operator.detach()


def _constant_coefficients(matrix, velocity=(0.0, 0.0), reaction=0.0):
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    velocity = torch.as_tensor(velocity, dtype=torch.float32)

    def diffusion(x: Tensor) -> Tensor:
        return matrix.to(x).expand(x.shape[0], -1, -1)

    def convection(x: Tensor) -> Tensor:
        return velocity.to(x).expand(x.shape[0], -1)

    def reaction_fn(x: Tensor) -> Tensor:
        return torch.full((x.shape[0],), float(reaction), device=x.device,
                          dtype=x.dtype)

    return diffusion, convection, reaction_fn


def a1(frequency: float = 16.0) -> AnisotropicCase:
    """Axis-aligned anisotropy with a single high-frequency y branch."""
    k = int(round(frequency))
    if k < 1 or k > 96:
        raise ValueError("A1 frequency must lie in [1, 96]")

    def exact(x: Tensor) -> Tensor:
        return torch.sin(torch.pi*x[:, 0])*torch.sin(k*torch.pi*x[:, 1])

    coefficients = _constant_coefficients(((1.0, 0.0), (0.0, 1.0/k**2)))
    return AnisotropicCase("A1", float(k), "frequency", exact, *coefficients,
                           rank=8, feature_kinds=("sine_pi", "sine_pi"))


def a2(signed_frequency: float = 16.0) -> AnisotropicCase:
    """Structural shift: the hard direction switches between x and y.

    A positive task value places that frequency along x; a negative value
    places its magnitude along y.  Sources use +8 and +12, while the held-out
    target is +16.  Thus ordinary scalar output interpolation cannot solve the
    axis-swap target.
    """
    signed = int(round(signed_frequency))
    if signed == 0 or abs(signed) > 96:
        raise ValueError("A2 signed frequency must be in [-96,-1] or [1,96]")
    axis = 0 if signed > 0 else 1
    k = abs(signed)

    def exact(x: Tensor) -> Tensor:
        modes = (k, 1) if axis == 0 else (1, k)
        return (torch.sin(modes[0]*torch.pi*x[:, 0])
                * torch.sin(modes[1]*torch.pi*x[:, 1]))

    diagonal = (1.0/k**2, 1.0) if axis == 0 else (1.0, 1.0/k**2)
    coefficients = _constant_coefficients(((diagonal[0], 0.0),
                                            (0.0, diagonal[1])))
    return AnisotropicCase("A2", float(signed), "axis_frequency", exact,
                           *coefficients, rank=8,
                           feature_kinds=("sine_pi", "sine_pi"))


def a3(angle_degrees: float = 45.0) -> AnisotropicCase:
    """Rotated anisotropic tensor with an oblique oscillatory solution."""
    theta = math.radians(float(angle_degrees))
    cosine, sine = math.cos(theta), math.sin(theta)
    rotation = torch.tensor(((cosine, -sine), (sine, cosine)))
    eigenvalues = torch.diag(torch.tensor((1.0, 0.01)))
    matrix = rotation @ eigenvalues @ rotation.T

    def exact(x: Tensor) -> Tensor:
        envelope = x[:, 0]*(1.0-x[:, 0])*x[:, 1]*(1.0-x[:, 1])
        oblique = cosine*x[:, 0] + sine*x[:, 1]
        return 32.0*envelope*torch.sin(12.0*torch.pi*oblique)

    coefficients = _constant_coefficients(matrix, velocity=(0.25, -0.10),
                                            reaction=0.5)
    return AnisotropicCase("A3", float(angle_degrees), "rotation", exact,
                           *coefficients, rank=24,
                           feature_kinds=("steerable_x", "steerable_y"),
                           max_modes=(18, 18))


def a4(epsilon: float = 2.5e-3) -> AnisotropicCase:
    """Convection-dominated problem with a thin outflow boundary layer.

    ``expm1`` keeps the expression stable: no term of the form ``exp(1/eps)``
    is ever evaluated.
    """
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    def exact(x: Tensor) -> Tensor:
        stream = x[:, 0]
        layer = -torch.expm1((stream-1.0)/epsilon)
        normalization = -math.expm1(-1.0/epsilon)
        return stream*(layer/normalization)*torch.sin(torch.pi*x[:, 1])

    coefficients = _constant_coefficients(
        ((epsilon, 0.0), (0.0, epsilon)), velocity=(1.0, 0.25),
    )
    return AnisotropicCase("A4", epsilon, "diffusion", exact, *coefficients,
                           rank=24,
                           feature_kinds=("boundary_layer", "hybrid"),
                           max_modes=(64, 32),
                           dictionary_parameters=(epsilon, None))


def get_case(name: str, parameter: float | None = None) -> AnisotropicCase:
    name = name.upper()
    defaults = {"A1": 16.0, "A2": 16.0, "A3": 45.0, "A4": 2.5e-3}
    if name not in defaults:
        raise ValueError(f"unknown steady anisotropic case {name!r}")
    value = defaults[name] if parameter is None else float(parameter)
    return {"A1": a1, "A2": a2, "A3": a3, "A4": a4}[name](value)
