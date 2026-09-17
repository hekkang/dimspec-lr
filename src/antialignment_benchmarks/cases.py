"""Definitions of the non-manufactured H1, A1, and A2 replacements.

The prescribed PDE data are independent of the numerical reference.  A
reference field is loaded only by :meth:`AntiAlignmentCase.reference`; it is
never called by a training objective or checkpoint-selection criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch


Tensor = torch.Tensor
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT / "results/antialignment_replacements_v1/references"
FINAL_REFERENCE_ROOT = (
    ROOT / "results/antialignment_replacements_v1/candidate_references"
)


@dataclass(frozen=True)
class AntiAlignmentCase:
    name: str
    family: str
    bounds: tuple[tuple[float, float], tuple[float, float]]
    rank: int
    feature_kinds: tuple[str, str]
    max_modes: tuple[int, int]
    dictionary_parameters: tuple[float | None, float | None]
    coefficient: Callable[[Tensor], Tensor] | None
    forcing: Callable[[Tensor], Tensor]
    diffusion: Callable[[Tensor], Tensor] | None
    convection: Callable[[Tensor], Tensor] | None
    reaction: Callable[[Tensor], Tensor] | None
    boundary_kind: str
    reference_file: Path

    @property
    def dimension(self) -> int:
        return 2

    def boundary_envelope(self, points: Tensor) -> Tensor:
        normalized = []
        for axis, (lower, upper) in enumerate(self.bounds):
            value = (points[:, axis] - lower) / (upper - lower)
            normalized.append(4.0 * value * (1.0 - value))
        return normalized[0] * normalized[1]

    def boundary_lift(self, points: Tensor) -> Tensor:
        if self.boundary_kind != "left_gaussian":
            # Retain a zero-valued autograd path so the common differential
            # operator can evaluate the homogeneous lift without a special
            # branch.
            return 0.0 * points[:, 0]
        x, y = points.unbind(dim=1)
        gaussian = torch.exp(-150.0 * (y - 0.25).square())
        # The nominal Gaussian is nonzero by 8.5e-5 at the lower-left corner,
        # whereas the adjacent edge is prescribed zero.  Subtracting the
        # endpoint interpolant gives the unique continuous corner-compatible
        # trace; the change is negligible away from the two corners.
        g0 = math.exp(-150.0 * 0.25**2)
        g1 = math.exp(-150.0 * 0.75**2)
        compatible = gaussian - (1.0 - y) * g0 - y * g1
        return (1.0 - x) * compatible

    def reference(self, points: Tensor) -> Tensor:
        if not self.reference_file.exists():
            raise FileNotFoundError(
                f"numerical reference missing: {self.reference_file}; "
                "run scripts/generate_antialignment_references.py"
            )
        data = np.load(self.reference_file)
        field = torch.as_tensor(data["u"], device=points.device,
                                dtype=points.dtype)
        (x0, x1), (y0, y1) = self.bounds
        nx = field.shape[1]
        ny = field.shape[0]
        tx = ((points[:, 0] - x0) / (x1 - x0) * (nx - 1)).clamp(0, nx - 1)
        ty = ((points[:, 1] - y0) / (y1 - y0) * (ny - 1)).clamp(0, ny - 1)
        ix0 = tx.floor().long().clamp(max=nx - 2)
        iy0 = ty.floor().long().clamp(max=ny - 2)
        ix1, iy1 = ix0 + 1, iy0 + 1
        wx, wy = tx - ix0, ty - iy0
        return (
            (1.0 - wx) * (1.0 - wy) * field[iy0, ix0]
            + wx * (1.0 - wy) * field[iy0, ix1]
            + (1.0 - wx) * wy * field[iy1, ix0]
            + wx * wy * field[iy1, ix1]
        )


def _zeros(points: Tensor) -> Tensor:
    return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)


def h1(k0: float = 5.0,
       reference_file: Path | None = None) -> AntiAlignmentCase:
    def coefficient(points: Tensor) -> Tensor:
        x, y = points.unbind(dim=1)
        refractive_index = (
            1.0
            + 0.35 * torch.exp(-((x - 0.25).square() +
                                 (y + 0.15).square()) / 0.12)
            + 0.15 * torch.sin(2.0 * torch.pi * x * y)
        )
        return float(k0)**2 * refractive_index.square()

    def forcing(points: Tensor) -> Tensor:
        x, y = points.unbind(dim=1)
        radius = torch.sqrt((x + 0.45).square() + (y - 0.15).square() + 1e-16)
        return torch.exp(-120.0 * (radius - 0.18).square())

    return AntiAlignmentCase(
        name="H1", family="helmholtz", bounds=((-1.0, 1.0), (-1.0, 1.0)),
        rank=32, feature_kinds=("hybrid", "hybrid"), max_modes=(32, 32),
        dictionary_parameters=(None, None), coefficient=coefficient,
        forcing=forcing, diffusion=None, convection=None, reaction=None,
        boundary_kind="homogeneous",
        reference_file=(reference_file or FINAL_REFERENCE_ROOT /
                        "H1_k5/H1/reference.npz"),
    )


def a1(epsilon: float = 1.0e-2,
       reference_file: Path | None = None) -> AntiAlignmentCase:
    def diffusion(points: Tensor) -> Tensor:
        identity = torch.eye(2, device=points.device, dtype=points.dtype)
        return (float(epsilon) * identity).expand(points.shape[0], -1, -1)

    def convection(points: Tensor) -> Tensor:
        x, y = points.unbind(dim=1)
        return torch.stack((
            torch.sin(torch.pi * x) * torch.cos(torch.pi * y),
            -torch.cos(torch.pi * x) * torch.sin(torch.pi * y),
        ), dim=1)

    def reaction(points: Tensor) -> Tensor:
        return torch.full((points.shape[0],), 0.1, device=points.device,
                          dtype=points.dtype)

    return AntiAlignmentCase(
        name="A1", family="anisotropic_cdr", bounds=((0.0, 1.0), (0.0, 1.0)),
        rank=32, feature_kinds=("hybrid", "hybrid"), max_modes=(48, 64),
        dictionary_parameters=(None, None), coefficient=None, forcing=_zeros,
        diffusion=diffusion, convection=convection, reaction=reaction,
        boundary_kind="left_gaussian",
        reference_file=(reference_file or FINAL_REFERENCE_ROOT /
                        "A1_eps_0p01/A1/reference.npz"),
    )


def a2() -> AntiAlignmentCase:
    def theta(points: Tensor) -> Tensor:
        x, y = points.unbind(dim=1)
        return 0.5 * torch.pi * x + torch.pi / 6.0 * torch.sin(2.0 * torch.pi * y)

    def diffusion(points: Tensor) -> Tensor:
        angle = theta(points)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        dxx = cosine.square() + 0.01 * sine.square()
        dyy = sine.square() + 0.01 * cosine.square()
        dxy = 0.99 * cosine * sine
        return torch.stack((dxx, dxy, dxy, dyy), dim=1).reshape(-1, 2, 2)

    def reaction(points: Tensor) -> Tensor:
        return torch.full((points.shape[0],), 0.1, device=points.device,
                          dtype=points.dtype)

    def forcing(points: Tensor) -> Tensor:
        x, y = points.unbind(dim=1)
        ring = (x - 0.5).square() + (y - 0.5).square() - 0.22**2
        return torch.exp(-100.0 * ring.square())

    return AntiAlignmentCase(
        name="A2", family="anisotropic_cdr", bounds=((0.0, 1.0), (0.0, 1.0)),
        rank=32, feature_kinds=("hybrid", "hybrid"), max_modes=(32, 32),
        dictionary_parameters=(None, None), coefficient=None, forcing=forcing,
        diffusion=diffusion, convection=None, reaction=reaction,
        boundary_kind="homogeneous", reference_file=REFERENCE_ROOT / "A2/reference.npz",
    )


def get_case(name: str) -> AntiAlignmentCase:
    key = name.upper()
    constructors = {"H1": h1, "A1": a1, "A2": a2}
    if key not in constructors:
        raise ValueError(f"unknown anti-alignment benchmark {name!r}")
    return constructors[key]()
