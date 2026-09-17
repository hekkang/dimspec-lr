"""Locked S1--S4 stationary radiative/particle transport definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TransportCase:
    name: str
    parameter: float
    task_kind: str
    legendre_coefficients: tuple[float, ...]
    material_edges: tuple[float, ...]
    scattering_values: tuple[float, ...]
    absorption_values: tuple[float, ...]
    left_inflow: float = 1.0
    right_inflow: float = 0.0
    rank: int = 16
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (-1.0, 1.0))
    feature_kinds: tuple[str, ...] = ("hybrid", "hybrid")
    max_modes: tuple[int, ...] = (64, 32)
    scattering_profile: str = "piecewise"
    moment_precondition_weight: float = 0.0
    lifting_kind: str = "linear"

    @property
    def dimension(self) -> int:
        return 2

    def _material_index_torch(self, x: torch.Tensor) -> torch.Tensor:
        if not self.material_edges:
            return torch.zeros_like(x, dtype=torch.long)
        edges = torch.tensor(self.material_edges, device=x.device, dtype=x.dtype)
        return torch.bucketize(x.contiguous(), edges)

    def coefficients_torch(self, x: torch.Tensor):
        if self.scattering_profile == "linear_x":
            scattering = self.parameter*x
            return scattering, torch.zeros_like(scattering)
        index = self._material_index_torch(x)
        scattering = torch.tensor(self.scattering_values, device=x.device,
                                  dtype=x.dtype)[index]
        absorption = torch.tensor(self.absorption_values, device=x.device,
                                  dtype=x.dtype)[index]
        return scattering, absorption

    def coefficients_numpy(self, x: np.ndarray):
        if self.scattering_profile == "linear_x":
            scattering = self.parameter*x
            return scattering, np.zeros_like(scattering)
        index = np.searchsorted(np.asarray(self.material_edges), x, side="right")
        scattering = np.asarray(self.scattering_values)[index]
        absorption = np.asarray(self.absorption_values)[index]
        return scattering, absorption

    def optical_depth_torch(self, x: torch.Tensor, from_left: bool = True):
        """Integrated total cross section from the selected slab boundary."""
        if self.scattering_profile == "linear_x":
            left = 0.5*self.parameter*x.square()
            total_integral = 0.5*self.parameter
            return left if from_left else total_integral-left
        edges = (0.0, *self.material_edges, 1.0)
        totals = tuple(s+a for s, a in zip(
            self.scattering_values, self.absorption_values,
        ))
        left = torch.zeros_like(x)
        total_integral = 0.0
        for index, coefficient in enumerate(totals):
            lower, upper = edges[index], edges[index+1]
            left = left+coefficient*(x.clamp(lower, upper)-lower)
            total_integral += coefficient*(upper-lower)
        return left if from_left else total_integral-left


PUBLISHED_LEGENDRE = (
    1.0, 1.98398, 1.50823, 0.70075,
    0.23489, 0.05133, 0.00760, 0.00048,
)


def s1(scale: float = 1.0) -> TransportCase:
    """Mishra--Molinaro 1-D monochromatic stationary benchmark.

    The published coefficient is sigma(x)=x.  ``scale`` is exposed only for
    source-task construction; the locked target uses one.
    """
    scale = float(scale)
    return TransportCase("S1", scale, "scattering_scale",
                         PUBLISHED_LEGENDRE, (), (scale,), (0.0,),
                         scattering_profile="linear_x", lifting_kind="linear")


def s2(scattering_ratio: float = 0.80) -> TransportCase:
    """Homogeneous strong-scattering transport target.

    The singularly perturbed c=0.98 near-diffusion case is retained as an
    explicit failure/extension test because it requires an AP micro--macro
    formulation rather than the direct residual shared by the main cases.
    """
    ratio = float(scattering_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("scattering ratio must lie in (0,1)")
    total = 1.0
    return TransportCase("S2", ratio, "scattering_ratio", (1.0,), (),
                         (total*ratio,), (total*(1.0-ratio),),
                         feature_kinds=("hybrid", "collision_eigen"),
                         moment_precondition_weight=1.0,
                         lifting_kind="uncollided")


def s3(anisotropy: float = 0.6) -> TransportCase:
    """P1 angular-anisotropy extrapolation with p=1+3g mu mu'."""
    g = float(anisotropy)
    if abs(g) >= 1.0/3.0:
        # For the unnormalized P1 convention used here, d1=3g and positivity
        # requires |d1|<=1.  Keep the public parameter in the physical range.
        raise ValueError("S3 anisotropy must satisfy |g| < 1/3")
    return TransportCase("S3", g, "angular_anisotropy", (1.0, 3.0*g), (),
                         (3.0,), (0.3,),
                         feature_kinds=("hybrid", "collision_eigen"),
                         moment_precondition_weight=1.0,
                         lifting_kind="uncollided")


def s4(contrast: float = 8.0) -> TransportCase:
    """Three-material slab with a high-scattering middle region."""
    contrast = float(contrast)
    if contrast <= 0:
        raise ValueError("contrast must be positive")
    return TransportCase("S4", contrast, "material_contrast", (1.0,),
                         (0.25, 0.75), (1.0, contrast, 1.0),
                         (0.5, 0.1, 0.5), rank=24,
                         feature_kinds=("interface_hybrid", "collision_eigen"),
                         moment_precondition_weight=1.0,
                         lifting_kind="uncollided")


def get_case(name: str, parameter: float | None = None) -> TransportCase:
    name = name.upper()
    defaults = {"S1": 1.0, "S2": 0.80, "S3": 0.30, "S4": 8.0}
    if name not in defaults:
        raise ValueError(f"unknown stationary transport case {name!r}")
    value = defaults[name] if parameter is None else float(parameter)
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}[name](value)
