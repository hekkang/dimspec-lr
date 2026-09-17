"""Analytic Helmholtz and Klein--Gordon benchmark definitions.

Reference solutions are deliberately kept in this evaluation/case module.  The
trainer uses them only for manufactured forcing, physical initial/boundary
values, and metrics; no interior reference values enter the optimization loss.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ManufacturedCase:
    name: str
    family: str
    bounds: tuple[tuple[float, float], ...]
    rank: int
    exact: Callable[[Tensor], Tensor]
    forcing: Callable[[Tensor], Tensor]
    feature_kinds: tuple[str, ...]
    max_modes: tuple[int, ...]
    time_axis: int | None = None
    coefficient: Callable[[Tensor], Tensor] | None = None

    @property
    def dimension(self) -> int:
        return len(self.bounds)


def _helmholtz_product_frequencies(name: str, frequencies: Sequence[int],
                                   bounds, rank=4) -> ManufacturedCase:
    frequencies = tuple(int(v) for v in frequencies)

    def exact(x: Tensor) -> Tensor:
        value = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        for axis, frequency in enumerate(frequencies):
            value = value * torch.sin(frequency * torch.pi * x[:, axis])
        return value

    eigenvalue = sum((frequency * torch.pi) ** 2 for frequency in frequencies)

    def forcing(x: Tensor) -> Tensor:
        return (1.0 - eigenvalue) * exact(x)

    return ManufacturedCase(
        name=name, family="helmholtz", bounds=tuple(bounds), rank=rank,
        exact=exact, forcing=forcing,
        feature_kinds=tuple("sine_pi" for _ in frequencies),
        max_modes=tuple(max(16, frequency) for frequency in frequencies),
    )


def h1() -> ManufacturedCase:
    return _helmholtz_product_frequencies(
        "H1", (4, 4, 3), ((-1.0, 1.0),) * 3, rank=4
    )


def h2() -> ManufacturedCase:
    """Off-dictionary variable-coefficient Helmholtz benchmark.

    This is the former HX2 stress test promoted to the formal H2 benchmark.
    The non-integer carrier frequencies prevent exact lookup in the locked
    sine dictionary, while the spatially varying reaction coefficient removes
    the constant-symbol shortcut available to the earlier product mode.
    """
    terms = (
        (1.0, 9.5, 3.75),
        (0.35, 4.25, 12.5),
    )

    def axis(value: Tensor, frequency: float) -> tuple[Tensor, Tensor]:
        wave = frequency * torch.pi
        sine = torch.sin(wave * value)
        cosine = torch.cos(wave * value)
        envelope = value * (1.0 - value)
        second = (
            -2.0 * sine
            + 2.0 * (1.0 - 2.0 * value) * wave * cosine
            - envelope * (wave ** 2) * sine
        )
        return envelope * sine, second

    def exact_and_laplacian(x: Tensor) -> tuple[Tensor, Tensor]:
        x0, x1 = x.unbind(dim=1)
        value = torch.zeros_like(x0)
        laplacian = torch.zeros_like(x0)
        for weight, frequency0, frequency1 in terms:
            value0, second0 = axis(x0, frequency0)
            value1, second1 = axis(x1, frequency1)
            value = value + weight * value0 * value1
            laplacian = laplacian + weight * (
                second0 * value1 + value0 * second1
            )
        return value, laplacian

    def exact(x: Tensor) -> Tensor:
        return exact_and_laplacian(x)[0]

    def coefficient(x: Tensor) -> Tensor:
        x0, x1 = x.unbind(dim=1)
        return (
            1.0
            + 0.75 * torch.sin(2.0 * torch.pi * x0)
            * torch.cos(2.0 * torch.pi * x1)
        )

    def forcing(x: Tensor) -> Tensor:
        value, laplacian = exact_and_laplacian(x)
        return laplacian + coefficient(x) * value

    return ManufacturedCase(
        name="H2", family="helmholtz", bounds=((0.0, 1.0),) * 2,
        rank=12, exact=exact, forcing=forcing,
        feature_kinds=("sine_pi", "sine_pi"), max_modes=(64, 64),
        coefficient=coefficient,
    )


def h3() -> ManufacturedCase:
    return _helmholtz_product_frequencies(
        "H3", (1, 16), ((0.0, 1.0),) * 2, rank=4
    )


def h4() -> ManufacturedCase:
    pairs = ((1, 16), (2, 8), (4, 4), (8, 2))

    def exact(x: Tensor) -> Tensor:
        return sum(
            torch.sin(a * torch.pi * x[:, 0])
            * torch.sin(b * torch.pi * x[:, 1]) / float(index)
            for index, (a, b) in enumerate(pairs, start=1)
        )

    def forcing(x: Tensor) -> Tensor:
        return sum(
            (1.0 - (a * torch.pi) ** 2 - (b * torch.pi) ** 2)
            * torch.sin(a * torch.pi * x[:, 0])
            * torch.sin(b * torch.pi * x[:, 1]) / float(index)
            for index, (a, b) in enumerate(pairs, start=1)
        )

    return ManufacturedCase(
        "H4", "helmholtz", ((0.0, 1.0),) * 2, 8, exact, forcing,
        ("sine_pi", "sine_pi"), (16, 16),
    )


def _kg_case(name: str, dimension: int, exact: Callable[[Tensor], Tensor],
             bounds, max_modes, rank=8) -> ManufacturedCase:
    # The source is evaluated with exact automatic derivatives.  create_graph
    # remains enabled so the same function is usable in derivative unit tests.
    def forcing(x: Tensor) -> Tensor:
        query = x.detach().clone().requires_grad_(True)
        u = exact(query)
        gradient = torch.autograd.grad(u.sum(), query, create_graph=True)[0]
        second = []
        for axis in range(dimension):
            second.append(torch.autograd.grad(
                gradient[:, axis].sum(), query, create_graph=True
            )[0][:, axis])
        result = second[-1] - sum(second[:-1]) + u.square()
        return result.detach()

    return ManufacturedCase(
        name, "klein_gordon", tuple(bounds), rank, exact, forcing,
        tuple("hybrid" for _ in range(dimension)), tuple(max_modes),
        time_axis=dimension - 1,
    )


def k1() -> ManufacturedCase:
    def exact(x: Tensor) -> Tensor:
        a, b, t = x.unbind(dim=1)
        return (a + b) * torch.cos(2.0 * t) + a * b * torch.sin(2.0 * t)

    return _kg_case(
        "K1", 3, exact, ((-1.0, 1.0), (-1.0, 1.0), (0.0, 10.0)),
        (4, 4, 8), rank=8,
    )


def k2() -> ManufacturedCase:
    def exact(x: Tensor) -> Tensor:
        a, b, c, t = x.unbind(dim=1)
        return (a + b + c) * torch.cos(t) + a * b * c * torch.sin(t)

    return _kg_case(
        "K2", 4, exact,
        ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 10.0)),
        (4, 4, 4, 8), rank=8,
    )


def k3() -> ManufacturedCase:
    omega = 20.0

    def exact(x: Tensor) -> Tensor:
        a, b, t = x.unbind(dim=1)
        return (a + b) * torch.cos(omega * t) + a * b * torch.sin(omega * t)

    return _kg_case(
        "K3", 3, exact, ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
        (4, 4, 24), rank=8,
    )


def k4() -> ManufacturedCase:
    def exact(x: Tensor) -> Tensor:
        a, b, t = x.unbind(dim=1)
        return (
            torch.sin(16.0 * torch.pi * a) * torch.sin(torch.pi * b) * torch.cos(2.0 * t)
            + 0.25 * torch.sin(4.0 * torch.pi * a)
            * torch.sin(8.0 * torch.pi * b) * torch.sin(6.0 * t)
        )

    return _kg_case(
        "K4", 3, exact, ((0.0, 1.0), (0.0, 1.0), (0.0, 2.0)),
        (16, 16, 12), rank=8,
    )


CASES = {function.__name__.upper(): function for function in (
    h1, h2, h3, h4, k1, k2, k3, k4,
)}


def get_case(name: str) -> ManufacturedCase:
    try:
        return CASES[name.upper()]()
    except KeyError as error:
        raise ValueError(f"unknown manufactured case {name!r}") from error


def _replace_kg_exact(case: ManufacturedCase,
                      exact: Callable[[Tensor], Tensor]) -> ManufacturedCase:
    """Rebuild a Klein--Gordon manufactured source after changing its task.

    This is intentionally kept in the case/evaluation module.  The trainer
    receives only the resulting PDE forcing and prescribed initial/boundary
    values; target interior solution values never enter optimization.
    """
    rebuilt = _kg_case(
        case.name, case.dimension, exact, case.bounds, case.max_modes,
        rank=case.rank,
    )
    return replace(rebuilt, feature_kinds=case.feature_kinds)


def parameterize_case(case: ManufacturedCase, kind: str,
                      value: float) -> ManufacturedCase:
    """Construct locked non-amplitude H/K generalization tasks.

    ``frequency`` changes an operator-relevant oscillation scale,
    ``coefficient`` extrapolates the mixture of fixed separable components,
    and ``structure`` interprets ``value`` as a bit mask selecting components.
    These transformations prevent the main transfer experiment from reducing
    to scalar output interpolation.
    """
    kind = kind.lower()
    if kind == "identity":
        return case
    if kind == "frequency":
        if case.name == "H3":
            frequency = int(round(value))
            if frequency < 1 or frequency > case.max_modes[1]:
                raise ValueError("frequency must lie in the fixed dictionary")
            frequencies = (1, frequency)
            rebuilt = _helmholtz_product_frequencies(
                case.name, frequencies, case.bounds, rank=case.rank,
            )
            return replace(rebuilt, max_modes=case.max_modes)
        if case.name == "K3":
            omega = float(value)
            if abs(omega-round(omega)) > 1e-8 or not 1 <= omega <= case.max_modes[-1]:
                raise ValueError("K3 frequency must be an integer dictionary mode")

            def exact(x: Tensor) -> Tensor:
                a, b, t = x.unbind(dim=1)
                return ((a+b)*torch.cos(omega*t)
                        +a*b*torch.sin(omega*t))

            return _replace_kg_exact(case, exact)
        raise ValueError(f"frequency task is not defined for {case.name}")
    if kind == "coefficient":
        beta = float(value)
        if case.name == "H4":
            pairs = ((1, 16), (2, 8), (4, 4), (8, 2))
            coefficients = tuple(beta**index for index in range(len(pairs)))

            def exact(x: Tensor) -> Tensor:
                return sum(
                    coefficient*torch.sin(a*torch.pi*x[:, 0])
                    *torch.sin(b*torch.pi*x[:, 1])
                    for coefficient, (a, b) in zip(coefficients, pairs)
                )

            def forcing(x: Tensor) -> Tensor:
                return sum(
                    coefficient*(1.0-(a*torch.pi)**2-(b*torch.pi)**2)
                    *torch.sin(a*torch.pi*x[:, 0])
                    *torch.sin(b*torch.pi*x[:, 1])
                    for coefficient, (a, b) in zip(coefficients, pairs)
                )

            return replace(case, exact=exact, forcing=forcing)
        if case.name == "K4":
            def exact(x: Tensor) -> Tensor:
                a, b, t = x.unbind(dim=1)
                first = (torch.sin(16.0*torch.pi*a)
                         *torch.sin(torch.pi*b)*torch.cos(2.0*t))
                second = (torch.sin(4.0*torch.pi*a)
                          *torch.sin(8.0*torch.pi*b)*torch.sin(6.0*t))
                # Quadratic task dependence makes coefficient extrapolation
                # non-trivial: two source output fields cannot recover the
                # held-out target by ordinary linear output extrapolation.
                return first+beta**2*second

            return _replace_kg_exact(case, exact)
        raise ValueError(f"coefficient task is not defined for {case.name}")
    if kind == "structure":
        mask = int(round(value))
        if mask <= 0:
            raise ValueError("structure bit mask must select at least one term")
        if case.name == "H4":
            pairs = ((1, 16), (2, 8), (4, 4), (8, 2))
            coefficients = tuple(1.0/(index+1) for index in range(len(pairs)))

            def exact(x: Tensor) -> Tensor:
                return sum(
                    coefficient*torch.sin(a*torch.pi*x[:, 0])
                    *torch.sin(b*torch.pi*x[:, 1])
                    for index, (coefficient, (a, b)) in enumerate(
                        zip(coefficients, pairs)
                    ) if mask & (1 << index)
                )

            def forcing(x: Tensor) -> Tensor:
                return sum(
                    coefficient*(1.0-(a*torch.pi)**2-(b*torch.pi)**2)
                    *torch.sin(a*torch.pi*x[:, 0])
                    *torch.sin(b*torch.pi*x[:, 1])
                    for index, (coefficient, (a, b)) in enumerate(
                        zip(coefficients, pairs)
                    ) if mask & (1 << index)
                )

            return replace(case, exact=exact, forcing=forcing)
        if case.name == "K4":
            def exact(x: Tensor) -> Tensor:
                a, b, t = x.unbind(dim=1)
                first = (torch.sin(16.0*torch.pi*a)
                         *torch.sin(torch.pi*b)*torch.cos(2.0*t))
                second = (0.25*torch.sin(4.0*torch.pi*a)
                          *torch.sin(8.0*torch.pi*b)*torch.sin(6.0*t))
                return ((first if mask & 1 else torch.zeros_like(first))
                        +(second if mask & 2 else torch.zeros_like(second)))

            return _replace_kg_exact(case, exact)
        raise ValueError(f"structure task is not defined for {case.name}")
    raise ValueError(f"unknown task parameter kind {kind!r}")
