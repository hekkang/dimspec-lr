#!/usr/bin/env python3
"""Generate independent finite-difference references for new H1/A1/A2.

The script solves each prescribed boundary-value problem on nested uniform
grids, records a grid-convergence discrepancy, and computes the best rank-R
SVD error of the fine-grid field.  It never accesses a neural model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/antialignment_replacements_v1/references"


def _assemble(size: int, bounds, stencil, boundary):
    x = np.linspace(bounds[0][0], bounds[0][1], size)
    y = np.linspace(bounds[1][0], bounds[1][1], size)
    hx, hy = x[1] - x[0], y[1] - y[0]
    if not np.isclose(hx, hy):
        raise ValueError("the locked references require a square mesh")
    h = float(hx)
    m = size - 2
    rows, cols, values = [], [], []
    rhs = np.empty(m * m, dtype=np.float64)
    for iy in range(1, size - 1):
        for ix in range(1, size - 1):
            row = (iy - 1) * m + ix - 1
            weights, source = stencil(x[ix], y[iy], h)
            rhs[row] = source
            for dx, dy, weight in weights:
                jx, jy = ix + dx, iy + dy
                if 1 <= jx < size - 1 and 1 <= jy < size - 1:
                    rows.append(row)
                    cols.append((jy - 1) * m + jx - 1)
                    values.append(weight)
                else:
                    rhs[row] -= weight * boundary(x[jx], y[jy])
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(m * m, m * m))
    solution = spsolve(matrix.tocsc(), rhs)
    relative_residual = np.linalg.norm(matrix @ solution - rhs) / max(
        np.linalg.norm(rhs), np.finfo(float).eps
    )
    field = np.empty((size, size), dtype=np.float64)
    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            field[iy, ix] = boundary(xx, yy)
    field[1:-1, 1:-1] = solution.reshape(m, m)
    return x, y, field, float(relative_residual)


def solve_h1(size: int, k0: float = 15.0):
    bounds = ((-1.0, 1.0), (-1.0, 1.0))

    def stencil(x, y, h):
        n = (1.0 + 0.35 * np.exp(-((x - 0.25)**2 + (y + 0.15)**2) / 0.12)
             + 0.15 * np.sin(2.0 * np.pi * x * y))
        q = float(k0)**2 * n**2
        source = np.exp(-120.0 * (np.hypot(x + 0.45, y - 0.15) - 0.18)**2)
        h2 = h * h
        return [(-1, 0, 1.0 / h2), (1, 0, 1.0 / h2),
                (0, -1, 1.0 / h2), (0, 1, 1.0 / h2),
                (0, 0, q - 4.0 / h2)], source

    return _assemble(size, bounds, stencil, lambda x, y: 0.0)


def _a1_trace(y: float) -> float:
    g = np.exp(-150.0 * (y - 0.25)**2)
    g0 = np.exp(-150.0 * 0.25**2)
    g1 = np.exp(-150.0 * 0.75**2)
    return float(g - (1.0 - y) * g0 - y * g1)


def solve_a1(size: int, epsilon: float = 2.0e-3):
    bounds = ((0.0, 1.0), (0.0, 1.0))

    def boundary(x, y):
        return _a1_trace(y) if np.isclose(x, 0.0) else 0.0

    def stencil(x, y, h):
        bx = np.sin(np.pi * x) * np.cos(np.pi * y)
        by = -np.cos(np.pi * x) * np.sin(np.pi * y)
        h2 = h * h
        return [
            (-1, 0, -epsilon / h2 - bx / (2.0 * h)),
            (1, 0, -epsilon / h2 + bx / (2.0 * h)),
            (0, -1, -epsilon / h2 - by / (2.0 * h)),
            (0, 1, -epsilon / h2 + by / (2.0 * h)),
            (0, 0, 4.0 * epsilon / h2 + 0.1),
        ], 0.0

    return _assemble(size, bounds, stencil, boundary)


def solve_a2(size: int):
    bounds = ((0.0, 1.0), (0.0, 1.0))

    def stencil(x, y, h):
        theta = 0.5 * np.pi * x + np.pi / 6.0 * np.sin(2.0 * np.pi * y)
        tx = 0.5 * np.pi
        ty = np.pi**2 / 3.0 * np.cos(2.0 * np.pi * y)
        c, s = np.cos(theta), np.sin(theta)
        dxx = c*c + 0.01*s*s
        dyy = s*s + 0.01*c*c
        dxy = 0.99*c*s
        dxx_t = -1.98*c*s
        dyy_t = 1.98*c*s
        dxy_t = 0.99*(c*c - s*s)
        qx = dxx_t*tx + dxy_t*ty
        qy = dxy_t*tx + dyy_t*ty
        h2 = h*h
        ring = (x - 0.5)**2 + (y - 0.5)**2 - 0.22**2
        source = np.exp(-100.0 * ring**2)
        return [
            (-1, 0, -dxx/h2 + qx/(2.0*h)),
            (1, 0, -dxx/h2 - qx/(2.0*h)),
            (0, -1, -dyy/h2 + qy/(2.0*h)),
            (0, 1, -dyy/h2 - qy/(2.0*h)),
            (1, 1, -dxy/(2.0*h2)),
            (1, -1, dxy/(2.0*h2)),
            (-1, 1, dxy/(2.0*h2)),
            (-1, -1, -dxy/(2.0*h2)),
            (0, 0, 2.0*(dxx+dyy)/h2 + 0.1),
        ], source

    return _assemble(size, bounds, stencil, lambda x, y: 0.0)


def svd_diagnostic(field: np.ndarray, ranks=(1, 2, 4, 8, 16, 32)):
    singular_values = np.linalg.svd(field, compute_uv=False)
    energy = np.square(singular_values)
    total = energy.sum()
    errors = {}
    for rank in ranks:
        errors[str(rank)] = float(np.sqrt(energy[rank:].sum() / total))
    return singular_values, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=("H1", "A1", "A2"))
    parser.add_argument("--coarse-size", type=int, default=129)
    parser.add_argument("--fine-size", type=int, default=257)
    parser.add_argument("--h1-k0", type=float, default=15.0)
    parser.add_argument("--a1-epsilon", type=float, default=2.0e-3)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    solvers = {"H1": solve_h1, "A1": solve_a1, "A2": solve_a2}
    for raw in args.cases:
        case = raw.upper()
        output = args.output_root / case
        canonical = output / "reference.npz"
        if canonical.exists() and not args.overwrite:
            print(json.dumps({"case": case, "event": "reuse", "path": str(canonical)}))
            continue
        output.mkdir(parents=True, exist_ok=True)
        solver_arguments = ((args.coarse_size, args.h1_k0)
                            if case == "H1" else
                            (args.coarse_size, args.a1_epsilon)
                            if case == "A1" else (args.coarse_size,))
        xc, yc, coarse, coarse_residual = solvers[case](*solver_arguments)
        solver_arguments = ((args.fine_size, args.h1_k0)
                            if case == "H1" else
                            (args.fine_size, args.a1_epsilon)
                            if case == "A1" else (args.fine_size,))
        xf, yf, fine, fine_residual = solvers[case](*solver_arguments)
        stride = (args.fine_size - 1) // (args.coarse_size - 1)
        if stride * (args.coarse_size - 1) != args.fine_size - 1:
            raise ValueError("fine/coarse grids must be nested")
        convergence = float(np.linalg.norm(fine[::stride, ::stride] - coarse)
                            / np.linalg.norm(fine[::stride, ::stride]))
        singular_values, rank_errors = svd_diagnostic(fine)
        metadata = {
            "case": case, "reference_method": "second_order_finite_difference",
            "coarse_size": args.coarse_size, "fine_size": args.fine_size,
            "k0": args.h1_k0 if case == "H1" else None,
            "epsilon": args.a1_epsilon if case == "A1" else None,
            "coarse_linear_residual": coarse_residual,
            "fine_linear_residual": fine_residual,
            "nested_grid_relative_difference": convergence,
            "best_separable_rank_relative_errors": rank_errors,
            "training_data_used": False,
        }
        np.savez_compressed(canonical, x=xf, y=yf, u=fine,
                            singular_values=singular_values,
                            metadata=json.dumps(metadata))
        (output / "diagnostics.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
