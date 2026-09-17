#!/usr/bin/env python3
"""High-order S_N source-iteration references for S1--S4.

The solver uses exponential step characteristics in every spatial cell.  It
is entirely independent of the neural model and its files are evaluation-only
artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stationary_transport.cases import get_case  # noqa: E402


def scattering_kernel(mu: np.ndarray, coefficients) -> np.ndarray:
    vandermonde = legvander(mu, len(coefficients)-1)
    return (vandermonde*np.asarray(coefficients)[None, :])@vandermonde.T


def solve(case_name: str, parameter: float | None, nx: int, nmu: int,
          tolerance: float, maximum_iterations: int, relaxation: float):
    case = get_case(case_name, parameter)
    mu, weights = leggauss(nmu)
    x = (np.arange(nx)+0.5)/nx
    dx = 1.0/nx
    scattering, absorption = case.coefficients_numpy(x)
    total = scattering+absorption
    kernel = scattering_kernel(mu, case.legendre_coefficients)
    psi = np.where(mu[None, :] > 0.0, case.left_inflow,
                   case.right_inflow).repeat(nx, axis=0).astype(np.float64)
    positive = mu > 0.0
    negative = ~positive
    converged = False
    final_change = np.inf
    for iteration in range(1, maximum_iterations+1):
        source = 0.5*scattering[:, None]*((psi*weights[None, :])@kernel.T)
        updated = np.empty_like(psi)
        incoming = np.full(positive.sum(), case.left_inflow)
        for cell in range(nx):
            half = np.exp(-total[cell]*dx/(2.0*mu[positive]))
            full = half*half
            equilibrium = source[cell, positive]/max(total[cell], 1e-14)
            updated[cell, positive] = incoming*half+equilibrium*(1.0-half)
            incoming = incoming*full+equilibrium*(1.0-full)
        incoming = np.full(negative.sum(), case.right_inflow)
        for cell in range(nx-1, -1, -1):
            speed = -mu[negative]
            half = np.exp(-total[cell]*dx/(2.0*speed))
            full = half*half
            equilibrium = source[cell, negative]/max(total[cell], 1e-14)
            updated[cell, negative] = incoming*half+equilibrium*(1.0-half)
            incoming = incoming*full+equilibrium*(1.0-full)
        updated = relaxation*updated+(1.0-relaxation)*psi
        final_change = (np.linalg.norm(updated-psi)
                        /max(np.linalg.norm(updated), 1e-14))
        psi = updated
        if final_change < tolerance:
            converged = True
            break
    scalar_flux = psi@weights
    metadata = {
        "case": case.name, "parameter": case.parameter, "nx": nx, "nmu": nmu,
        "iterations": iteration, "relative_iteration_change": final_change,
        "tolerance": tolerance, "converged": converged,
        "method": "SN_exponential_step_characteristics_source_iteration",
        "reference_role": "evaluation_only", "training_data": False,
    }
    return {"x": x, "mu": mu, "weights": weights, "psi": psi,
            "scalar_flux": scalar_flux, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("S1", "S2", "S3", "S4"), required=True)
    parser.add_argument("--parameter", type=float, default=None)
    parser.add_argument("--nx", type=int, default=2048)
    parser.add_argument("--nmu", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-11)
    parser.add_argument("--maximum-iterations", type=int, default=20000)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = solve(args.case, args.parameter, args.nx, args.nmu, args.tolerance,
                   args.maximum_iterations, args.relaxation)
    if not result["metadata"]["converged"]:
        raise RuntimeError(f"reference did not converge: {result['metadata']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=result["x"], mu=result["mu"],
                        weights=result["weights"], psi=result["psi"],
                        scalar_flux=result["scalar_flux"])
    args.output.with_suffix(".json").write_text(json.dumps(
        result["metadata"], indent=2,
    ))
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
