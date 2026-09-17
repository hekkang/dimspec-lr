import ast
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_case
from dimensionwise_12case.model import FixedFeatureDictionary
from steady_anisotropic.cases import a4
from validate_release import validate


def test_source_integrity():
    assert validate()["source_integrity"] == "passed"


@pytest.mark.parametrize("case", run_case.CASES)
def test_case_configurations(case):
    cfg = run_case.configuration(case)
    assert cfg["case"] == case and cfg["rank"] == 16 and cfg["steps"] == 3000
    assert "/home/" not in json.dumps(cfg)
    native = cfg["native"]
    if "profile_selection" in native:
        assert native["profile_selection"].startswith("fixed_")
    if "capacity_routing" in native:
        assert native["capacity_routing"] == "none"
    if case[0] == "S":
        assert native["fixed_family"] in ("physics_correction", "modified_mlp")


def test_no_neutron_imports():
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(m.startswith(("neutron", "uniform_froe", "heterogeneous_froe")) for m in modules), path


@pytest.mark.parametrize("x_mode,y_kind,y_mode", [(64, "hybrid", 32), (44, "boundary_layer", 44)])
def test_a4_exact_factor_inclusion(x_mode, y_kind, y_mode):
    epsilon = .0025
    x = torch.tensor([0., .1, .5, .99, .9975, .9999, 1.], dtype=torch.float64)
    y = torch.linspace(0., 1., 19, dtype=torch.float64)
    bx = FixedFeatureDictionary("boundary_layer", x_mode, 0., 1., epsilon)
    by = FixedFeatureDictionary(y_kind, y_mode, 0., 1., epsilon)
    layer = bx(x)[:, -1] / (-math.expm1(-1. / epsilon))
    transverse = by(y)[:, 4 + 2*y_mode]
    torch.testing.assert_close(torch.outer(layer, transverse).reshape(-1),
                               a4(epsilon).exact(torch.cartesian_prod(x, y)), rtol=1e-12, atol=1e-14)


def test_cli_works_outside_repository(tmp_path):
    process = subprocess.run([sys.executable, str(ROOT / "run_case.py"), "--list"],
                             cwd=tmp_path, capture_output=True, text=True, check=True)
    assert len(process.stdout.splitlines()) == 16


def test_smoke_records_when_available():
    if not (ROOT / "smoke_runs/K4/seed_42/completed.json").exists():
        pytest.skip("Run the documented smoke suite to check saved outputs")
    assert len(validate(smoke_results=True)["smoke_runs"]) == 16
