#!/usr/bin/env python3
"""Check source integrity and optionally all sixteen smoke-run records."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from run_case import CASES, ROOT, configuration


def validate(smoke_results=False):
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for entry in manifest:
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
    assert len(list((ROOT / "configs").glob("*.json"))) == 16
    for case in CASES:
        cfg = configuration(case)
        assert cfg["rank"] == 16 and cfg["steps"] == 3000
        assert cfg["expected_parameters"] > 0
    rows = []
    if smoke_results:
        for case in CASES:
            folder = ROOT / "smoke_runs" / case / "seed_42"
            completed = json.loads((folder / "completed.json").read_text())
            assert completed["smoke"] and not completed["formal_protocol"]
            assert completed["parameters"] == configuration(case)["expected_parameters"]
            summary = json.loads((folder / completed["summary"]).read_text())
            errors = [v for k, v in summary.items() if "relative_l2" in k and isinstance(v, (int, float))]
            assert errors and all(math.isfinite(v) for v in errors), case
            assert list(folder.rglob("best_physics.pt")), case
            histories = list(folder.rglob("history.jsonl")) + list(folder.rglob("history.json"))
            assert len(histories) == 1, case
            history = (json.loads(histories[0].read_text()) if histories[0].suffix == ".json" else
                       [json.loads(line) for line in histories[0].read_text().splitlines()])
            assert history[-1]["step"] == 2, case
            assert all(math.isfinite(r["physics_validation"]) for r in history), case
            rows.append(dict(case=case, parameters=completed["parameters"], updates=2, status="passed"))
    report = dict(source_integrity="passed", cases=16, smoke_runs=rows,
                  full_budget_retrained=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-results", action="store_true")
    args = parser.parse_args()
    validate(args.smoke_results)
