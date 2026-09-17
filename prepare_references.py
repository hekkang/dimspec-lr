#!/usr/bin/env python3
"""Generate evaluation-only numerical references from the prescribed PDEs."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=("H1", "A1", "S1", "S2", "S3", "S4", "all"), default=["all"])
    parser.add_argument("--smoke", action="store_true", help="Separate coarse references for executable smoke tests")
    args = parser.parse_args()
    cases = ("H1", "A1", "S1", "S2", "S3", "S4") if "all" in args.cases else args.cases
    base = ROOT / "references" / ("smoke" if args.smoke else "formal")
    for case in cases:
        path = base / case / "reference.npz"
        if path.exists():
            print(f"Using existing {path}", flush=True)
            continue
        if case in ("H1", "A1"):
            command = [sys.executable, str(ROOT / "scripts/generate_antialignment_references.py"),
                       "--cases", case, "--h1-k0", "5", "--a1-epsilon", "0.01",
                       "--coarse-size", "17" if args.smoke else "257",
                       "--fine-size", "33" if args.smoke else "513", "--output-root", str(base)]
        else:
            command = [sys.executable, str(ROOT / "src/stationary_transport/reference.py"),
                       "--case", case, "--nx", "64" if args.smoke else "2048",
                       "--nmu", "16" if args.smoke else "128", "--output", str(path)]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
