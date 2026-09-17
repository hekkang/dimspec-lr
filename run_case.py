#!/usr/bin/env python3
"""Run the final fixed-profile configurations for the sixteen H/A/S/K cases."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/dimensionwise_12case"), str(ROOT / "scripts")]
CASES = tuple(f"{prefix}{index}" for prefix in "HASK" for index in range(1, 5))


def configuration(case):
    return json.loads((ROOT / "configs" / f"{case}.json").read_text())


def run_script(path, arguments):
    previous = sys.argv
    try:
        sys.argv = [str(path), *map(str, arguments)]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = previous


def reference_path(case, smoke):
    return ROOT / "references" / ("smoke" if smoke else "formal") / case / "reference.npz"


def train(args):
    import torch
    torch.set_num_threads(args.threads)
    config = configuration(args.case)
    steps = 2 if args.smoke else args.steps
    root = (args.output or ROOT / ("smoke_runs" if args.smoke else "runs")).resolve()
    output = root / args.case / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    ref = reference_path(args.case, args.smoke)
    if args.case in ("H1", "A1", "S1", "S2", "S3", "S4") and not ref.exists():
        raise FileNotFoundError(f"Run: python prepare_references.py --cases {args.case}" +
                                (" --smoke" if args.smoke else ""))
    output.mkdir(parents=True)
    invocation = dict(config=config, seed=args.seed, device=str(device), threads=args.threads,
                      torch_version=torch.__version__, python_version=sys.version,
                      cuda_version=torch.version.cuda,
                      steps=steps, smoke=args.smoke, formal_protocol=not args.smoke and steps == 3000)
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    if config["backend"] == "cosine":
        import train_selected
        train_selected.run(config, output, device, args.seed, steps, args.smoke)
    elif config["backend"] == "curved":
        native = config["native"]
        command = ["--case", args.case, "--method", "DimSpec-LR-Fixed", "--cp-rank", "16",
                   "--steps", steps, "--lr", native["lr"], "--lr-schedule", native["lr_schedule"],
                   "--helmholtz-loss", "combined", "--seed", args.seed, "--device", device,
                   "--batch", 32 if args.smoke else 2048, "--validation", 32 if args.smoke else 4096,
                   "--evaluation", 128 if args.smoke else 16384,
                   "--eval-every", 1 if args.smoke else 100, "--reference-file", ref,
                   "--output", output, "--h1-k0" if args.case == "H1" else "--a1-epsilon",
                   5.0 if args.case == "H1" else .01]
        run_script(ROOT / "scripts/train_antialignment_replacement.py", command)
    elif config["backend"] == "transport":
        import stationary_transport.train_branch_candidates as native
        options = native.parser().parse_args(["--case", args.case])
        vars(options).update(config["native"])
        options.families = [options.fixed_family]
        options.probe_steps = 0
        options.refine_steps = steps
        options.reference = ref
        options.seed, options.device, options.output = args.seed, str(device), output
        options.variant = "fixed_profile"
        if args.smoke:
            options.batch_x = options.validation_x = 8
            options.eval_every = 1
        native.run(options)
    elif config["backend"] == "layer":
        import steady_anisotropic.train as native
        options = native.parser().parse_args(["--case", args.case, "--output", str(output)])
        vars(options).update(config["native"])
        options.output, options.device, options.seed, options.steps = output, str(device), args.seed, steps
        options.resume = None
        if args.smoke:
            options.batch = options.validation = 32
            options.evaluation = 128
            options.boundary_batch, options.eval_every = 8, 1
        native.run(options)
    else:
        import dimensionwise_12case.train_branch_select_two_level_route as native
        options = native.parser().parse_args(["--case", args.case])
        vars(options).update(config["native"])
        options.output, options.device, options.seed, options.steps = output, str(device), args.seed, steps
        options.capacity_routing, options.variant = "none", "fixed_profile"
        if args.smoke:
            options.collocation = options.physics_validation = 32
            options.evaluation, options.boundary, options.eval_every = 128, 16, 1
        native.run(options)
    summaries = list(output.rglob("summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one completed summary, found {summaries}")
    result = json.loads(summaries[0].read_text())
    parameters = next((result[k] for k in ("effective_active_parameters", "effective_parameters", "trainable_parameters") if k in result), None)
    if parameters != config["expected_parameters"]:
        raise RuntimeError(f"Parameter count differs from manuscript: {parameters}")
    (output / "completed.json").write_text(json.dumps(dict(
        summary=str(summaries[0].relative_to(output)), parameters=parameters,
        smoke=args.smoke, formal_protocol=invocation["formal_protocol"]), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--smoke", action="store_true", help="Two updates, small point sets; NOT an accuracy reproduction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            cfg = configuration(case)
            print(f"{case}: R=16, parameters={cfg['expected_parameters']}, backend={cfg['backend']}")
        return
    if not args.case:
        parser.error("--case or --list is required")
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    if args.case == "all":
        for case in CASES:
            command = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                       "--device", args.device, "--threads", str(args.threads),
                       "--seed", str(args.seed), "--steps", str(args.steps)]
            if args.smoke:
                command.append("--smoke")
            if args.output:
                command.extend(["--output", str(args.output.resolve())])
            subprocess.run(command, check=True, cwd=ROOT)
    else:
        train(args)


if __name__ == "__main__":
    main()
