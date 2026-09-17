"""Fixed-profile cosine training for H2, A2, A3 and K2."""
import argparse
import copy
from dataclasses import replace
import json
import math
import time

import torch

import dimensionwise_12case.train_helmholtz_hx_audit as hx
import dimensionwise_12case.train_branch_select_two_level_route as hk
import train_static_branch_select_two_level_route as anisotropic


class Problem:
    def __init__(self, config, device, seed, smoke=False):
        self.name, self.device, self.seed = config["case"], device, seed
        torch.manual_seed(seed)
        self.smoke = smoke
        if self.name == "H2":
            self.case = replace(hx.get_case("H2"), rank=16)
            torch.manual_seed(seed + 991)
            self.validation = hx.sample(32 if smoke else 2048, device)
            self.clip = 100.
        elif self.name.startswith("A"):
            self.args = anisotropic.parser().parse_args(["--family", "anisotropic", "--case", self.name])
            vars(self.args).update(config["native"])
            self.args.seed = seed
            if smoke:
                self.args.batch = self.args.validation = self.args.evaluation = 32
                self.args.boundary_batch = 8
            self.native = anisotropic.StaticProblem(self.args, device)
            self.case = anisotropic.profile_case(self.native.case, self.args.profile_selection[6:])
            self.validation, self.clip = self.native.validation, self.args.grad_clip
        else:
            self.args = hk.parser().parse_args(["--case", self.name])
            vars(self.args).update(config["native"])
            self.args.seed = seed
            if smoke:
                self.args.collocation = self.args.physics_validation = self.args.evaluation = 32
                self.args.boundary = 16
            self.case = hk.fixed_profile_case(replace(hk.get_case(self.name), rank=16), "compact")
            self.validation = hk.sample(self.case.bounds, self.args.physics_validation, device)
            self.clip = self.args.grad_clip

    def model(self):
        torch.manual_seed(self.seed)
        self.initialization = None
        if self.name == "H2":
            return hx.BranchProfileCP(self.case, "sine_sine").to(self.device)
        if self.name.startswith("A"):
            model = anisotropic.build(self.case, self.seed, base_scale=self.args.base_scale).to(self.device)
            if self.args.physics_projected_warm_start:
                self.initialization = anisotropic.physics_projected_cp_initialization(
                    self.case, model, self.args.warm_start_points_per_axis, self.args.warm_start_ridge)
            return model
        return hk.build(self.case, self.seed).to(self.device)

    def batch(self, step):
        torch.manual_seed(self.seed + (10000 if self.name == "H2" else 900001) + step)
        if self.name == "H2":
            return hx.sample(32 if self.smoke else 2048, self.device)
        if self.name.startswith("A"):
            return self.native.sample(self.case)
        return hk.sample(self.case.bounds, self.args.collocation, self.device, requires_grad=False)

    def loss(self, model, points, validation=False):
        if self.name == "H2":
            return hx.equation_residual(self.case, model, points)[2]
        if self.name.startswith("A"):
            return self.native.loss(self.case, model, points, deterministic=validation)[0]
        return hk.objective(self.case, model, points, self.args, deterministic=validation)[0]

    def evaluate(self, model):
        if self.name == "H2":
            return {"relative_l2": hx.relative_l2(self.case, model, self.device, count=128 if self.smoke else 16384)}
        if self.name.startswith("A"):
            return self.native.metric(self.case, model)
        return hk.metrics(self.case, model, self.args.evaluation, self.device)


def run(config, output, device, seed, steps, smoke=False):
    problem = Problem(config, device, seed, smoke)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model = problem.model()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert parameters == config["expected_parameters"], (problem.name, parameters)
    lr = config["native"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=lr*.01)
    best, state, best_step = math.inf, None, None
    for step in range(steps + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = problem.loss(model, problem.batch(step))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), problem.clip)
            optimizer.step()
            scheduler.step()
        if step % (1 if smoke else 100) == 0 or step == steps:
            model.eval()
            score = float(problem.loss(model, problem.validation, True).detach())
            row = dict(step=step, physics_validation=score, lr=optimizer.param_groups[0]["lr"])
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(dict(case=problem.name, **row)), flush=True)
            if math.isfinite(score) and score < best:
                best, best_step = score, step
                state = copy.deepcopy(model.state_dict())
    if state is None:
        raise RuntimeError("No finite physics checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    torch.save(dict(model=state, best_step=best_step, best_physics_validation=best), output / "best_physics.pt")
    model.load_state_dict(state)
    model.eval()
    metrics = problem.evaluate(model)
    result = dict(case=problem.name, seed=seed, steps=steps, rank=16,
                  best_step=best_step, best_physics_validation=best, seconds=elapsed,
                  trainable_parameters=parameters, initialization=problem.initialization,
                  physics_selected_relative_l2=metrics["relative_l2"], evaluation=metrics)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
