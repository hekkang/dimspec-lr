import math
import torch
from torch import nn

class StepLockedHardConcrete(nn.Module):
    """Hard-concrete gate with one sample per training step."""

    def __init__(self, shape, initial_log_alpha: float=3.0, temperature: float=2.0 / 3.0, lower: float=-0.1, upper: float=1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(shape, initial_log_alpha))
        self.temperature = float(temperature)
        (self.lower, self.upper) = (float(lower), float(upper))
        self.register_buffer('fixed_mask', torch.ones(shape))
        self.routing = False
        self.consolidated = False
        self.route_strength = 0.0
        self._cached_sample = None

    def expected_l0(self) -> torch.Tensor:
        offset = self.temperature * math.log(-self.lower / self.upper)
        return torch.sigmoid(self.log_alpha - offset)

    def deterministic(self) -> torch.Tensor:
        stretched = torch.sigmoid(self.log_alpha) * (self.upper - self.lower) + self.lower
        return stretched.clamp(0.0, 1.0)

    def begin_step(self) -> None:
        self._cached_sample = None
        if not self.routing or self.consolidated:
            return
        uniform = torch.rand_like(self.log_alpha).clamp_(1e-06, 1.0 - 1e-06)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        relaxed = torch.sigmoid((logistic + self.log_alpha) / self.temperature)
        stretched = relaxed * (self.upper - self.lower) + self.lower
        self._cached_sample = stretched.clamp(0.0, 1.0)

    def forward(self) -> torch.Tensor:
        if self.consolidated:
            return self.fixed_mask
        if not self.routing:
            return torch.ones_like(self.log_alpha)
        if self.training:
            if self._cached_sample is None:
                raise RuntimeError('begin_step() must be called before routed training')
            routed = self._cached_sample
        else:
            routed = self.deterministic()
        return 1.0 + self.route_strength * (routed - 1.0)

    @torch.no_grad()
    def consolidate(self, threshold: float, minimum_active: int, eligible: torch.Tensor | None=None) -> torch.Tensor:
        probability = self.expected_l0()
        mask = probability >= threshold
        if eligible is not None:
            mask &= eligible.to(mask.device, dtype=torch.bool)
        flat_probability = probability.reshape(-1)
        flat_mask = mask.reshape(-1)
        eligible_flat = torch.ones_like(flat_mask) if eligible is None else eligible.to(mask.device, dtype=torch.bool).reshape(-1)
        available = int(eligible_flat.sum())
        required = min(max(0, int(minimum_active)), available)
        if int(flat_mask.sum()) < required:
            scores = flat_probability.masked_fill(~eligible_flat, -torch.inf)
            selected = torch.topk(scores, required).indices
            flat_mask[selected] = True
        self.fixed_mask.copy_(flat_mask.reshape_as(mask).to(self.fixed_mask))
        self.consolidated = True
        self.routing = False
        self._cached_sample = None
        self.log_alpha.requires_grad_(False)
        return self.fixed_mask.clone()
