"""Matched neural baselines for the manufactured H/K benchmark.

These implementations deliberately share coordinate normalization and the
Helmholtz hard-boundary envelope.  In addition to the small historical
baselines, this module contains paper-faithful architectural reproductions of
SPINN-mod (modified-MLP body networks) and PirateNet (Fourier embedding plus
zero-initialised adaptive residual blocks).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def parameter_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    fixed += sum(b.numel() for b in model.buffers())
    return trainable, fixed


class CaseModule(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        self.register_buffer("lower", torch.tensor([v[0] for v in case.bounds]))
        self.register_buffer("upper", torch.tensor([v[1] for v in case.bounds]))

    def normalize(self, x):
        return 2.0 * (x-self.lower)/(self.upper-self.lower)-1.0

    def apply_helmholtz_boundary(self, x, value):
        family = getattr(self.case, "family", None)
        case_name = getattr(self.case, "name", "")
        # H and A benchmarks both prescribe homogeneous Dirichlet data on a
        # box.  Keep the historical method name for checkpoint compatibility,
        # while applying the same zero-trace envelope to anisotropic CDR.
        if family != "helmholtz" and not case_name.startswith("A"):
            return value
        normalized = (x-self.lower)/(self.upper-self.lower)
        envelope = (4.0*normalized*(1.0-normalized)).prod(dim=1)
        return envelope*value


def mlp(input_size, output_size, width, depth):
    layers = [nn.Linear(input_size, width), nn.Tanh()]
    for _ in range(depth-1):
        layers.extend((nn.Linear(width, width), nn.Tanh()))
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


class VanillaPINN(CaseModule):
    def __init__(self, case, width=128, depth=4):
        super().__init__(case)
        self.network = mlp(case.dimension, 1, width, depth)

    def forward(self, x):
        value = self.network(self.normalize(x)).squeeze(1)
        return self.apply_helmholtz_boundary(x, value)


class ModifiedMLPPINN(CaseModule):
    """Gated modified-MLP baseline used in the SPINN/PINN literature.

    Two input encoders ``U`` and ``V`` are reused at every hidden layer.  A
    learned gate interpolates between them, which shortens gradient paths for
    the coordinate derivatives entering a PINN residual.  This is the
    architecture introduced by Wang, Teng & Perdikaris (2021), not a
    dimension-wise or low-rank model.
    """

    def __init__(self, case, width=128, depth=4):
        super().__init__(case)
        self.encoder_u = nn.Linear(case.dimension, width)
        self.encoder_v = nn.Linear(case.dimension, width)
        self.input_layer = nn.Linear(case.dimension, width)
        self.gates = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.output_layer = nn.Linear(width, 1)

    def forward(self, x):
        z = self.normalize(x)
        u = torch.tanh(self.encoder_u(z))
        v = torch.tanh(self.encoder_v(z))
        hidden = torch.tanh(self.input_layer(z))
        for layer in self.gates:
            gate = torch.tanh(layer(hidden))
            hidden = (1.0-gate)*u + gate*v
        value = self.output_layer(hidden).squeeze(1)
        return self.apply_helmholtz_boundary(x, value)


class FourierPINN(CaseModule):
    def __init__(self, case, width=128, depth=4):
        super().__init__(case)
        maximum = max(case.max_modes)
        self.register_buffer("modes", torch.arange(1, maximum+1).float())
        encoded = case.dimension*(1+2*maximum)
        self.network = mlp(encoded, 1, width, depth)

    def forward(self, x):
        z = self.normalize(x)
        phase = math.pi*z.unsqueeze(2)*self.modes.reshape(1, 1, -1)
        features = torch.cat((z.unsqueeze(2), torch.sin(phase), torch.cos(phase)), dim=2)
        value = self.network(features.flatten(1)).squeeze(1)
        return self.apply_helmholtz_boundary(x, value)


class AxisBranch(nn.Module):
    def __init__(self, rank, width, depth):
        super().__init__()
        self.network = mlp(1, rank, width, depth)

    def forward(self, x):
        return self.network(x.reshape(-1, 1))


class CPAxisBranch(nn.Module):
    """One coordinate network in the published CP-PINN architecture.

    Vemuri et al. use one tanh MLP per input coordinate, set every hidden
    width to the CP rank, and initialize all affine weights with Xavier
    normal initialization.  With ``depth=3`` this branch has three hidden
    affine layers and one rank-valued output layer, matching the four-layer
    configuration in the reference implementation.
    """

    def __init__(self, rank=32, depth=3):
        super().__init__()
        layers: list[nn.Module] = []
        input_size = 1
        for _ in range(depth):
            linear = nn.Linear(input_size, rank)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            input_size = rank
        output = nn.Linear(rank, rank)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.reshape(-1, 1))


class CPPINN(CaseModule):
    """Paper-faithful canonical-polyadic PINN under the common PDE protocol.

    The represented field is ``sum_r prod_d f_{d,r}(x_d)``.  Unlike the
    historical SPINN control, the published CP-PINN does not divide this sum
    by ``sqrt(rank)`` and uses rank-width coordinate MLPs.  Boundary handling
    remains identical to every other baseline in this repository so that the
    comparison isolates the representation rather than the constraint code.
    """

    def __init__(self, case, rank=32, depth=3):
        super().__init__(case)
        self.rank = int(rank)
        # The published CP-PINN contracts factors without SPINN's sqrt(R)
        # normalization.  Tensor-grid residual code reads this value so that
        # its optimized contraction is identical to ``forward``.
        self.cp_normalizer = 1.0
        self.branches = nn.ModuleList([
            CPAxisBranch(self.rank, depth) for _ in range(case.dimension)
        ])

    def forward(self, x):
        z = self.normalize(x)
        product = torch.ones(
            x.shape[0], self.rank, device=x.device, dtype=x.dtype,
        )
        for axis, branch in enumerate(self.branches):
            product = product * branch(z[:, axis])
        value = product.sum(dim=1)
        return self.apply_helmholtz_boundary(x, value)

    def separable_factors(self, x, mu):
        coordinates = (x, mu)
        factors = []
        for axis, (coordinate, branch) in enumerate(
            zip(coordinates, self.branches)
        ):
            normalized = 2.0 * (coordinate-self.lower[axis]) / (
                self.upper[axis]-self.lower[axis]
            ) - 1.0
            factors.append(branch(normalized))
        return tuple(factors)


class ModifiedAxisBranch(nn.Module):
    """One SPINN modified-MLP body network.

    This follows the official SPINN implementation: two persistent encodings
    U/V are mixed by a learned gate at every hidden layer, and a final linear
    layer emits all CP factors for this coordinate.
    """

    def __init__(self, rank, width=64, depth=4):
        super().__init__()
        self.encoder_u = nn.Linear(1, width)
        self.encoder_v = nn.Linear(1, width)
        self.input_layer = nn.Linear(1, width)
        self.gates = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.output_layer = nn.Linear(width, rank)

    def forward(self, x):
        x = x.reshape(-1, 1)
        u = torch.tanh(self.encoder_u(x))
        v = torch.tanh(self.encoder_v(x))
        hidden = torch.tanh(self.input_layer(x))
        for layer in self.gates:
            gate = torch.tanh(layer(hidden))
            hidden = (1.0-gate)*u + gate*v
        return self.output_layer(hidden)


class SeparablePINN(CaseModule):
    """SPINN-style per-axis MLP representation with a CP product output."""

    def __init__(self, case, rank=32, width=64, depth=3):
        super().__init__(case)
        self.rank = rank
        self.branches = nn.ModuleList([
            AxisBranch(rank, width, depth) for _ in range(case.dimension)
        ])

    def forward(self, x):
        z = self.normalize(x)
        product = torch.ones(x.shape[0], self.rank, device=x.device, dtype=x.dtype)
        for axis, branch in enumerate(self.branches):
            product = product*branch(z[:, axis])
        value = product.sum(dim=1)/math.sqrt(self.rank)
        return self.apply_helmholtz_boundary(x, value)

    def separable_factors(self, x, mu):
        coordinates = (x, mu)
        factors = []
        for axis, (coordinate, branch) in enumerate(zip(coordinates, self.branches)):
            normalized = 2.0 * (coordinate-self.lower[axis]) / (
                self.upper[axis]-self.lower[axis]
            ) - 1.0
            factors.append(branch(normalized))
        return tuple(factors)


class SPINNModified(CaseModule):
    """SPINN with the modified-MLP body used in the NeurIPS implementation."""

    def __init__(self, case, rank=32, width=64, depth=4):
        super().__init__(case)
        self.rank = rank
        self.branches = nn.ModuleList([
            ModifiedAxisBranch(rank, width, depth)
            for _ in range(case.dimension)
        ])

    def forward(self, x):
        z = self.normalize(x)
        product = torch.ones(x.shape[0], self.rank, device=x.device, dtype=x.dtype)
        for axis, branch in enumerate(self.branches):
            product = product*branch(z[:, axis])
        value = product.sum(dim=1)/math.sqrt(self.rank)
        return self.apply_helmholtz_boundary(x, value)

    def separable_factors(self, x, mu):
        """Return axis factors for tensor-grid transport evaluation."""
        coordinates = (x, mu)
        factors = []
        for axis, (coordinate, branch) in enumerate(zip(coordinates, self.branches)):
            normalized = 2.0 * (coordinate-self.lower[axis]) / (
                self.upper[axis]-self.lower[axis]
            ) - 1.0
            factors.append(branch(normalized))
        return tuple(factors)


class PirateBlock(nn.Module):
    """Three-layer physics-informed modified residual block.

    ``alpha=0`` makes every block an exact identity at initialisation.  During
    optimisation alpha learns how much depth to admit, which is PirateNet's
    defining adaptive-depth mechanism.
    """

    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList(FactorizedLinear(width, width) for _ in range(3))
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, hidden, u, v):
        identity = hidden
        value = hidden
        for index, layer in enumerate(self.layers):
            value = torch.tanh(layer(value))
            if index < 2:
                value = value*u+(1.0-value)*v
        return self.alpha*value+(1.0-self.alpha)*identity


class FactorizedLinear(nn.Module):
    """Dense layer with the PirateNet/JAX-PI weight-factorization parameterization."""

    def __init__(self, input_size, output_size, mean=1.0, std=0.1):
        super().__init__()
        weight = torch.empty(output_size, input_size)
        nn.init.xavier_normal_(weight)
        log_scale = mean+std*torch.randn(output_size)
        scale = torch.exp(log_scale)
        self.log_scale = nn.Parameter(log_scale)
        self.direction = nn.Parameter(weight/scale[:, None])
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        weight = torch.exp(self.log_scale)[:, None]*self.direction
        return torch.nn.functional.linear(x, weight, self.bias)


class PirateNetPINN(CaseModule):
    """PyTorch reproduction of PirateNet for scalar PDE solutions.

    The official architecture requires an embedding whose width equals the
    residual state width.  We use a trainable random Fourier embedding, two
    persistent U/V encoders, and three adaptive blocks (effective depth 9),
    matching the default architecture used by the official implementation.
    """

    def __init__(self, case, width=256, blocks=3, embedding_scale=2.0):
        super().__init__(case)
        if width % 2:
            raise ValueError("PirateNet width must be even")
        fourier = torch.randn(case.dimension, width//2)*embedding_scale
        self.fourier = nn.Parameter(fourier)
        self.encoder_u = FactorizedLinear(width, width)
        self.encoder_v = FactorizedLinear(width, width)
        self.blocks = nn.ModuleList(PirateBlock(width) for _ in range(blocks))
        self.output_layer = FactorizedLinear(width, 1)

    def forward(self, x):
        z = self.normalize(x)
        phase = z@self.fourier
        hidden = torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)
        u = torch.tanh(self.encoder_u(hidden))
        v = torch.tanh(self.encoder_v(hidden))
        for block in self.blocks:
            hidden = block(hidden, u, v)
        value = self.output_layer(hidden).squeeze(1)
        return self.apply_helmholtz_boundary(x, value)


class RandomFeaturePINN(CaseModule):
    """Partition-of-unity random-feature collocation model (RFM).

    Centres, local scales, hidden directions and biases are sampled once and
    frozen.  Only the linear expansion coefficients are fitted.  The smooth
    normalized Gaussian windows are a mesh-free partition of unity and make
    this closer to the PDE RFM literature than a generic frozen dense layer.
    """

    def __init__(self, case, feature_count=2048, localization=8.0):
        super().__init__(case)
        dimension = case.dimension
        per_axis = {1: 16, 2: 4, 3: 3, 4: 2}.get(dimension, 2)
        axis_centres = torch.linspace(-1.0+1.0/per_axis,
                                     1.0-1.0/per_axis, per_axis)
        centres = torch.cartesian_prod(*([axis_centres]*dimension)).reshape(-1, dimension)
        partition_count = centres.shape[0]
        features_per_partition = math.ceil(feature_count/partition_count)
        feature_count = partition_count*features_per_partition
        partition_index = torch.arange(partition_count).repeat_interleave(
            features_per_partition
        )
        self.register_buffer("centres", centres)
        self.register_buffer("partition_index", partition_index)
        self.register_buffer("directions", 2.0*torch.rand(feature_count, dimension)-1.0)
        self.register_buffer("biases", 2.0*math.pi*torch.rand(feature_count))
        # A small deterministic scale mixture prevents one arbitrary RFM
        # bandwidth from dominating the comparison on oscillatory cases.
        maximum_frequency = max(4.0, math.pi*max(case.max_modes))
        local_scales = torch.logspace(
            math.log10(0.5), math.log10(maximum_frequency),
            features_per_partition,
        )
        scales = local_scales.repeat(partition_count)
        self.register_buffer("scales", scales)
        self.localization = float(localization)
        self.cell_width = 2.0/per_axis
        self.coefficients = nn.Parameter(torch.zeros(feature_count))
        self.offset = nn.Parameter(torch.zeros(()))

    def basis(self, x):
        z = self.normalize(x)
        partition_delta = z[:, None, :]-self.centres[None, :, :]
        log_windows = -self.localization*partition_delta.square().sum(dim=2)
        windows = torch.softmax(log_windows, dim=1)
        selected_centres = self.centres[self.partition_index]
        delta = (z[:, None, :]-selected_centres[None, :, :])/self.cell_width
        phase = (delta*self.directions[None, :, :]).sum(dim=2)
        phase = phase*self.scales[None, :]+self.biases[None, :]
        # Each partition owns several random features.  The PoU is normalized
        # across partitions (not across individual features), as in RFM.
        return windows[:, self.partition_index]*torch.tanh(phase)

    def forward(self, x):
        value = self.basis(x)@self.coefficients+self.offset
        return self.apply_helmholtz_boundary(x, value)

    def linear_features(self, x):
        """Return columns whose coefficients form the represented solution."""
        columns = torch.cat((self.basis(x), torch.ones_like(x[:, :1])), dim=1)
        family = getattr(self.case, "family", None)
        if family == "helmholtz" or getattr(self.case, "name", "").startswith("A"):
            normalized = (x-self.lower)/(self.upper-self.lower)
            envelope = (4.0*normalized*(1.0-normalized)).prod(dim=1, keepdim=True)
            columns = envelope*columns
        return columns

    def set_linear_solution(self, solution):
        with torch.no_grad():
            self.coefficients.copy_(solution[:-1])
            self.offset.copy_(solution[-1])


def fit_rfm_linear_pde(model: RandomFeaturePINN, case, points,
                       ridge=1e-6):
    """Fit RFM coefficients by physics collocation for a linear second-order PDE.

    No solution samples are used: the right-hand side is the prescribed PDE
    forcing and every matrix column is the differential operator applied to
    one frozen random feature.
    """
    from torch.func import jacfwd, vmap

    def single_feature_vector(point):
        return model.linear_features(point[None, :])[0]

    values = vmap(single_feature_vector)(points)
    jacobian = vmap(jacfwd(single_feature_vector))(points)
    hessian = vmap(jacfwd(jacfwd(single_feature_vector)))(points)
    if getattr(case, "family", None) == "helmholtz":
        coefficient_fn = getattr(case, "coefficient", None)
        if coefficient_fn is None:
            coefficient = torch.ones(
                points.shape[0], device=points.device, dtype=points.dtype,
            )
        else:
            coefficient = coefficient_fn(points)
        design = (hessian.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                  +coefficient[:, None]*values)
    elif getattr(case, "name", "").startswith("A"):
        diffusion = case.diffusion(points)
        convection = case.convection(points)
        reaction = case.reaction(points)
        design = -(hessian*diffusion[:, None, :, :]).sum(dim=(-2, -1))
        design = design+(jacobian*convection[:, None, :]).sum(dim=-1)
        design = design+reaction[:, None]*values
    else:
        raise ValueError("direct RFM collocation is only defined for linear H/A cases")
    target = case.forcing(points)
    # Column equilibration is essential because differentiated local features
    # can differ by several orders of magnitude.
    column_scale = design.square().mean(dim=0).sqrt().clamp_min(1e-8)
    equilibrated = design/column_scale
    # Form the small normal system in float64.  Random local features are
    # intentionally redundant; float32 normal equations can otherwise return
    # a numerically worse solution than the zero field.
    equilibrated64 = equilibrated.double()
    target64 = target.double()
    gram = equilibrated64.T@equilibrated64
    rhs = equilibrated64.T@target64
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    normalized_solution = torch.linalg.solve(
        gram+ridge*gram.diagonal().mean().clamp_min(1e-12)*identity, rhs,
    )
    solution = (normalized_solution/column_scale.double()).to(points.dtype)
    model.set_linear_solution(solution)
    residual = design@solution-target
    return {
        "collocation_points": int(points.shape[0]),
        "ridge": float(ridge),
        "relative_collocation_residual": float(
            torch.linalg.vector_norm(residual)
            /torch.linalg.vector_norm(target).clamp_min(1e-12)
        ),
    }


def build_baseline(name, case):
    if name in ("vanilla_mlp_hardbc", "causal_mlp_hardbc"):
        return VanillaPINN(case)
    if name == "fourier_mlp_hardbc":
        return FourierPINN(case)
    if name == "modified_mlp_hardbc":
        return ModifiedMLPPINN(case)
    if name == "spinn_style_hardbc":
        return SeparablePINN(case)
    if name == "cp_pinn_hardbc":
        return CPPINN(case)
    if name == "spinn_mod_hardbc":
        return SPINNModified(case)
    if name == "piratenet_hardbc":
        return PirateNetPINN(case)
    if name == "rfm_hardbc":
        return RandomFeaturePINN(case)
    raise ValueError(name)
