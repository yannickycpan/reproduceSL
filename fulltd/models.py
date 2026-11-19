"""Fixed compact mean predictors for the TD-v7 application protocol.

Independent PyTorch implementations of the cited architecture operations.
There are no training schedules, normalization of inputs, or loss terms here.
See docs/fulltd_architectures.md for the exact reference revisions and choices.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def initialization_seed(seed: int, stream: str) -> int:
    """Stable independent streams, unaffected by Python's process hash seed."""
    payload = f"fulltd-initialization-v1:{int(seed)}:{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def initialize_modules(module: nn.Module) -> None:
    """Declared initializations; called once after constructing the full mean."""
    for layer in module.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            # TCN and pointwise ReLU paths use Kaiming; gating convolutions
            # are overridden below with a symmetric Xavier initialization.
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


class TemporalResidual(nn.Module):
    """Two strictly causal convolutions and one residual addition."""

    def __init__(self, input_channels: int, channels: int, dilation: int):
        super().__init__()
        self.left_padding = 2 * dilation
        self.first = nn.Conv1d(input_channels, channels, 3, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.projection = (
            nn.Conv1d(input_channels, channels, 1)
            if input_channels != channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.first(F.pad(x, (self.left_padding, 0))))
        value = F.relu(self.second(F.pad(value, (self.left_padding, 0))))
        return F.relu(value + self.projection(x))


class TCN(nn.Module):
    def __init__(self, input_channels: int, blocks: int, outputs: int):
        super().__init__()
        self.feature_dim = 64
        self.output_dim = outputs
        self.blocks = nn.Sequential(*[
            TemporalResidual(input_channels if layer == 0 else 64, 64, 2**layer)
            for layer in range(blocks)
        ])
        self.head = nn.Linear(64, outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)[..., -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResidualAttentionLayer(nn.Module):
    """PatchTST post-norm layer retaining residual attention logits.

    The previous layer's pre-softmax scores are added before this layer's
    softmax, with their gradient path intact (the reference default).
    """

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.attention_output = nn.Linear(64, 64)
        self.attention_norm = nn.LayerNorm(64)
        self.feedforward = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
        self.feedforward_norm = nn.LayerNorm(64)

    def forward(self, x: torch.Tensor, previous_scores: torch.Tensor | None):
        batch, length, _ = x.shape
        def heads(projection):
            return projection(x).reshape(batch, length, 4, 16).transpose(1, 2)
        query, key, value = heads(self.query), heads(self.key), heads(self.value)
        scores = (query @ key.transpose(-2, -1)) * (16**-0.5)
        if previous_scores is not None:
            scores = scores + previous_scores
        attended = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, length, 64)
        x = self.attention_norm(x + self.attention_output(attended))
        x = self.feedforward_norm(x + self.feedforward(x))
        return x, scores


class PatchTST(nn.Module):
    """Shared channel-independent 168-to-24 predictor, 20 unpadded patches."""

    def __init__(self):
        super().__init__()
        self.feature_dim = 20 * 64
        self.output_dim = 24
        self.patch_projection = nn.Linear(16, 64)
        self.position = nn.Parameter(torch.empty(20, 64))
        # Explicit independent construction, never deepcopy a prototype layer.
        self.layers = nn.ModuleList([ResidualAttentionLayer() for _ in range(2)])
        self.head = nn.Linear(self.feature_dim, 24)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, 168):
            raise ValueError("PatchTST expects customer-window input [batch,1,168]")
        patches = x[:, 0].unfold(-1, 16, 8)
        value = self.patch_projection(patches) + self.position
        scores = None
        for layer in self.layers:
            value, scores = layer(value, scores)
        # Reference flatten order is width first, then patch.
        return value.transpose(1, 2).reshape(x.shape[0], self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class DLinear(nn.Module):
    """Shared seasonal/trend temporal projections with kernel-25 decomposition."""

    def __init__(self):
        super().__init__()
        self.output_dim = 24
        self.seasonal = nn.Linear(168, 24)
        self.trend = nn.Linear(168, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, 168):
            raise ValueError("DLinear expects customer-window input [batch,1,168]")
        trend = F.avg_pool1d(F.pad(x, (12, 12), mode="replicate"), kernel_size=25, stride=1)
        return self.seasonal((x - trend)[:, 0]) + self.trend(trend[:, 0])


def row_normalized_adjacency(adjacency: np.ndarray | torch.Tensor) -> torch.Tensor:
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square and in recorded sensor order")
    if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("Supplied adjacency must be finite and nonnegative")
    sums = adjacency.sum(dim=1, keepdim=True)
    return adjacency / torch.where(sums > 0, sums, torch.ones_like(sums))


class DiffusionConvolution(nn.Module):
    def __init__(self):
        super().__init__()
        # Identity plus first/second powers of each of three supports.
        self.mixing = nn.Conv2d(32 * 7, 32, 1)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]):
        terms = [x]
        for support in supports:
            first = torch.einsum("bcnt,nm->bcmt", x, support)
            second = torch.einsum("bcnt,nm->bcmt", first, support)
            terms.extend((first, second))
        return self.mixing(torch.cat(terms, dim=1))


class GraphWaveNet(nn.Module):
    """Fixed compact temporal gating and order-two graph diffusion network."""

    def __init__(self, adjacency: np.ndarray | torch.Tensor):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("forward_support", row_normalized_adjacency(adjacency))
        self.register_buffer("backward_support", row_normalized_adjacency(adjacency.T))
        nodes = adjacency.shape[0]
        self.node_left = nn.Parameter(torch.empty(nodes, 10))
        self.node_right = nn.Parameter(torch.empty(10, nodes))
        self.feature_dim = 192
        self.output_dim = 12
        self.receptive_field = 13
        self.start = nn.Conv2d(2, 32, 1)
        dilations = [1, 2] * 4
        # Conv2d preserves [batch, channels, sensors, time] throughout. The
        # legacy author's Conv1d calls with 2-D kernels are not valid today.
        self.filters = nn.ModuleList([nn.Conv2d(32, 32, (1, 2), dilation=(1, d)) for d in dilations])
        self.gates = nn.ModuleList([nn.Conv2d(32, 32, (1, 2), dilation=(1, d)) for d in dilations])
        self.skips = nn.ModuleList([nn.Conv2d(32, 96, 1) for _ in dilations])
        # The final skip is read before diffusion; the reference's last
        # diffusion output never reaches prediction. Omit that dead module.
        self.graph_layers = nn.ModuleList([DiffusionConvolution() for _ in dilations[:-1]])
        self.end = nn.Conv2d(96, 192, 1)
        # Nodewise linear = reference 1x1 output convolution, with explicit
        # final hidden representation for the matched Gaussian variance head.
        self.head = nn.Linear(192, 12)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2 or x.shape[2] != self.node_left.shape[0]:
            raise ValueError("GraphWaveNet expects [batch,2,sensors,time]")
        if x.shape[-1] != 12:
            raise ValueError("The fixed GraphWaveNet protocol requires 12 history steps")
        value = self.start(F.pad(x, (self.receptive_field - x.shape[-1], 0)))
        supports = [self.forward_support, self.backward_support,
                    F.relu(self.node_left @ self.node_right).softmax(dim=1)]
        skipped = None
        for index, (filtering, gating, skip) in enumerate(zip(self.filters, self.gates, self.skips)):
            residual = value
            gated = torch.tanh(filtering(value)) * torch.sigmoid(gating(value))
            current = skip(gated)
            skipped = current if skipped is None else current + skipped[..., -current.shape[-1]:]
            if index < len(self.graph_layers):
                value = self.graph_layers[index](gated, supports) + residual[..., -gated.shape[-1]:]
        hidden = F.relu(self.end(F.relu(skipped)))
        return hidden[..., -1].transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(dataset: str, backbone: str, seed: int, adjacency=None) -> nn.Module:
    """Construct an identical mean for every method/rate sharing this seed."""
    dataset = dataset.lower().replace("-", "").replace("_", "")
    backbone = backbone.lower().replace("-", "").replace("_", "")
    # fork_rng restores global CPU state: auxiliary objectives cannot affect
    # subsequent independently constructed mean networks.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed(seed, f"mean:{dataset}:{backbone}"))
        if dataset == "electricity" and backbone == "tcn":
            model = TCN(1, 6, 24)
        elif dataset == "fd001" and backbone == "tcn":
            model = TCN(24, 4, 1)
        elif dataset == "electricity" and backbone == "patchtst":
            model = PatchTST()
        elif dataset == "electricity" and backbone == "dlinear":
            model = DLinear()
        elif dataset == "metrla" and backbone in ("graphwavenet", "gwnet"):
            if adjacency is None:
                raise ValueError("GraphWaveNet requires the supplied ordered METR-LA adjacency")
            model = GraphWaveNet(adjacency)
        else:
            raise ValueError(f"Unprescribed dataset/backbone: {dataset}/{backbone}")
        initialize_modules(model)
        if isinstance(model, PatchTST):
            # Small learned positional offsets, fixed initialization policy.
            nn.init.uniform_(model.position, -0.02, 0.02)
        if isinstance(model, GraphWaveNet):
            for layer in list(model.filters) + list(model.gates):
                nn.init.xavier_uniform_(layer.weight)
            # Reference standard-normal adaptive adjacency initialization,
            # independent of the mean convolution initialization stream.
            torch.manual_seed(initialization_seed(seed, "graph-adaptive"))
            nn.init.normal_(model.node_left)
            nn.init.normal_(model.node_right)
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
