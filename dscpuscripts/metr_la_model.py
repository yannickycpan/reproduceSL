"""Construct unchanged METR-LA Graph WaveNet with its existing initialization."""
import torch
from torch import nn
from fulltd.models import GraphWaveNet, initialization_seed, initialize_modules


def build_model(seed, adjacency):
    if adjacency is None:
        raise ValueError("GraphWaveNet requires the supplied ordered adjacency")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed(seed, "mean:metrla:graphwavenet"))
        model = GraphWaveNet(adjacency)
        initialize_modules(model)
        for layer in list(model.filters) + list(model.gates):
            nn.init.xavier_uniform_(layer.weight)
        torch.manual_seed(initialization_seed(seed, "graph-adaptive"))
        nn.init.normal_(model.node_left)
        nn.init.normal_(model.node_right)
        return model
