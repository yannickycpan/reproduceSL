"""Construct unchanged Electricity PatchTST with its existing initialization."""
import torch
from torch import nn
from fulltd.models import PatchTST, initialization_seed, initialize_modules


def build_model(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed(seed, "mean:electricity:patchtst"))
        model = PatchTST()
        initialize_modules(model)
        nn.init.uniform_(model.position, -0.02, 0.02)
        return model
