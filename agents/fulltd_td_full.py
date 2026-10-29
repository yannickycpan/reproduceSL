"""Full-gradient TD with two unrestricted signed temporal coefficients."""
from agents.fulltd_objectives import TDObjective


class TDFull(TDObjective):
    def __init__(self, model):
        super().__init__(model, coordinates=2)
