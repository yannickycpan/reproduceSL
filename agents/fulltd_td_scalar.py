"""Full-gradient TD with one coefficient for the same two temporal pair sets."""
from agents.fulltd_objectives import TDObjective


class TDScalar(TDObjective):
    def __init__(self, model):
        super().__init__(model, coordinates=1)
