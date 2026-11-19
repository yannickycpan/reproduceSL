"""Electricity-specific source sampling, isolated from METR-LA and FD001."""
from __future__ import annotations

import math
import numpy as np

from fulltd.sampling import rng_for

ELECTRICITY_ROUNDS = 240
SOURCE_DIVISOR = 24


def draws_per_round(n_sources):
    if type(n_sources) is not int or n_sources < 1:
        raise ValueError("Invalid Electricity source count")
    return math.ceil(n_sources / SOURCE_DIVISOR)


def sources_for_round(n_sources, seed, round_number):
    """Independent ceil(N/24) source draws; identical to the original 240-round law."""
    if not 1 <= round_number <= ELECTRICITY_ROUNDS:
        raise ValueError("Invalid Electricity stochastic round")
    return rng_for("electricity", seed, round_number, 11).integers(
        0, n_sources, size=draws_per_round(n_sources), dtype=np.int64)
