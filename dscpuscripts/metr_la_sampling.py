"""METR-LA source sampling: a configurable fraction of the full training universe.

At 100%, the exact original random reshuffling law is recovered. At smaller
fractions source indices are sampled independently with replacement every
round. This affects *training source exposure*, not the eligible TD partners.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import os

import numpy as np

from fulltd.sampling import rng_for, sources_for_round as original_sources_for_round


def parse_fraction(value):
    try:
        fraction = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("METR-LA source fraction must be numeric") from exc
    if not fraction.is_finite() or not 0 < fraction <= 1:
        raise ValueError("METR-LA source fraction must lie in (0,1]")
    return float(fraction)


SOURCE_FRACTION = parse_fraction(os.environ.get("FULLTD_METR_LA_SOURCE_FRACTION", "0.10"))
ROUNDS = 100


def fraction_tag(fraction):
    value = Decimal(str(parse_fraction(fraction))) * 100
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "f" + text.replace(".", "p") + "pct"


def evaluation_period(fraction):
    f = parse_fraction(fraction)
    return 500 if f == 1.0 else max(1, round(1000 * f))


def draws_per_round(n_sources, fraction):
    if type(n_sources) is not int or n_sources < 1:
        raise ValueError("Invalid METR-LA source count")
    return math.ceil(n_sources * parse_fraction(fraction))


def sources_for_round(n_sources, seed, round_number, fraction):
    """Round-local deterministic draws, shared across methods for a fixed seed."""
    if type(round_number) is not int or not 1 <= round_number <= ROUNDS:
        raise ValueError("Invalid METR-LA round")
    f = parse_fraction(fraction)
    n = draws_per_round(n_sources, f)
    if f == 1.0:
        return original_sources_for_round("metr_la", n_sources, seed, round_number)
    return rng_for("metr_la", seed, round_number, 11).integers(
        0, n_sources, size=n, dtype=np.int64)
