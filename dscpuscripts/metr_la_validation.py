"""METR-LA's unchanged complete-validation observation policy."""
from __future__ import annotations

import numpy as np
from fulltd.sampling import sequence_hash

POLICY = "complete_fixed_validation_v1"


def selection(dataset):
    record = dict(policy=POLICY, population_size=len(dataset),
                  sampled_size=len(dataset),
                  indices_sha256=sequence_hash(np.arange(len(dataset))))
    return dataset, record
