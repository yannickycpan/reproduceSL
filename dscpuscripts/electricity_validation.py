"""Electricity's frozen, customer-stratified 20% validation subset."""
from __future__ import annotations

import numpy as np
from fulltd.sampling import sequence_hash

VALIDATION_SEED = 0x54445633
POLICY = "fixed_customer_ceil20pct_without_replacement_v1"


def selection(dataset):
    entities = np.asarray(dataset.entity_ids)
    if (entities.shape != (len(dataset),) or not np.issubdtype(entities.dtype, np.integer)
            or bool((entities < 0).any())):
        raise ValueError("Electricity validation requires nonnegative integer customer IDs")
    order = np.argsort(entities, kind="stable")
    customers, counts = np.unique(entities[order], return_counts=True)
    selected, customer_counts = [], []
    start = 0
    for customer, count in zip(customers, counts):
        count = int(count)
        size = (count + 4) // 5
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([VALIDATION_SEED, int(customer)])))
        selected.append(rng.choice(order[start:start + count], size=size, replace=False))
        customer_counts.append(dict(customer_id=int(customer), population_size=count, sampled_size=size))
        start += count
    indices = np.sort(np.concatenate(selected))
    record = dict(policy=POLICY, population_size=len(dataset),
                  sampled_size=len(indices), indices_sha256=sequence_hash(indices),
                  rng="numpy.PCG64", validation_seed=VALIDATION_SEED, per_customer=customer_counts)
    return indices, record
