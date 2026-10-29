"""Uniform sources and the unchanged sparse, boundary-aware conditional P.

Electricity: for each round, M=ceil(N/24) iid source draws. Thus for any fixed
parameters E[mean loss] = (1/N) sum_i sum_j P_ij loss_ij. A partner always
comes from the entire eligible training universe, not the sampled round.
Traffic missing-mask ratio normalization is prescribed separately; it is not
claimed to be an unbiased estimator of a fixed-denominator masked loss.
"""
import hashlib
import math
import numpy as np

DATASET_STREAM = {"electricity": 101, "metr_la": 202, "fd001": 303}


def rng_for(dataset, seed, epoch, stream):
    return np.random.default_rng(np.random.SeedSequence([DATASET_STREAM[dataset], seed, epoch, stream]))


def sources_for_round(dataset, n, seed, epoch):
    if n < 1 or epoch not in range(1, 101):
        raise ValueError("Positive source universe and epoch/round 1..100 required")
    rng = rng_for(dataset, seed, epoch, 11)
    if dataset == "electricity":
        return rng.integers(0, n, size=math.ceil(n / 24), dtype=np.int64)
    return rng.permutation(n)


def sequence_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


class LocalPairLaw:
    """O(N) metadata, O(B log N) lookup; no N-by-N or N-by-4 table.

    Origin coordinates must be integer native hours/5-minute steps/cycles.
    Positive temporal gaps are left in those coordinates and never compressed.
    """
    def __init__(self, entity_ids, origins, lags):
        self.entity_ids = np.asarray(entity_ids, dtype=np.int64)
        self.origins = np.asarray(origins, dtype=np.int64)
        if self.entity_ids.shape != self.origins.shape or self.origins.ndim != 1 or not len(self.origins):
            raise ValueError("Nonempty equally shaped entity/origin vectors required")
        self.lags = tuple(lags)
        if len(self.lags) != 2 or any(int(x) != x or x <= 0 for x in self.lags):
            raise ValueError("Two positive integer native-unit lags required")
        self.minimum = int(self.origins.min())
        self.span = int(self.origins.max()) - self.minimum + 1
        self.entity_min = int(self.entity_ids.min())
        if (int(self.entity_ids.max()) - self.entity_min + 1) * self.span >= np.iinfo(np.int64).max:
            raise ValueError("Origin/entity metadata cannot be encoded safely")
        keys = (self.entity_ids - self.entity_min) * self.span + self.origins - self.minimum
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]
        if np.any(np.diff(self.keys) == 0):
            raise ValueError("Duplicate entity-origin examples")

    def enumerate(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        entity = self.entity_ids[indices]
        candidates = self.origins[indices, None] + np.asarray([-self.lags[0], self.lags[0], -self.lags[1], self.lags[1]])
        keys = (entity[:, None] - self.entity_min) * self.span + candidates - self.minimum
        positions = np.searchsorted(self.keys, keys)
        safe = np.minimum(positions, len(self.keys) - 1)
        valid = (positions < len(self.keys)) & (candidates >= self.minimum) & (candidates < self.minimum + self.span) & (self.keys[safe] == keys)
        partners = np.where(valid, self.order[safe], -1)
        valid_types = valid.reshape(-1, 2, 2).any(axis=2)
        type_count = valid_types.sum(axis=1)
        direction_count = valid.reshape(-1, 2, 2).sum(axis=2).repeat(2, axis=1)
        denominator = type_count[:, None] * direction_count
        probabilities = np.divide(valid.astype(np.float64), denominator,
                                  out=np.zeros(valid.shape, dtype=np.float64), where=denominator > 0)
        types = np.broadcast_to(np.array([0, 0, 1, 1]), partners.shape)
        return partners, types, probabilities

    def sample(self, indices, rng):
        partners, types, probabilities = self.enumerate(indices)
        if np.any(probabilities.sum(axis=1) == 0):
            raise ValueError("Isolated sources must be removed from every method's universe before fitting")
        # One inverse-CDF uniform realizes exactly the declared conditional law.
        uniforms = rng.random(len(indices))
        choices = (uniforms[:, None] >= probabilities.cumsum(axis=1)).sum(axis=1)
        choices = np.minimum(choices, 3)
        rows = np.arange(len(indices))
        return partners[rows, choices], types[rows, choices]

    def nonisolated(self, chunk_size=65536):
        keep = []
        for start in range(0, len(self.origins), chunk_size):
            ids = np.arange(start, min(start + chunk_size, len(self.origins)))
            _, _, probabilities = self.enumerate(ids)
            keep.append(ids[probabilities.sum(axis=1) > 0])
        return np.concatenate(keep)


class SourceSubset:
    def __init__(self, dataset, indices):
        self.base = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.entity_ids = dataset.entity_ids[self.indices]
        self.origins = dataset.origins[self.indices]
        self.metadata = dict(dataset.metadata)
        self.metadata["common_source_count"] = len(self.indices)
        self.metadata["isolated_excluded_count"] = len(dataset) - len(self.indices)
        self.metadata["source_index_sha256"] = sequence_hash(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_batch(self, indices):
        return self.base.get_batch(self.indices[indices])

    def inverse(self, values, indices):
        return self.base.inverse(values, self.indices[indices])

    def persistence(self, indices):
        return self.base.persistence(self.indices[indices])


def training_universe(dataset, lags):
    law = LocalPairLaw(dataset.entity_ids, dataset.origins, lags)
    keep = law.nonisolated()
    if not len(keep):
        raise ValueError("No sources with supported training partners")
    isolated = np.setdiff1d(np.arange(len(dataset)), keep, assume_unique=True)
    excluded = [{"entity": int(dataset.entity_ids[i]), "origin": int(dataset.origins[i])} for i in isolated]
    subset = SourceSubset(dataset, keep)
    return subset, LocalPairLaw(subset.entity_ids, subset.origins, lags), excluded
