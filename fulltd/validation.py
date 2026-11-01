"""Prespecified validation views; never consume a training random stream."""
import hashlib

import numpy as np

from .protocol import EVALUATIONS

VALIDATION_SEED = 0x54445632
ELECTRICITY_POLICY = "per_run_customer_ceil20pct_without_replacement_v2"
FULL_POLICY = "complete_fixed_validation_v1"


def indices_hash(indices):
    """Canonical parent-index identity, independent of machine byte order."""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


class ValidationView:
    """Map compact indices into the original lazy dataset, without window copies."""
    def __init__(self, parent, indices, metadata):
        self.parent = parent
        self.indices = np.asarray(indices, dtype=np.int64)
        self.indices.flags.writeable = False
        self.validation_metadata = metadata
        self.entity_ids = np.asarray(parent.entity_ids)[self.indices]
        self.origins = np.asarray(parent.origins)[self.indices]

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def __len__(self):
        return len(self.indices)

    def get_batch(self, indices):
        return self.parent.get_batch(self.indices[np.asarray(indices, dtype=np.int64)])

    def inverse(self, values, indices):
        return self.parent.inverse(values, self.indices[np.asarray(indices, dtype=np.int64)])

    def persistence(self, indices):
        return self.parent.persistence(self.indices[np.asarray(indices, dtype=np.int64)])


def validation_view(dataset, epoch, run_id=None):
    """Return this scheduled evaluation's view and exact sampling provenance.

    Electricity samples each customer's eligible origins uniformly without
    replacement, with ceil(n_customer / 5) draws. Equal-customer MSE on this
    stratified sample is unbiased for the complete equal-customer validation
    MSE at a fixed predictor. A stable run identity gives each candidate and
    epoch its own reproducible stream, independent of training random streams.
    """
    if dataset.split not in ("val", "validation") or epoch not in EVALUATIONS or len(dataset) < 1:
        raise ValueError("Validation views require a nonempty validation split and a scheduled epoch")
    if dataset.dataset != "electricity":
        record = dict(policy=FULL_POLICY, epoch=int(epoch), population_size=len(dataset),
                      sampled_size=len(dataset), indices_sha256=indices_hash(np.arange(len(dataset))))
        return dataset, record
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Electricity validation requires its run_id")
    run_hash = hashlib.sha256(run_id.encode("utf-8")).digest()
    run_entropy = np.frombuffer(run_hash, dtype="<u4").tolist()
    entities = np.asarray(dataset.entity_ids)
    if entities.shape != (len(dataset),) or not np.issubdtype(entities.dtype, np.integer) or (entities < 0).any():
        raise ValueError("Electricity validation requires nonnegative integer customer IDs")
    order = np.argsort(entities, kind="stable")
    customers, counts = np.unique(entities[order], return_counts=True)
    selected, customer_counts = [], []
    start = 0
    for customer, count in zip(customers, counts):
        count = int(count)
        size = (count + 4) // 5
        # Use a cryptographic run hash, never Python's process-randomized hash.
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [VALIDATION_SEED, int(epoch), int(customer), *run_entropy])))
        selected.append(rng.choice(order[start:start + count], size=size, replace=False))
        customer_counts.append(dict(customer_id=int(customer), population_size=count, sampled_size=size))
        start += count
    indices = np.sort(np.concatenate(selected))
    record = dict(policy=ELECTRICITY_POLICY, epoch=int(epoch), population_size=len(dataset),
                  sampled_size=len(indices), indices_sha256=indices_hash(indices),
                  rng="numpy.PCG64", validation_seed=VALIDATION_SEED, run_id=run_id,
                  run_stream_sha256=run_hash.hex(), per_customer=customer_counts)
    return ValidationView(dataset, indices, record), record


def validate_sampling_record(dataset_name, epoch, record, run_id=None):
    """Audit sidecar counts without opening validation values or any test data."""
    if not isinstance(record, dict) or record.get("epoch") != epoch:
        raise ValueError("Missing or inconsistent validation subset record")
    expected_policy = ELECTRICITY_POLICY if dataset_name == "electricity" else FULL_POLICY
    if record.get("policy") != expected_policy:
        raise ValueError("Incorrect validation sampling policy")
    n, m = record.get("population_size"), record.get("sampled_size")
    if type(n) is not int or type(m) is not int or not 0 < m <= n:
        raise ValueError("Invalid validation population/sample counts")
    checksum = record.get("indices_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Invalid validation index hash")
    if dataset_name != "electricity":
        if m != n or checksum != indices_hash(np.arange(n)):
            raise ValueError("Non-Electricity validation must use the complete fixed set")
        return
    if record.get("rng") != "numpy.PCG64" or record.get("validation_seed") != VALIDATION_SEED:
        raise ValueError("Incorrect validation random stream")
    if (not isinstance(run_id, str) or not run_id or record.get("run_id") != run_id
            or record.get("run_stream_sha256") != hashlib.sha256(run_id.encode("utf-8")).hexdigest()):
        raise ValueError("Validation random stream does not match its candidate run_id")
    customers = record.get("per_customer")
    if not isinstance(customers, list) or not customers:
        raise ValueError("Missing per-customer validation counts")
    ids, populations, sampled = [], [], []
    for customer in customers:
        c, available, size = (customer.get(k) for k in ("customer_id", "population_size", "sampled_size"))
        if any(type(v) is not int for v in (c, available, size)) or c < 0 or available < 1 or size != (available + 4) // 5:
            raise ValueError("Validation must sample ceil(n_customer/5) windows per customer")
        ids.append(c)
        populations.append(available)
        sampled.append(size)
    if ids != sorted(set(ids)) or sum(populations) != n or sum(sampled) != m:
        raise ValueError("Validation customer counts do not reconcile")
