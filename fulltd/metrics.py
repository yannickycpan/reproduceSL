"""Streaming float64 point-prediction metrics; targets are read only on request."""
import numpy as np
import torch


def rul_score(prediction, target):
    z = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.expm1(np.where(z < 0, -z / 13., z / 10.))
    return float(np.sum(terms, dtype=np.float64))


class MetricSums:
    def __init__(self, equal_entity=False, allow_nonfinite=False):
        self.equal_entity = equal_entity
        self.allow_nonfinite = allow_nonfinite
        self.sums = {}

    def add(self, prediction, target, mask, entities):
        error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
        mask = np.asarray(mask, bool)
        if error.shape != mask.shape:
            raise ValueError("Metric mask shape mismatch")
        if not self.allow_nonfinite and not np.isfinite(error[mask]).all():
            raise FloatingPointError("Nonfinite point predictions")
        # Record all horizons and the all-horizon aggregate without averaging RMSEs.
        for horizon in range(error.shape[-1]):
            self._add_key(horizon + 1, error[..., horizon], mask[..., horizon], entities)
        self._add_key("all", error, mask, entities)

    def _add_key(self, horizon, errors, mask, entities):
        for entity in np.unique(entities) if self.equal_entity else [0]:
            select = np.asarray(entities) == entity if self.equal_entity else np.ones(len(errors), bool)
            valid = mask[select]
            values = errors[select][valid]
            key = (int(entity), str(horizon))
            previous = self.sums.get(key, np.zeros(3, np.float64))
            self.sums[key] = previous + [np.sum(np.abs(values)), np.sum(values * values), len(values)]

    def records(self, units):
        result = []
        horizons = sorted({key[1] for key in self.sums}, key=lambda x: 99999 if x == "all" else int(x))
        for h in horizons:
            values = np.array([v for (entity, horizon), v in self.sums.items() if horizon == h and v[2] > 0])
            if not len(values):
                raise ValueError("No observed targets for metric")
            if self.equal_entity:
                mae = np.mean(values[:, 0] / values[:, 2])
                mse = np.mean(values[:, 1] / values[:, 2])
            else:
                mae = values[:, 0].sum() / values[:, 2].sum()
                mse = values[:, 1].sum() / values[:, 2].sum()
            for metric, value in (("mae", mae), ("mse", mse), ("rmse", np.sqrt(mse))):
                result.append(dict(units=units, horizon=h, metric=metric, value=float(value),
                                   valid_count=int(values[:, 2].sum()), entity_count=len(values),
                                   aggregation="equal_entity" if self.equal_entity else "valid_scalar"))
        return result


@torch.no_grad()
def evaluate(model, dataset, batch_size=256, device="cpu", persistence=False, include_raw=True,
             allow_nonfinite=False):
    """Return records and per-engine endpoints. Never called with test inside training."""
    if model is not None:
        model.eval()
    equal_entity = dataset.dataset in ("electricity", "fd001") and dataset.split != "train"
    # Electricity uses equal customers for every split. FD training uses source weights.
    if dataset.dataset == "electricity":
        equal_entity = True
    standardized = MetricSums(equal_entity, allow_nonfinite)
    raw = MetricSums(equal_entity, allow_nonfinite)
    endpoints = []
    score = 0.
    for start in range(0, len(dataset), batch_size):
        ids = np.arange(start, min(start + batch_size, len(dataset)))
        batch = dataset.get_batch(ids)
        if persistence:
            pred = dataset.persistence(ids)
        else:
            x = torch.as_tensor(np.asarray(batch["x"]), dtype=torch.float32, device=device)
            pred = model(x).detach().cpu().numpy()
        target, mask = batch["y"], batch["mask"]
        entities = dataset.entity_ids[ids]
        standardized.add(pred, target, mask, entities)
        if include_raw:
            raw_pred, raw_target = dataset.inverse(pred.astype(np.float64), ids), dataset.inverse(target.astype(np.float64), ids)
            raw.add(raw_pred, raw_target, mask, entities)
            if dataset.dataset == "fd001":
                score += rul_score(raw_pred[mask], raw_target[mask])
                if dataset.split == "test":
                    endpoints.extend(dict(engine_id=int(e), prediction=float(p), target=float(t))
                                     for e, p, t in zip(entities, raw_pred[:, 0], raw_target[:, 0]))
    records = standardized.records("standardized")
    if include_raw:
        units = {"electricity": "kW", "metr_la": "mph", "fd001": "cycles"}[dataset.dataset]
        records.extend(raw.records(units))
        if dataset.dataset == "fd001":
            records.extend([dict(units="score", horizon="all", metric="rul_score_sum", value=score,
                                 valid_count=len(dataset), aggregation="sum"),
                            dict(units="score", horizon="all", metric="rul_score_mean", value=score / len(dataset),
                                 valid_count=len(dataset), aggregation="mean")])
    return records, endpoints


@torch.no_grad()
def validation_mse(model, dataset, batch_size=256, device="cpu", progress=None, after_batch=None):
    """Complete fixed-set score, resumable at a batch boundary.

    ``progress`` contains only a cursor and float64 sufficient metric sums. A
    requested pause returns None; it never records a partial validation score.
    ``after_batch`` may persist that progress along with the unchanged model.
    """
    if batch_size < 1 or len(dataset) < 1:
        raise ValueError("Positive validation batch and nonempty dataset required")
    model.eval()
    progress = {} if progress is None else progress
    cursor = progress.setdefault("cursor", 0)
    if not 0 <= cursor <= len(dataset):
        raise ValueError("Invalid validation recovery cursor")
    sums = MetricSums(dataset.dataset in ("electricity", "fd001"))
    sums.sums = progress.setdefault("sums", {})
    for start in range(cursor, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        ids = np.arange(start, end)
        batch = dataset.get_batch(ids)
        x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
        prediction = model(x).detach().cpu().numpy()
        sums.add(prediction, batch["y"], batch["mask"], dataset.entity_ids[ids])
        progress["cursor"] = end
        if after_batch is not None and after_batch():
            return None
    return next(r["value"] for r in sums.records("standardized")
                if r["metric"] == "mse" and r["horizon"] == "all")
