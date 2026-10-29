"""The prespecified 630 runs. No data-dependent configuration generation."""
import argparse
import hashlib
import json
from pathlib import Path

from . import PROTOCOL_ID

SEEDS = tuple(range(5))
EVALUATIONS = tuple(range(10, 101, 10))
SELECTION_EPOCHS = (80, 90, 100)
RATES = (0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001)
TD_RATES = ((.0001, .0001), (.0001, .001), (.0003, .0003),
            (.0003, .003), (.001, .001), (.001, .01))
LAGS = {"electricity": (1, 24), "metr_la": (1, 3), "fd001": (1, 5)}
MATRIX = (
    ("electricity", "tcn", ("mse", "td_full", "td_scalar", "gaussian_nll")),
    ("electricity", "patchtst", ("mse", "td_full")),
    ("electricity", "dlinear", ("dlinear_mse",)),
    ("metr_la", "graph_wavenet", ("mse", "td_full", "td_scalar", "gaussian_nll", "mae")),
    ("fd001", "tcn", ("mse", "td_full", "gaussian_nll")),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_manifest():
    runs = []
    for dataset, backbone, methods in MATRIX:
        for method in methods:
            td = method.startswith("td_")
            for batch in ((256, 128) if td else (256,)):
                for configuration, (alpha, beta) in enumerate(TD_RATES if td else [(a, None) for a in RATES]):
                    for seed in SEEDS:
                        runs.append(dict(
                            run_id=f"{dataset}_{backbone}_{method}_b{batch}_h{configuration}_s{seed}",
                            dataset=dataset, backbone=backbone, method=method, batch_size=batch,
                            alpha=alpha, beta=beta, seed=seed, configuration=configuration,
                            protocol_id=PROTOCOL_ID, epochs=100,
                            training_unit="stochastic_round" if dataset == "electricity" else "full_epoch",
                            source_sampling="iid_uniform_with_replacement" if dataset == "electricity" else "random_reshuffling",
                            source_divisor=24 if dataset == "electricity" else 1,
                            validation_epochs=list(EVALUATIONS), selection_epochs=list(SELECTION_EPOCHS)))
    assert len(runs) == 630
    return runs


def validate_config(config):
    expected = {r["run_id"]: r for r in make_manifest()}
    if config.get("run_id") not in expected or config != expected[config["run_id"]]:
        raise ValueError("Configuration differs from the frozen fullTD protocol; do not relabel old runs.")


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or len({r["run_id"] for r in data}) != len(data):
        raise ValueError("Manifest must contain unique explicit run records")
    for record in data:
        validate_config(record)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="configs/fulltd")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest()
    for name in ("manifest", *LAGS):
        subset = manifest if name == "manifest" else [r for r in manifest if r["dataset"] == name]
        (destination / (name + ".json")).write_text(json.dumps(subset, indent=2) + "\n")
    print(f"Wrote {len(manifest)} candidates (450 primary, 180 sensitivity)")


if __name__ == "__main__":
    main()
