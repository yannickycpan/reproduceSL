"""Analytical checks for locked selection, uncertainty, and exact readout fits."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fulltd.analysis import (MATRIX, OTHER_RATES, PROTOCOL_ID, TD_RATES,
                             aggregate_metrics, analyze_results, paired_comparisons,
                             select_candidates, validate_manifest, verify_numerical_tables)
from fulltd.refit import refit_readouts, solve_normal_equations, RefittedPredictor
from fulltd.protocol import make_manifest, digest
from fulltd.sampling import sources_for_round, sequence_hash


def fixture_manifest():
    return make_manifest()


def fixture_accounting(config, failed=False):
    n = 9
    td = config["method"].startswith("td_")
    n_rounds = 0 if failed else 100
    rounds = [dict(epoch=e, source_draws=n,
                   source_sha256=sequence_hash(sources_for_round("fd001", n, config["seed"], e)),
                   partner_sha256=f"fixture-partners-{config['seed']}-{e}" if td else None,
                   type_counts=[n, 0] if td else None, optimizer_steps_cumulative=e)
              for e in range(1, n_rounds + 1)]
    status = dict(protocol_id=PROTOCOL_ID, config=config,
                  state="numerical_failure" if failed else "complete", epochs_completed=n_rounds,
                  identity=dict(config_sha256=digest(config), protocol_id=PROTOCOL_ID,
                                code_sha256="fixture-code", data_sha256="fixture-data", validation_sha256="fixture-val"),
                  initial_mean_sha256=f"fixture-mean-{config['backbone']}-{config['seed']}",
                  n_sources=n, source_draws=n * n_rounds, partner_draws=n * n_rounds if td else 0,
                  update_opportunities=n_rounds, optimizer_steps=n_rounds, empty_mask_batches=0,
                  effective_original_passes=float(n_rounds))
    return status, rounds


class AnalysisTests(unittest.TestCase):
    def test_manifest_checks_matrix_not_count(self):
        manifest = fixture_manifest()
        self.assertEqual(len(validate_manifest(manifest)), 630)
        manifest[0]["alpha"] = 42
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_pooled_last_three_and_failure_rule(self):
        # Candidate 0 has excellent epoch100, but candidate1 wins the declared
        # 80/90/100 average. Candidate2 fails just one seed and must receive inf.
        manifest = fixture_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for config in manifest:
                if config["dataset"] != "fd001":
                    continue
                dest = root / "runs" / config["run_id"]
                dest.mkdir(parents=True)
                h, seed = config["configuration"], config["seed"]
                failed = h == 2 and seed == 4
                state, rounds = fixture_accounting(config, failed)
                (dest / "status.json").write_text(json.dumps(state))
                (dest / "rounds.json").write_text(json.dumps(rounds))
                values = []
                for epoch in range(10, 101, 10):
                    mse = (0 if epoch == 100 else 9) if h == 0 else 4 if h == 1 else .01 if h == 2 else 20 + h
                    # Early validation values cannot influence selection.
                    if epoch < 80:
                        mse = 1e9 if h == 1 else 0
                    values.append(dict(epoch=epoch, mse=mse))
                (dest / "validation.json").write_text(json.dumps(values))
                (dest / "final.pt").touch()
            selected = select_candidates(manifest, root / "runs", root / "out", "fd001")
            self.assertEqual(len(selected["groups"]), 4)
            for group in selected["groups"]:
                self.assertEqual(group["configuration"], 1)
                self.assertEqual(group["score"], 4)
                self.assertEqual(len(group["run_ids"]), 5)
            # A changed source sequence must be caught even if counts still match.
            first = next(c for c in manifest if c["dataset"] == "fd001")
            round_path = root / "runs" / first["run_id"] / "rounds.json"
            original_rounds = round_path.read_text()
            corrupted = json.loads(original_rounds)
            corrupted[0]["source_sha256"] = "incorrect-but-same-count"
            round_path.write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, "Source sequence"):
                select_candidates(manifest, root / "runs", root / "bad-audit", "fd001")
            round_path.write_text(original_rounds)
            # An incomplete candidate cannot be scored as a scientific failure.
            first = next(c for c in manifest if c["dataset"] == "fd001")
            p = root / "runs" / first["run_id"] / "status.json"
            status = json.loads(p.read_text())
            status["state"] = "running"
            p.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "Operationally incomplete"):
                select_candidates(manifest, root / "runs", root / "out2", "fd001")
            self.assertFalse((root / "out2" / "selections.json").exists())

    def test_all_ten_validation_records_required(self):
        manifest = fixture_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = manifest[0]
            p = root / config["run_id"]
            p.mkdir()
            (p / "status.json").write_text(json.dumps(dict(protocol_id=PROTOCOL_ID, config=config,
                                            state="complete", epochs_completed=100)))
            (p / "final.pt").touch()
            (p / "validation.json").write_text(json.dumps([dict(epoch=e, mse=1) for e in (80, 90, 100)]))
            with self.assertRaisesRegex(ValueError, "exactly ten"):
                select_candidates(manifest, root, root / "out")

    def test_all_failed_is_reported_and_preview_does_not_rewrite_selection(self):
        manifest = fixture_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for config in manifest:
                if config["dataset"] != "fd001":
                    continue
                dest = root / config["run_id"]
                dest.mkdir()
                state, _ = fixture_accounting(config, failed=True)
                (dest / "status.json").write_text(json.dumps(state))
            selected = select_candidates(manifest, root, root / "tables", "fd001")
            self.assertTrue(all(g["state"] == "all_configurations_failed" for g in selected["groups"]))
            saved = json.loads((root / "tables" / "selections.json").read_text())
            self.assertEqual(digest(selected), digest(saved))
            preview = select_candidates(manifest, root, root / "no-writes", "fd001", write_outputs=False)
            self.assertEqual(preview, selected)
            self.assertFalse((root / "no-writes").exists())

    def test_statistics_paired_sign_and_independent_recomputation(self):
        rows = []
        for method, offset in (("mse", 2), ("td_full", 0)):
            for seed in range(5):
                rows.append(dict(run_id=f"{method}{seed}", dataset="fd001", backbone="tcn", method=method,
                                 batch_size=256, seed=seed, split="test", units="cycles", horizon="all",
                                 metric="mae", value=seed + 1 + offset))
        normalized, summary = aggregate_metrics(rows)
        td = next(r for r in summary if r["method"] == "td_full")
        self.assertEqual(td["mean"], 3)
        self.assertAlmostEqual(td["sample_sd"], math.sqrt(2.5))
        self.assertAlmostEqual(td["standard_error"], math.sqrt(.5))
        paired = paired_comparisons(normalized)[0]
        self.assertEqual(paired["mean_difference"], -2)
        self.assertEqual(paired["descriptive_t95_upper"], -2)
        self.assertAlmostEqual(paired["relative_improvement"], .4)
        with tempfile.TemporaryDirectory() as tmp:
            from fulltd.analysis import write_csv
            p = Path(tmp)
            write_csv(p / "metrics.csv", rows)
            analyze_results(p / "metrics.csv", p / "tables")
            self.assertEqual(verify_numerical_tables(p / "tables"), 2)
        with self.assertRaises(ValueError):
            aggregate_metrics(rows[:-1])

    def test_persistence_is_one_deterministic_result(self):
        row = dict(run_id="electricity_persistence", dataset="electricity", backbone="persistence",
                   method="daily_persistence", batch_size=256, seed=-1, split="test", units="kW",
                   horizon="24", metric="rmse", value=3)
        _, summary = aggregate_metrics([row])
        self.assertEqual(summary[0]["n"], 1)
        self.assertIsNone(summary[0]["sample_sd"])
        self.assertIsNone(summary[0]["standard_error"])
        self.assertEqual(paired_comparisons([row]), [])

    def test_nonfinite_seed_is_not_dropped(self):
        rows = [dict(run_id=str(s), dataset="fd001", backbone="tcn", method="mse", batch_size=256,
                     seed=s, split="test", units="cycles", horizon="all", metric="rul_score",
                     value=float("inf") if s == 4 else 1) for s in range(5)]
        _, summary = aggregate_metrics(rows)
        self.assertTrue(math.isinf(summary[0]["mean"]))
        self.assertEqual(summary[0]["finite_seeds"], 4)
        self.assertEqual(summary[0]["n"], 5)

    def test_vector_figures_use_the_saved_numerical_table(self):
        from fulltd.analysis import write_csv, read_csv
        rows = [dict(run_id=f"td{s}", dataset="fd001", backbone="tcn", method="td_full",
                     batch_size=256, seed=s, split="test", units="cycles", horizon="all", metric="rmse", value=s+2)
                for s in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_csv(path / "metrics.csv", rows)
            analyze_results(path / "metrics.csv", path / "tables", path / "figures")
            stem = path / "figures" / "primary_fd001_cycles_rmse"
            self.assertTrue(stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
            self.assertIn("<svg", stem.with_suffix(".svg").read_text(encoding="utf-8"))
            self.assertEqual(read_csv(stem.with_suffix(".csv")), read_csv(path / "tables" / "summary_metrics.csv"))


class IdentityFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def features(self, x):
        return x[:, :, 0] + self.dummy * 0


class FixtureData:
    def __init__(self):
        self.x = np.array([[1, 0], [2, 1], [3, -1], [5, 2], [7, -2]], dtype=np.float32)
        self.y = np.array([[2, 1], [3, 4], [-2, 1], [7, 3], [1, -4]], dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def get_batch(self, indices):
        return {"x": self.x[indices, :, None], "y": self.y[indices], "mask": np.ones_like(self.y[indices], dtype=bool)}


class FixtureLaw:
    """Lag1 and lag2 law; boundaries are deliberately not doubly stochastic."""
    def enumerate(self, indices):
        partners = np.full((len(indices), 4), -1, dtype=np.int64)
        types = np.tile([0, 0, 1, 1], (len(indices), 1))
        probs = np.zeros_like(partners, dtype=np.float64)
        for row, i in enumerate(indices):
            groups = [[j for j in (i - lag, i + lag) if 0 <= j < 5] for lag in (1, 2)]
            available = sum(bool(g) for g in groups)
            for t, group in enumerate(groups):
                for k, j in enumerate(group):
                    partners[row, 2 * t + k] = j
                    probs[row, 2 * t + k] = 1 / available / len(group)
        return partners, types, probs


class RefitTests(unittest.TestCase):
    def test_streamed_exact_td_equals_explicit_weighted_lstsq(self):
        data, law, model = FixtureData(), FixtureLaw(), IdentityFeatures()
        theta = np.array([.35, -.2])
        fits = refit_readouts(model, data, law, theta, batch_size=2)
        z = np.column_stack((data.x.astype(np.float64), np.ones(len(data))))
        ols = np.linalg.lstsq(z, data.y.astype(np.float64), rcond=None)[0]
        np.testing.assert_allclose(fits["ols"].coefficients, ols, atol=1e-11)
        partners, types, probabilities = law.enumerate(np.arange(len(data)))
        transformed_x, transformed_y = [], []
        for i in range(len(data)):
            for slot in range(4):
                if probabilities[i, slot] == 0:
                    continue
                j, gamma = partners[i, slot], theta[types[i, slot]]
                weight = np.sqrt(probabilities[i, slot] / len(data))
                transformed_x.append(weight * (z[i] - gamma * z[j]))
                transformed_y.append(weight * (data.y[i] - gamma * data.y[j]))
        # Use double-precision target calculation, as in the streamed equations.
        transformed_y = []
        for i in range(len(data)):
            for slot in range(4):
                if probabilities[i, slot] > 0:
                    j, gamma = partners[i, slot], theta[types[i, slot]]
                    transformed_y.append(np.sqrt(probabilities[i, slot] / len(data)) *
                                         (data.y[i].astype(np.float64) - gamma * data.y[j].astype(np.float64)))
        direct = np.linalg.lstsq(np.array(transformed_x), np.array(transformed_y), rcond=None)[0]
        np.testing.assert_allclose(fits["td"].coefficients, direct, atol=1e-11)
        self.assertLess(fits["td"].diagnostics["relative_normal_equation_residual"], 1e-12)
        predictor = RefittedPredictor(model, fits["td"])
        predicted = predictor(torch.tensor(data.x[:, :, None])).detach().numpy()
        np.testing.assert_allclose(predicted, z @ direct, atol=1e-11)
        self.assertTrue(model.training)  # preserved caller mode

    def test_zero_theta_refits_identical_and_chunk_invariant(self):
        data, law, model = FixtureData(), FixtureLaw(), IdentityFeatures()
        small = refit_readouts(model, data, law, [0, 0], batch_size=1)
        large = refit_readouts(model, data, law, [0, 0], batch_size=5)
        np.testing.assert_allclose(small["ols"].coefficients, small["td"].coefficients, atol=1e-11)
        np.testing.assert_allclose(small["td"].coefficients, large["td"].coefficients, atol=1e-11)

    def test_rank_deficiency_reports_minimum_norm_without_ridge(self):
        z = np.array([[1., 1., 1.], [2., 2., 1.], [3., 3., 1.]])
        y = np.array([[2.], [3.], [4.]])
        fit = solve_normal_equations(z.T @ z, z.T @ y, n_sources=3, n_terms=3, label="test")
        np.testing.assert_allclose(fit.coefficients, [[.5], [.5], [1]], atol=1e-12)
        self.assertEqual(fit.diagnostics["rank"], 2)
        self.assertEqual(fit.diagnostics["regularization"], "none")
        self.assertTrue(math.isinf(fit.diagnostics["full_design_condition_number"]))


if __name__ == "__main__":
    unittest.main()
