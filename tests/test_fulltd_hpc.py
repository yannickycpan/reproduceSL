"""Verify the real CPU submission commands without contacting Slurm."""
import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fulltd.hpc import THREAD_VARIABLES, allocation_command, load_submission, main, submit, submit_preflight
from fulltd.protocol import make_manifest


def arguments(root, manifest, **overrides):
    values = dict(manifest=str(manifest), dataset="all", data_dir=str(root / "data with spaces"),
                  results_dir=str(root / "results with spaces"), log_dir=str(root / "logs with spaces"),
                  partition="batch", memory="400G", time="3-00:00:00",
                  account=None, qos=None, constraint=None, dry_run=False, concurrency=7,
                  tasks_per_node=90, microbatch=256, no_auto_requeue=False, signal_seconds=180)
    values.update(overrides)
    return argparse.Namespace(**values)


def wrapped(command):
    outer = shlex.split(command[command.index("--wrap") + 1])
    assert outer[:3] == ["exec", "bash", "-c"]
    assert outer[3].startswith("set -euo pipefail; source ")
    assert "scripts" in outer[3] and "fulltd_environment.sh" in outer[3]
    return ["exec"] + shlex.split(outer[3].rsplit("; exec ", 1)[1])


def option(command, name):
    return command[command.index(name) + 1]


class SubmissionTests(unittest.TestCase):
    def test_training_cli_submits_one_array_without_any_preflight_or_dependency(self):
        for scope, duration, concurrency in (("electricity", "3-00:00:00", "3"),
                                              ("metr_la_fd001", "12:00:00", "4")):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "manifest.json"
                path.write_text(json.dumps(make_manifest()))
                commands = []

                def cityu_sbatch(command, **kwargs):
                    commands.append(command)
                    # Reproduce the precise rejected request in the user's log;
                    # do not infer other site thresholds from that observation.
                    if (option(command, "--partition") == "batch" and "--ntasks=1" in command
                            and option(command, "--mem") == "32G"):
                        raise subprocess.CalledProcessError(1, command, stderr="1 CPUs-per-node and 32768 MB are too small")
                    return str(100 + len(commands)) + "\n"

                argv = ["fulltd.hpc", "--manifest", str(path), "--dataset", scope,
                        "--time", duration, "--concurrency", concurrency,
                        "--data-dir", str(root / "data"), "--results-dir", str(root / "results")]
                with patch("sys.argv", argv), patch("fulltd.hpc.subprocess.check_output", side_effect=cityu_sbatch), \
                        contextlib.redirect_stdout(io.StringIO()):
                    main()
                self.assertEqual(len(commands), 1)
                campaign = commands[0]
                self.assertEqual(option(campaign, "--partition"), "batch")
                self.assertEqual(option(campaign, "--time"), duration)
                self.assertEqual(option(campaign, "--mem"), "400G")
                self.assertEqual(option(campaign, "--array"), f"0-3%{concurrency}")
                self.assertIn("--ntasks=90", campaign)
                self.assertFalse(any(token.startswith("--dependency") for token in campaign))
                self.assertNotIn("fulltd.preflight", wrapped(campaign))
                self.assertFalse(list((root / "results").glob("*preflight*")))
                receipts = list((root / "results").glob(f"submission_{scope}_*.json"))
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text())
                self.assertEqual(receipt["campaign_job"], "101")
                self.assertNotIn("preflight_job", receipt)
                for command in commands:
                    self.assertEqual(command.count("--partition"), 1)
                    self.assertEqual(command.count("--time"), 1)

    def test_preflight_cli_is_independent_and_uses_small_job_defaults(self):
        for scope_args, scope in (([], "all"), (["--dataset", "metr_la_fd001"], "metr_la_fd001")):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                argv = ["fulltd.hpc", "preflight", *scope_args, "--data-dir", str(root / "data"),
                        "--results-dir", str(root / "results")]
                with patch("sys.argv", argv), patch("fulltd.hpc.subprocess.check_output", return_value="100;cluster\n") as sbatch, \
                        contextlib.redirect_stdout(io.StringIO()):
                    main()
                self.assertEqual(sbatch.call_count, 1)
                command = sbatch.call_args.args[0]
                self.assertEqual(option(command, "--partition"), "tiny")
                self.assertEqual(option(command, "--time"), "01:00:00")
                self.assertEqual(option(command, "--mem"), "32G")
                for flag in ("--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "--cpus-per-task=1"):
                    self.assertIn(flag, command)
                self.assertFalse(any(token.startswith(("--array", "--dependency")) for token in command))
                executable = wrapped(command)
                self.assertIn("fulltd.preflight", executable)
                self.assertNotIn("supervise", executable)
                self.assertEqual(option(executable, "--dataset"), scope)
                self.assertEqual(option(executable, "--device"), "cpu")
                self.assertEqual(option(executable, "--threads"), "1")
                receipts = list((root / "results").glob("*.json"))
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text())
                self.assertEqual(receipt["preflight_job"], "100")
                self.assertNotIn("campaign_job", receipt)

    def test_independent_preflight_accepts_resource_overrides(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            receipt = submit_preflight(arguments(root, root / "unused.json", dry_run=True,
                                                partition="custom_small", time="02:00:00", memory="48G"))
            self.assertEqual(len(receipt["commands"]), 1)
            preflight = receipt["commands"][0]
            self.assertEqual(option(preflight, "--partition"), "custom_small")
            self.assertEqual(option(preflight, "--time"), "02:00:00")
            self.assertEqual(option(preflight, "--mem"), "48G")
            self.assertFalse(any(token.startswith(("--array", "--dependency")) for token in preflight))

    def test_preflight_rejection_does_not_submit_any_training_job(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args = arguments(root, root / "unused.json", partition="tiny", memory="32G", time="01:00:00")
            with patch("fulltd.hpc.subprocess.check_output", side_effect=subprocess.CalledProcessError(1, ["sbatch"])) as sbatch:
                with self.assertRaises(RuntimeError):
                    submit_preflight(args)
            self.assertEqual(sbatch.call_count, 1)
            receipts = list(Path(args.results_dir).glob("*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertIsNone(receipt["preflight_job"])
            self.assertNotIn("campaign_job", receipt)
            self.assertEqual(receipt["state"], "submission_failed")

    def test_allocation_environment_and_python_arguments_are_shell_quoted(self):
        command = ["/venv with spaces/bin/python", "-m", "fulltd.hpc", "--data-dir", "/data/it's $(not code)"]
        outer = shlex.split(allocation_command(command, "/repo with spaces"))
        self.assertEqual(outer[:3], ["exec", "bash", "-c"])
        setup, executable = outer[3].rsplit("; exec ", 1)
        self.assertEqual(shlex.split(setup.removeprefix("set -euo pipefail; source ")),
                         [str(Path("/repo with spaces") / "scripts" / "fulltd_environment.sh")])
        self.assertEqual(shlex.split(executable), command)

    def test_group_entrypoints_are_self_contained_and_keep_resource_defaults(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "scripts" / "submit_fulltd.sh").exists())
        environment = (root / "scripts" / "fulltd_environment.sh").read_text()
        self.assertIn("module load Python/3.12.3-GCCcore-13.3.0", environment)
        self.assertIn("/etc/profile.d/modules.sh", environment)
        for group, duration, concurrency in (("electricity", "3-00:00:00", 3),
                                              ("metr_fd001", "12:00:00", 4)):
            entrypoint = (root / "scripts" / f"submit_fulltd_{group}.sh").read_text()
            self.assertIn('source "$ROOT/scripts/fulltd_environment.sh"', entrypoint)
            self.assertIn('exec "$FULLTD_ENV/bin/python" -m fulltd.hpc', entrypoint)
            self.assertIn(f"--time {duration} --concurrency {concurrency}", entrypoint)
            self.assertNotIn("submit_fulltd.sh", entrypoint)

    def test_split_submissions_partition_canonical_630_indices_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root / "manifest.json"
            canonical = make_manifest()
            path.write_text(json.dumps(canonical))
            maps = []
            frozen_paths = []
            for scope, count, limit, concurrency, last in (
                    ("electricity", 300, "3-00:00:00", 3, 30),
                    ("metr_la_fd001", 330, "12:00:00", 4, 60)):
                args = arguments(root, path, dataset=scope, dry_run=True, time=limit, concurrency=concurrency)
                receipt = submit(args)
                self.assertEqual(len(receipt["commands"]), 1)
                campaign = receipt["commands"][0]
                self.assertEqual(receipt["candidate_count"], count)
                self.assertEqual(receipt["array_elements"], 4)
                self.assertEqual(option(campaign, "--array"), f"0-3%{concurrency}")
                self.assertEqual(option(campaign, "--time"), limit)
                self.assertFalse(any(token.startswith("--dependency") for token in campaign))
                argv = wrapped(campaign)
                frozen = argparse.Namespace(manifest=option(argv, "--manifest"), mapping=option(argv, "--mapping"),
                                            submission_digest=option(argv, "--submission-digest"))
                frozen_manifest, indices = load_submission(frozen)
                self.assertEqual(frozen_manifest, canonical)
                self.assertEqual([len(indices[start:start + 90]) for start in range(0, count, 90)], [90, 90, 90, last])
                expected = {"electricity"} if scope == "electricity" else {"metr_la", "fd001"}
                self.assertEqual({canonical[i]["dataset"] for i in indices}, expected)
                maps.append(indices)
                frozen_paths.append(frozen.mapping)
            self.assertFalse(set(maps[0]) & set(maps[1]))
            self.assertEqual(sorted(maps[0] + maps[1]), list(range(630)))
            self.assertNotEqual(*frozen_paths)
            self.assertEqual(len(list(Path(args.results_dir).glob("submission_electricity_*.json"))), 1)
            self.assertEqual(len(list(Path(args.results_dir).glob("submission_metr_la_fd001_*.json"))), 1)

    def test_all_630_indices_once_in_seven_90_process_elements(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root / "manifest.json"
            path.write_text(json.dumps(make_manifest()))
            args = arguments(root, path)
            with patch("fulltd.hpc.subprocess.check_output", return_value="100;cluster\n") as sbatch:
                receipt = submit(args)
            self.assertEqual(sbatch.call_count, 1)
            self.assertEqual(receipt["candidate_count"], 630)
            self.assertEqual(receipt["array_elements"], 7)
            self.assertEqual(len(receipt["commands"]), 1)
            campaign = receipt["commands"][0]
            for flag in ("--nodes=1", "--ntasks=90", "--ntasks-per-node=90", "--cpus-per-task=1",
                         "--ntasks-per-core=1", "--requeue", "--signal=B:USR1@180"):
                self.assertIn(flag, campaign)
            self.assertFalse(any(token.startswith("--dependency") for token in campaign))
            self.assertEqual(option(campaign, "--array"), "0-6%7")
            self.assertEqual(option(campaign, "--mem"), "400G")
            self.assertEqual(option(campaign, "--time"), "3-00:00:00")
            self.assertFalse(any("--gres" in token or "--gpu" in token for c in receipt["commands"] for token in c))
            argv = wrapped(campaign)
            self.assertEqual(argv[0], "exec")
            self.assertIn("supervise", argv)
            frozen = argparse.Namespace(manifest=option(argv, "--manifest"), mapping=option(argv, "--mapping"),
                                        submission_digest=option(argv, "--submission-digest"))
            manifest, mapping = load_submission(frozen)
            seen = [mapping[element * 90 + rank] for element in range(7) for rank in range(90)]
            self.assertEqual(seen, list(range(630)))
            self.assertEqual(len({manifest[i]["run_id"] for i in seen}), 630)
            self.assertEqual(option(argv, "--expected-code-hash"), receipt["scientific_code_sha256"])
            self.assertEqual(option(argv, "--data-dir"), str(Path(args.data_dir).resolve()))
            self.assertTrue(all(os.environ[v] == "1" for v in THREAD_VARIABLES))

    def test_original_file_edits_do_not_change_queued_manifest_or_mapping(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root / "manifest.json"
            manifest = make_manifest()
            path.write_text(json.dumps(manifest))
            args = arguments(root, path, dataset="fd001", dry_run=True)
            first = submit(args)
            argv = wrapped(first["commands"][0])
            paths = [Path(option(argv, flag)) for flag in ("--manifest", "--mapping")]
            saved = [p.read_bytes() for p in paths]
            path.write_text(json.dumps([r for r in manifest if r["dataset"] == "fd001"]))
            submit(args)
            self.assertEqual([p.read_bytes() for p in paths], saved)

    def test_tampered_snapshot_is_rejected_at_submission_and_worker_load(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root / "manifest.json"
            path.write_text(json.dumps(make_manifest()[:2]))
            args = arguments(root, path, dry_run=True)
            receipt = submit(args)
            argv = wrapped(receipt["commands"][0])
            mapping = Path(option(argv, "--mapping"))
            mapping.write_text("[1,0]")
            with self.assertRaisesRegex(ValueError, "Immutable submission"):
                submit(args)
            frozen = argparse.Namespace(manifest=option(argv, "--manifest"), mapping=str(mapping),
                                        submission_digest=option(argv, "--submission-digest"))
            with self.assertRaisesRegex(ValueError, "Immutable submission"):
                load_submission(frozen)

    def test_submission_receipt_survives_command_and_bad_receipt_errors(self):
        for failure in (subprocess.CalledProcessError(1, ["sbatch"]), "not-a-job-id\n"):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "manifest.json"
                path.write_text(json.dumps(make_manifest()[:2]))
                args = arguments(root, path)
                with patch("fulltd.hpc.subprocess.check_output", side_effect=[failure]) as sbatch:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                        submit(args)
                self.assertEqual(sbatch.call_count, 1)
                receipts = list(Path(args.results_dir).glob("submission_all_*.json"))
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text())
                self.assertNotIn("preflight_job", receipt)
                self.assertIsNone(receipt["campaign_job"])
                self.assertEqual(receipt["state"], "submission_failed")
                self.assertIn("error", receipt)

    def test_manual_resume_option_and_custom_warning_reach_supervisor(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root / "manifest.json"
            path.write_text(json.dumps(make_manifest()[:3]))
            receipt = submit(arguments(root, path, dry_run=True, no_auto_requeue=True, signal_seconds=300))
            command = receipt["commands"][0]
            self.assertIn("--signal=B:USR1@300", command)
            self.assertIn("--no-auto-requeue", wrapped(command))

    def test_invalid_resource_counts_fail_before_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for changes in (dict(tasks_per_node=0), dict(concurrency=0), dict(signal_seconds=65536)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    submit(arguments(root, root / "not-needed.json", **changes))


class ScopedPreflightTests(unittest.TestCase):
    def test_only_requested_datasets_are_prepared_and_verified(self):
        from fulltd import preflight
        for scope, expected in (("electricity", ["electricity"]),
                                ("metr_la_fd001", ["metr_la", "fd001"])):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset = MagicMock()
                dataset.__len__.return_value = 8
                dataset.metadata = {"fingerprint": "fixture"}
                argv = ["fulltd.preflight", "--dataset", scope, "--data-dir", str(root),
                        "--output", str(root / "preflight.json")]
                with patch("sys.argv", argv), patch("fulltd.preflight.make_manifest", return_value=[]), \
                        patch("fulltd.preflight.load_dataset", return_value=dataset) as load, \
                        patch("fulltd.preflight.training_universe", return_value=(dataset, None, [])), \
                        patch("fulltd.data.prepare_electricity") as electricity, \
                        patch("fulltd.data.prepare_metr_la") as traffic, \
                        patch("fulltd.data.prepare_fd001") as engines, \
                        patch("fulltd.data.verify_dataset") as verify, \
                        patch("fulltd.preflight.torch.set_num_threads"), \
                        patch("fulltd.preflight.torch.get_num_interop_threads", return_value=1), \
                        contextlib.redirect_stdout(io.StringIO()):
                    preflight.main()
                self.assertEqual([call.args[0] for call in load.call_args_list], expected)
                self.assertEqual([call.args[0] for call in verify.call_args_list], expected)
                for name, prepare in (("electricity", electricity), ("metr_la", traffic), ("fd001", engines)):
                    self.assertEqual(prepare.call_count, int(name in expected))
                report = json.loads((root / "preflight.json").read_text())
                self.assertEqual(report["scope"], scope)
                self.assertEqual(list(report["datasets"]), expected)


if __name__ == "__main__":
    unittest.main()
