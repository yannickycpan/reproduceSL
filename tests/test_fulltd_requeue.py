"""Test rank routing and the single requeue coordinator with mocked Slurm."""
import argparse
import contextlib
import io
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fulltd.hpc import supervise, worker
from fulltd.protocol import digest, make_manifest


def option(command, name):
    return command[command.index(name) + 1]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = make_manifest()[:6]
        self.mapping = [5, 2, 4, 1, 0, 3]
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        mapping_path = self.root / "mapping.json"
        mapping_path.write_text(json.dumps(self.mapping))
        self.args = argparse.Namespace(manifest=str(manifest_path), mapping=str(mapping_path),
            submission_digest=digest(dict(manifest=self.manifest, indices=self.mapping)),
            data_dir=str(self.root / "data"), results_dir=str(self.root / "results"),
            log_dir=str(self.root / "logs"), attempt_dir=str(self.root / "worker_attempt"),
            tasks_per_node=3, offset=3, task_count=3, microbatch=16,
            expected_code_hash="submitted-code", no_auto_requeue=False)
        self.enterContext(patch.dict(os.environ, {"SLURM_PROCID": "1", "SLURM_CPUS_PER_TASK": "1",
            "SLURM_ARRAY_JOB_ID": "4321", "SLURM_ARRAY_TASK_ID": "1", "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "3", "SLURM_RESTART_COUNT": "2"}))
        self.enterContext(patch("fulltd.train.code_hash", return_value="submitted-code"))
        self.enterContext(patch("fulltd.hpc.signal.SIGUSR1", 777, create=True))
        self.set_signal = self.enterContext(patch("fulltd.hpc.signal.signal", return_value=signal.SIG_DFL))
        self.enterContext(patch("torch.set_num_threads"))
        self.enterContext(patch("torch.get_num_interop_threads", return_value=1))
        self.scheduler = self.enterContext(patch("fulltd.hpc.subprocess.run"))
        self.launch = self.enterContext(patch("fulltd.hpc.subprocess.Popen"))
        self.train = self.enterContext(patch("fulltd.train.run_index"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def warn(self):
        self.set_signal.call_args_list[0].args[1](777, None)

    def assert_restored(self):
        self.assertEqual(self.set_signal.call_args.args, (777, signal.SIG_DFL))

    def test_rank_plus_array_offset_routes_to_original_manifest_index(self):
        self.train.return_value = dict(state="complete", next_epoch=101)
        result = worker(self.args)
        self.assertEqual(self.train.call_args.args[1], self.mapping[4])
        self.assertEqual(self.train.call_args.kwargs["device"], "cpu")
        self.assertEqual(self.train.call_args.kwargs["threads"], 1)
        self.assertEqual(self.train.call_args.kwargs["microbatch"], 16)
        self.assertEqual(result["index"], 0)
        self.scheduler.assert_not_called()
        self.assert_restored()

    def test_invalid_rank_and_offset_cannot_run_other_elements_candidates(self):
        for rank, offset in ((-1, 3), (3, 3), (0, -1), (2, 5)):
            with self.subTest(rank=rank, offset=offset):
                self.set_signal.reset_mock()
                os.environ["SLURM_PROCID"] = str(rank)
                self.args.offset = offset
                with self.assertRaisesRegex(ValueError, "outside"):
                    worker(self.args)
                self.assert_restored()
        self.train.assert_not_called()
        self.scheduler.assert_not_called()

    def test_code_change_stops_queued_worker(self):
        self.args.expected_code_hash = "changed"
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            worker(self.args)
        self.train.assert_not_called()
        self.scheduler.assert_not_called()
        self.assert_restored()

    def test_worker_returns_only_durable_epoch_and_never_requeues(self):
        def train(*args, **kwargs):
            self.assertFalse(kwargs["stop_requested"]())
            self.warn()
            self.assertTrue(kwargs["stop_requested"]())
            return dict(state="checkpointed", next_epoch=10, cursor=0)
        self.train.side_effect = train
        result = worker(self.args)
        self.assertEqual(result["next_epoch"], 10)
        self.assertEqual(result["state"], "checkpointed")
        record = json.loads((Path(self.args.attempt_dir) / "rank_1.json").read_text())
        self.assertEqual(record, result)
        self.scheduler.assert_not_called()

    def test_worker_no_checkpoint_is_reported_without_requeue(self):
        def train(*args, **kwargs):
            self.warn()
            return dict(state="stopped_no_checkpoint", next_epoch=1)
        self.train.side_effect = train
        self.assertEqual(worker(self.args)["state"], "stopped_no_checkpoint")
        self.scheduler.assert_not_called()

    def test_worker_cancellation_and_bug_are_not_continuations(self):
        for error in (KeyboardInterrupt(), SystemExit(143), RuntimeError("bug")):
            with self.subTest(error=type(error).__name__):
                self.set_signal.reset_mock()
                self.train.side_effect = error
                with self.assertRaises(type(error)):
                    worker(self.args)
                record = json.loads((Path(self.args.attempt_dir) / "rank_1.json").read_text())
                self.assertEqual(record["state"], "operational_failure")
                self.assert_restored()
        self.scheduler.assert_not_called()

    def test_terminal_candidates_skip_fitting_on_requeue(self):
        run_dir = Path(self.args.results_dir) / self.manifest[0]["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "final.pt").write_bytes(b"test-only-final")
        for state in ("complete", "numerical_failure"):
            with self.subTest(state=state):
                self.set_signal.reset_mock()
                (run_dir / "status.json").write_text(json.dumps(dict(state=state, next_epoch=101,
                    config=self.manifest[0], identity=dict(code_sha256="submitted-code"))))
                self.assertEqual(worker(self.args)["state"], state)
        self.train.assert_not_called()
        self.scheduler.assert_not_called()

    def simulate_step(self, states, warning=True, returncode=0, before=1, after=2, wait_error=None):
        events = []
        process = Mock()
        process.poll.return_value = returncode
        def start(command):
            attempt = Path(option(command, "--attempt-dir"))
            for rank, state in enumerate(states):
                (attempt / f"ready_{rank}.json").write_text("{}")
                if state != "missing":
                    (attempt / f"rank_{rank}.json").write_text(json.dumps(dict(rank=rank, state=state,
                        before_epoch=before, next_epoch=after)))
            if warning:
                self.warn()
            return process
        def wait(timeout):
            events.append("all_workers_drained")
            if wait_error:
                raise wait_error
            return returncode
        process.wait.side_effect = wait
        process.send_signal.side_effect = lambda number: events.append("warning_forwarded")
        self.launch.side_effect = start
        self.scheduler.side_effect = lambda *a, **k: events.append("requeue")
        return process, events

    def test_one_supervisor_requeues_only_its_array_element_after_workers_drain(self):
        process, events = self.simulate_step(["checkpointed", "complete", "numerical_failure"])
        receipt = supervise(self.args)
        self.assertEqual(receipt["state"], "requeue_accepted")
        self.scheduler.assert_called_once_with(["scontrol", "requeue", "4321_1"], check=True, timeout=30)
        self.assertEqual(events, ["warning_forwarded", "all_workers_drained", "requeue"])
        process.send_signal.assert_called_once_with(777)
        command = self.launch.call_args.args[0]
        for flag in ("--nodes=1", "--ntasks=3", "--ntasks-per-node=3", "--cpus-per-task=1",
                     "--ntasks-per-core=1", "--cpu-bind=cores", "--kill-on-bad-exit=0", "--wait=0"):
            self.assertIn(flag, command)
        self.assertEqual(option(command, "--offset"), "3")
        self.assertEqual(option(command, "--task-count"), "3")
        self.assert_restored()

    def test_finished_or_numerical_failure_only_does_not_requeue(self):
        self.simulate_step(["complete", "numerical_failure", "complete"])
        self.assertEqual(supervise(self.args)["state"], "complete")
        self.scheduler.assert_not_called()

    def test_warning_waits_for_every_worker_to_install_its_handler(self):
        process = Mock()
        process.poll.return_value = 0
        events = []
        def start(command):
            attempt = Path(option(command, "--attempt-dir"))
            self.warn()
            calls = 0
            def wait(timeout):
                nonlocal calls
                calls += 1
                if calls == 1:
                    process.send_signal.assert_not_called()
                    for rank in range(3):
                        (attempt / f"ready_{rank}.json").write_text("{}")
                        (attempt / f"rank_{rank}.json").write_text(json.dumps(dict(rank=rank,
                            state="checkpointed", before_epoch=1, next_epoch=2)))
                    raise subprocess.TimeoutExpired(command, timeout)
                events.append("drained")
                return 0
            process.wait.side_effect = wait
            return process
        process.send_signal.side_effect = lambda _: events.append("forwarded")
        self.launch.side_effect = start
        supervise(self.args)
        self.assertEqual(events, ["forwarded", "drained"])
        process.send_signal.assert_called_once_with(777)
        self.scheduler.assert_called_once()

    def test_manual_resume_disables_coordinator_requeue(self):
        self.args.no_auto_requeue = True
        self.simulate_step(["checkpointed"] * 3)
        with self.assertRaisesRegex(RuntimeError, "automatic requeue disabled"):
            supervise(self.args)
        self.scheduler.assert_not_called()

    def test_stopped_without_warning_does_not_requeue(self):
        self.simulate_step(["checkpointed"] * 3, warning=False)
        with self.assertRaisesRegex(RuntimeError, "without a scheduler warning"):
            supervise(self.args)
        self.scheduler.assert_not_called()

    def test_no_epoch_progress_or_no_checkpoint_requires_longer_allocation(self):
        for state, before, after in (("checkpointed", 5, 5), ("stopped_no_checkpoint", 1, 1)):
            with self.subTest(state=state):
                self.set_signal.reset_mock()
                self.simulate_step([state, "complete", "complete"], before=before, after=after)
                with self.assertRaisesRegex(RuntimeError, "increase wall time"):
                    supervise(self.args)
        self.scheduler.assert_not_called()

    def test_worker_failure_missing_record_and_scheduler_error_do_not_requeue(self):
        for states, code in ((["operational_failure", "complete", "complete"], 1),
                             (["missing", "complete", "complete"], 0),
                             (["checkpointed"] * 3, 137)):
            with self.subTest(states=states, code=code):
                self.set_signal.reset_mock()
                self.simulate_step(states, returncode=code)
                with self.assertRaisesRegex(RuntimeError, "operational failure"):
                    supervise(self.args)
        self.scheduler.assert_not_called()

    def test_cancellation_of_coordinator_does_not_requeue(self):
        process, _ = self.simulate_step(["checkpointed"] * 3, wait_error=KeyboardInterrupt())
        process.poll.return_value = None
        with self.assertRaises(KeyboardInterrupt):
            supervise(self.args)
        process.terminate.assert_called_once()
        self.scheduler.assert_not_called()
        self.assert_restored()

    def test_denied_requeue_records_operational_error_without_touching_checkpoints(self):
        self.simulate_step(["checkpointed"] * 3)
        checkpoint = Path(self.args.results_dir) / "durable-test.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"last complete epoch")
        self.scheduler.side_effect = subprocess.CalledProcessError(1, ["scontrol"])
        with self.assertRaises(subprocess.CalledProcessError):
            supervise(self.args)
        records = list((Path(self.args.results_dir) / "allocation_attempts").glob("*/coordinator.json"))
        record = json.loads(records[0].read_text())
        self.assertEqual(record["state"], "operational_failure")
        self.assertIn("error", record)
        self.assertEqual(checkpoint.read_bytes(), b"last complete epoch")

    def test_mismatched_resource_allocation_fails_before_srun(self):
        os.environ["SLURM_NTASKS"] = "2"
        with self.assertRaisesRegex(RuntimeError, "Allocated nodes/tasks"):
            supervise(self.args)
        self.launch.assert_not_called()
        self.scheduler.assert_not_called()


class IntegratedWorkerTest(unittest.TestCase):
    def test_real_training_endpoint_satisfies_worker_result_contract(self):
        import torch
        from agents.fulltd_objectives import make_objective
        from tests.test_fulltd_training import SyntheticData, TinyMean, config_for
        config = config_for(method="mse")
        fitting, validation = SyntheticData(8), SyntheticData(5, split="validation")
        old_threads = torch.get_num_threads()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, old_threads)
        self.addCleanup(torch.use_deterministic_algorithms, old_deterministic)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, mapping = root / "manifest.json", root / "mapping.json"
            manifest.write_text(json.dumps([config]))
            mapping.write_text("[0]")
            args = argparse.Namespace(manifest=str(manifest), mapping=str(mapping),
                submission_digest=digest(dict(manifest=[config], indices=[0])),
                expected_code_hash="synthetic-training-fixture-v1", offset=0, task_count=1,
                microbatch=4, data_dir=str(root / "unused-data"), results_dir=str(root / "results"),
                attempt_dir=str(root / "attempt"))
            def load_dataset(dataset, directory, split):
                self.assertEqual(dataset, "fd001")
                self.assertIn(split, ("train", "validation"))
                return fitting if split == "train" else validation
            def objective(config, dataset):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(1234)
                    return make_objective(config["method"], TinyMean())
            with patch.dict(os.environ, {"SLURM_PROCID": "0", "SLURM_CPUS_PER_TASK": "1"}), \
                    patch("fulltd.train.code_hash", return_value=args.expected_code_hash), \
                    patch("fulltd.train.make_objective_for", side_effect=objective), \
                    patch("fulltd.train.subprocess.check_output", return_value="fixture-revision"), \
                    patch("fulltd.train.platform.processor", return_value="fixture-cpu"), \
                    patch("fulltd.data.load_dataset", side_effect=load_dataset), \
                    patch("fulltd.hpc.signal.SIGUSR1", 777, create=True), \
                    patch("fulltd.hpc.signal.signal", return_value=signal.SIG_DFL), \
                    patch("fulltd.hpc.subprocess.run") as scheduler:
                result = worker(args)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["next_epoch"], 101)
            checkpoint = Path(args.results_dir) / config["run_id"] / "final.pt"
            final = torch.load(checkpoint, weights_only=False)
            self.assertEqual(final["next_epoch"], 101)
            self.assertEqual(final["counters"]["optimizer_steps"], 100)
            self.assertEqual(len(final["validation"]), 10)
            scheduler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
