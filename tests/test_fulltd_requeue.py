"""Exercise recovery decisions without a scheduler, accelerator, or dataset."""
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
from unittest.mock import patch

from fulltd.hpc import worker
from fulltd.protocol import make_manifest


class WorkerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = make_manifest()[:5]
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        mapping = self.root / "mapping.json"
        mapping.write_text(json.dumps(list(range(5))))
        self.args = argparse.Namespace(manifest=str(manifest_path), mapping=str(mapping), offset=0,
                                       data_dir=str(self.root / "data"), results_dir=str(self.root / "results"),
                                       threads=4, microbatch=16, expected_code_hash="submitted-code",
                                       no_auto_requeue=False)
        self.environment = self.enterContext(patch.dict(os.environ, {
            "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_JOB_ID": "4321", "SLURM_RESTART_COUNT": "2"}))
        self.enterContext(patch("fulltd.train.code_hash", return_value="submitted-code"))
        # SIGUSR1 does not exist on Windows. This fixture exercises the exact
        # registration/handler route portably without sending a process signal.
        self.enterContext(patch("fulltd.hpc.signal.SIGUSR1", 777, create=True))
        self.set_signal = self.enterContext(patch("fulltd.hpc.signal.signal", return_value=signal.SIG_DFL))
        self.scheduler = self.enterContext(patch("fulltd.hpc.subprocess.run"))
        self.train = self.enterContext(patch("fulltd.train.run_index"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def run_dir(self, index=0):
        return Path(self.args.results_dir) / self.manifest[index]["run_id"]

    def write_status(self, index=0, **values):
        directory = self.run_dir(index)
        directory.mkdir(parents=True, exist_ok=True)
        status = dict(state="checkpointed", next_epoch=1, cursor=256, validation_cursor=0)
        status.update(values)
        (directory / "status.json").write_text(json.dumps(status))
        return status

    def checkpoint_on_warning(self, **status_values):
        def run(*args, **kwargs):
            self.assertFalse(kwargs["stop_requested"]())
            warning_handler = self.set_signal.call_args_list[0].args[1]
            warning_handler(777, None)
            self.assertTrue(kwargs["stop_requested"]())
            self.write_status(args[1], **status_values)
            return {"state": "checkpointed"}
        self.train.side_effect = run

    def assert_handler_restored(self):
        self.assertEqual(self.set_signal.call_args.args, (777, signal.SIG_DFL))

    def test_mapping_uses_original_index_plus_chunk_offset(self):
        self.args.offset = 2
        os.environ["SLURM_ARRAY_TASK_ID"] = "1"
        self.train.return_value = {"state": "complete"}
        worker(self.args)
        call = self.train.call_args
        self.assertEqual(call.args[1], 3)
        self.assertEqual(call.kwargs["microbatch"], 16)
        self.assertEqual(call.kwargs["device"], "cuda")
        self.scheduler.assert_not_called()
        self.assert_handler_restored()

    def test_warning_requeues_only_current_array_element(self):
        os.environ["SLURM_ARRAY_TASK_ID"] = "3"
        self.checkpoint_on_warning(cursor=512)
        result = worker(self.args)
        self.assertEqual(result["state"], "checkpointed")
        self.scheduler.assert_called_once_with(["scontrol", "requeue", "4321_3"], check=True, timeout=30)
        record = json.loads((self.run_dir(3) / "requeue.json").read_text())
        self.assertEqual(record["state"], "accepted")
        self.assertEqual(record["restart_count"], 2)
        self.assertEqual(record["progress"], [1, 512, 0])
        self.assert_handler_restored()

    def test_negative_or_out_of_range_task_and_offsets_never_train(self):
        for task, offset in ((-1, 0), (0, -1), (5, 0), (1, 4)):
            with self.subTest(task=task, offset=offset):
                self.set_signal.reset_mock()
                os.environ["SLURM_ARRAY_TASK_ID"] = str(task)
                self.args.offset = offset
                with self.assertRaisesRegex(ValueError, "outside the submitted mapping"):
                    worker(self.args)
                self.assert_handler_restored()
        self.train.assert_not_called()
        self.scheduler.assert_not_called()

    def test_invalid_mapped_indices_are_rejected(self):
        for value in (-1, 5, "0", True):
            with self.subTest(value=value):
                self.set_signal.reset_mock()
                Path(self.args.mapping).write_text(json.dumps([value]))
                with self.assertRaisesRegex(ValueError, "Mapped manifest index"):
                    worker(self.args)
                self.assert_handler_restored()
        self.train.assert_not_called()
        self.scheduler.assert_not_called()

    def test_queued_scientific_code_change_stops_before_training(self):
        self.args.expected_code_hash = "different-code"
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            worker(self.args)
        self.train.assert_not_called()
        self.scheduler.assert_not_called()
        self.assert_handler_restored()

    def test_complete_and_numerical_failure_never_requeue_even_after_warning(self):
        for state in ("complete", "numerical_failure"):
            with self.subTest(state=state):
                self.set_signal.reset_mock()
                def run(*args, **kwargs):
                    self.set_signal.call_args_list[0].args[1](777, None)
                    self.assertTrue(kwargs["stop_requested"]())
                    return {"state": state}
                self.train.side_effect = run
                self.assertEqual(worker(self.args)["state"], state)
                self.assert_handler_restored()
        self.scheduler.assert_not_called()

    def test_cancellation_or_training_error_does_not_requeue(self):
        for error in (KeyboardInterrupt(), SystemExit(143), RuntimeError("training bug")):
            with self.subTest(error=type(error).__name__):
                self.set_signal.reset_mock()
                self.train.side_effect = error
                with self.assertRaises(type(error)):
                    worker(self.args)
                self.assert_handler_restored()
        self.scheduler.assert_not_called()
        self.assertFalse((self.run_dir() / "requeue.json").exists())

    def test_denied_requeue_keeps_checkpoint_status_and_records_operational_failure(self):
        self.checkpoint_on_warning()
        self.scheduler.side_effect = subprocess.CalledProcessError(1, ["scontrol", "requeue", "4321_0"])
        with self.assertRaisesRegex(RuntimeError, "Checkpoint preserved"):
            worker(self.args)
        status = json.loads((self.run_dir() / "status.json").read_text())
        self.assertEqual(status["state"], "checkpointed")
        self.assertEqual(status["cursor"], 256)
        record = json.loads((self.run_dir() / "requeue.json").read_text())
        self.assertEqual(record["state"], "requeue_failed")
        self.assertIn("error", record)
        self.assert_handler_restored()

    def test_manual_mode_saves_progress_without_requeue(self):
        self.args.no_auto_requeue = True
        self.checkpoint_on_warning()
        with self.assertRaisesRegex(RuntimeError, "automatic requeue disabled"):
            worker(self.args)
        self.assertTrue((self.run_dir() / "status.json").exists())
        self.scheduler.assert_not_called()

    def test_no_progress_fails_instead_of_requeueing_forever(self):
        self.write_status(cursor=512, validation_cursor=12)
        self.checkpoint_on_warning(cursor=512, validation_cursor=12)
        with self.assertRaisesRegex(RuntimeError, "No progress"):
            worker(self.args)
        self.scheduler.assert_not_called()
        self.assertEqual(json.loads((self.run_dir() / "status.json").read_text())["state"], "checkpointed")

    def test_partial_validation_progress_counts_toward_continuation(self):
        self.write_status(next_epoch=10, cursor=512, validation_cursor=12)
        self.checkpoint_on_warning(next_epoch=10, cursor=512, validation_cursor=24)
        worker(self.args)
        self.scheduler.assert_called_once_with(["scontrol", "requeue", "4321_0"], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
