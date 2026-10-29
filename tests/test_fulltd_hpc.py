"""No cluster connection: verify the actual submitted arrays and worker mapping."""
import argparse
import contextlib
import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fulltd.hpc import submit, worker
from fulltd.protocol import make_manifest


def arguments(root, manifest, **overrides):
    values = dict(manifest=str(manifest), dataset="all", data_dir=str(root / "data with spaces"),
                  results_dir=str(root / "results with spaces"), log_dir=str(root / "logs with spaces"),
                  partition="test_partition", gres="gpu:1", cpus=4, memory="32G", time="24:00:00",
                  account="test_account", qos=None, constraint=None, dry_run=False,
                  array_chunk=97, concurrency=3, microbatch=32)
    values.update(overrides)
    return argparse.Namespace(**values)


def wrapped(command):
    return shlex.split(command[command.index("--wrap") + 1])


def option(command, name):
    return command[command.index(name) + 1]


class SubmissionTests(unittest.TestCase):
    def test_all_630_candidates_once_with_chained_array_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(make_manifest()))
            args = arguments(root, manifest)
            with patch("fulltd.hpc.subprocess.check_output", side_effect=[f"{100 + i};cluster\n" for i in range(8)]) as sbatch:
                with contextlib.redirect_stdout(io.StringIO()):
                    receipt = submit(args)
            self.assertEqual(receipt["candidate_count"], 630)
            self.assertEqual(sbatch.call_count, 8)  # one preflight and seven arrays
            self.assertEqual(receipt["preflight_job"], "100")
            self.assertIn("fulltd.preflight", wrapped(receipt["commands"][0]))
            seen = []
            for chunk, command in enumerate(receipt["commands"][1:]):
                argv = wrapped(command)
                mapping = json.loads(Path(option(argv, "--mapping")).read_text())
                offset = int(option(argv, "--offset"))
                array = option(command, "--array")
                interval, concurrency = array.split("%")
                low, high = map(int, interval.split("-"))
                self.assertEqual(low, 0)
                self.assertLess(high, args.array_chunk)
                self.assertEqual(int(concurrency), 3)
                self.assertIn("--dependency=afterok:100" + (f",afterany:{100 + chunk}" if chunk else ""), command)
                self.assertEqual(option(argv, "--data-dir"), str(Path(args.data_dir).resolve()))
                seen.extend(mapping[offset:offset + high + 1])
            self.assertEqual(seen, list(range(630)))

    def test_repeated_submission_cannot_remap_existing_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "manifest.json"
            manifest = make_manifest()
            path.write_text(json.dumps(manifest))
            args = arguments(root, path, dataset="fd001", dry_run=True)
            with contextlib.redirect_stdout(io.StringIO()):
                first = submit(args)
            first_argv = wrapped(first["commands"][1])
            first_mapping_path = Path(option(first_argv, "--mapping"))
            first_manifest_path = Path(option(first_argv, "--manifest"))
            old_map = first_mapping_path.read_bytes()
            old_manifest = first_manifest_path.read_bytes()
            # Same externally supplied pathname now contains a dataset-only map.
            path.write_text(json.dumps([r for r in manifest if r["dataset"] == "fd001"]))
            with contextlib.redirect_stdout(io.StringIO()):
                submit(args)
            self.assertEqual(first_mapping_path.read_bytes(), old_map)
            self.assertEqual(first_manifest_path.read_bytes(), old_manifest)
            frozen = json.loads(old_manifest)
            index = json.loads(old_map)[0]
            self.assertEqual(frozen[index]["dataset"], "fd001")

    def test_worker_uses_array_task_id_plus_chunk_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.json"
            path.write_text(json.dumps([90, 91, 92, 93, 94]))
            args = argparse.Namespace(mapping=str(path), offset=2, manifest="frozen.json", data_dir="data",
                                      results_dir="results", threads=4, microbatch=16)
            with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}), patch("fulltd.hpc.subprocess.run") as run:
                worker(args)
            command = run.call_args.args[0]
            self.assertEqual(option(command, "--index"), "93")
            self.assertEqual(option(command, "--microbatch"), "16")
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
