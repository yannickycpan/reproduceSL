"""Read-only progress report for the 180-run FD001 Gamma campaign.

No Slurm, optimizer, refit, test evaluation, or dataset loading is involved.
This is a quick status report; use fd001_gamma.analysis check for the complete
statistical and provenance audit after all candidates are terminal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from fd001_gamma.common import campaign_paths, read_json, scientific_hash
from fd001_gamma.protocol import digest, make_manifest


TERMINAL = frozenset(("complete", "numerical_failure"))


def inspect(results_dir: Path, *, code_hash: str | None = None):
    results = Path(results_dir)
    expected_hash = scientific_hash() if code_hash is None else code_hash
    rows = []
    for config in make_manifest():
        run_id = config["run_id"]
        root = results / run_id
        status_file = root / "status.json"
        state = "missing"
        epochs = 0
        problem = None
        if status_file.is_file():
            try:
                status = read_json(status_file)
                identity = status.get("identity") or {}
                if (status.get("config") != config
                        or identity.get("config_sha256") != digest(config)
                        or identity.get("code_sha256") != expected_hash):
                    state = "incompatible"
                    problem = "config/scientific-code identity mismatch"
                else:
                    epochs = status.get("epochs_completed", 0)
                    reported = status.get("state", "unknown")
                    if (reported == "complete"
                            and epochs == 100 and status.get("next_epoch") == 101
                            and all((root / name).is_file() for name in
                                    ("final.pt", "validation.json", "rounds.json"))):
                        state = "complete"
                    elif reported == "complete":
                        state = "incomplete_final"
                        problem = "completed status lacks final checkpoint or evidence"
                    elif reported == "numerical_failure":
                        state = "numerical_failure"
                    else:
                        state = reported
            except (OSError, ValueError, TypeError, KeyError) as exc:
                state = "unreadable"
                problem = f"{type(exc).__name__}: {exc}"
        elif (root / "recovery.pt").is_file():
            state = "checkpoint_without_status"
        elif (root / "final.pt").is_file():
            state = "final_checkpoint_without_status"
        elif root.is_dir():
            state = "directory_without_status"
        rows.append(dict(run_id=run_id, state=state,
                         epochs_completed=epochs if isinstance(epochs, int) else 0,
                         problem=problem))
    return rows


def report(rows, *, show=8):
    counts = Counter(row["state"] for row in rows)
    complete = counts["complete"]
    failed = counts["numerical_failure"]
    outstanding = len(rows) - complete - failed
    print(f"FD001 Gamma: {complete}/{len(rows)} complete; "
          f"{failed} numerical failures; {outstanding} unfinished/invalid")
    print("States:", dict(sorted(counts.items())))
    if outstanding:
        active = [row for row in rows if row["state"] not in TERMINAL]
        print("Unfinished/invalid examples:")
        for row in active[:show]:
            print(f"  {row['run_id']}: {row['state']} "
                  f"(epoch {row['epochs_completed']}/100)"
                  + (f" -- {row['problem']}" if row["problem"] else ""))
        if len(active) > show:
            print(f"  ... and {len(active) - show} more")
    if failed:
        print("Numerical failures (terminal, not successful):")
        for row in (r for r in rows if r["state"] == "numerical_failure"):
            print(f"  {row['run_id']} at epoch {row['epochs_completed']}/100")
    if complete == len(rows):
        print("ALL 180 COMPLETE: ready for analysis.")
    elif outstanding == 0:
        print("ALL 180 TERMINAL: check the failure count before interpreting analysis.")
    else:
        print("NOT READY: some candidates are unfinished or incompatible.")
    return complete == len(rows)


def main():
    from dscpuscripts.result_paths import configure_environment
    configure_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Override the existing FD001 Gamma results directory")
    completion = parser.add_mutually_exclusive_group()
    completion.add_argument("--strict", action="store_true",
                            help="Return nonzero unless all 180 runs completed successfully")
    completion.add_argument("--terminal", action="store_true",
                            help="Return zero if all 180 runs are complete or documented numerical failures; "
                                 "the HP selector excludes any failed configuration")
    parser.add_argument("--show", type=int, default=8,
                        help="Number of unfinished candidates to print")
    args = parser.parse_args()
    if args.show < 0:
        parser.error("--show must be nonnegative")
    results = args.results_dir if args.results_dir is not None else campaign_paths()[1]
    print("Results:", results)
    rows = inspect(results)
    if all(row["state"] == "missing" for row in rows):
        # Catch exactly the mistaken nested path from the previous runner
        # release. Never move, rename, or infer successful training from dirs.
        flat = results.parent
        examples = sum((flat / c["run_id"]).is_dir()
                       for c in make_manifest()[:8])
        if examples >= 3:
            print(f"PATH MISMATCH: run directories are under {flat}, not {results}. "
                  "Use the flat dataset_results root; do not retrain them.")
    all_complete = report(rows, show=args.show)
    all_terminal = all(row["state"] in TERMINAL for row in rows)
    if args.strict:
        return 0 if all_complete else 1
    if args.terminal:
        if all_terminal:
            print("All 180 runs terminal; numerical failures are excluded from HP selection.")
        return 0 if all_terminal else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
