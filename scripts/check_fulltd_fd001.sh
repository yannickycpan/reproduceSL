#!/usr/bin/env bash
# Read-only verification of all FD001 candidate results. No training is started.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export FULLTD_RESULTS_DIR="${FULLTD_RESULTS_DIR:-$HOME/scratch/fulltd/results}"

python3 - <<'PY'
import json
import os
import sys
from collections import Counter
from pathlib import Path

manifest_path = Path('configs/fulltd/manifest.json')
results = Path(os.environ['FULLTD_RESULTS_DIR'])
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
runs = [item for item in manifest if item['dataset'] == 'fd001']
if len(runs) != 120:
    raise SystemExit(f'ERROR: expected 120 FD001 configurations, found {len(runs)}')

counts = Counter()
by_group = {}
issues = []
for config in runs:
    run_id = config['run_id']
    directory = results / run_id
    group = f"{config['method']}/b{config['batch_size']}"
    by_group.setdefault(group, Counter())
    status_path = directory / 'status.json'
    reason = None
    if not status_path.is_file():
        state = 'missing'
        reason = 'status.json missing'
    else:
        try:
            status = json.loads(status_path.read_text(encoding='utf-8'))
            state = status.get('state', 'unknown')
            if status.get('config') != config:
                reason = 'config mismatch'
            elif state == 'complete':
                required = ['final.pt', 'validation.json', 'rounds.json']
                missing = [name for name in required if not (directory / name).is_file()]
                if status.get('epochs_completed') != 100:
                    reason = 'fewer than 100 completed epochs'
                elif missing:
                    reason = 'missing ' + ', '.join(missing)
                else:
                    validation = json.loads((directory / 'validation.json').read_text())
                    rounds = json.loads((directory / 'rounds.json').read_text())
                    if [r.get('epoch') for r in validation] != list(range(10, 101, 10)):
                        reason = 'incomplete validation records'
                    elif [r.get('epoch') for r in rounds] != list(range(1, 101)):
                        reason = 'incomplete epoch records'
            elif state == 'numerical_failure':
                pass  # Terminal scientific failure; analysis retains this evidence.
            else:
                reason = f'not terminal (state={state}, next_epoch={status.get("next_epoch")})'
        except (OSError, ValueError, TypeError, KeyError) as exc:
            state = 'invalid'
            reason = f'unreadable or invalid record: {exc}'
    category = 'incomplete' if reason else state
    counts[category] += 1
    by_group[group][category] += 1
    if reason:
        issues.append((run_id, reason))

print(f'FD001: {len(runs)} candidates; results: {results}')
print('Summary: ' + ', '.join(f'{key}={counts[key]}' for key in
                           ('complete', 'numerical_failure', 'incomplete')))
for group in sorted(by_group):
    sub = by_group[group]
    print(f'  {group:23s} complete={sub["complete"]:2d}  '
          f'numerical_failure={sub["numerical_failure"]:2d}  incomplete={sub["incomplete"]:2d}')
if issues:
    print('First incomplete/invalid runs:')
    for run_id, reason in issues[:20]:
        print(f'  {run_id}: {reason}')
    print('STOP: FD001 analysis requires every candidate to be terminal.')
    sys.exit(2)
print('READY: all FD001 candidates have terminal records; analysis may proceed.')
PY
