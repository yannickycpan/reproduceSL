#!/usr/bin/env bash
# 300 candidates: 90/90/90/30 ranks; at most 3 nodes with the default concurrency.
# 72 hours is an allocation limit, not an estimated campaign completion time.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/submit_fulltd.sh" --time 3-00:00:00 --concurrency 3 "$@" --dataset electricity
