#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/dscpuscripts/metr_la_sampling_env.sh"
exec bash "$root/dscpuscripts/run_metr_la.sh" "$@"
