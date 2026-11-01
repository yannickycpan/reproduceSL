#!/usr/bin/env bash
# Sourced by the CityU entrypoints and by each submitted allocation.
# Module initializers may reference unset shell variables.
set +u
if ! type module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/Modules/init/bash; do
    if [[ -r "$init" ]]; then
      source "$init"
      type module >/dev/null 2>&1 && break
    fi
  done
fi
if ! type module >/dev/null 2>&1; then
  printf 'CityU environment modules are unavailable in this shell.\n' >&2
  return 1
fi
module load Python/3.12.3-GCCcore-13.3.0
set -u
export FULLTD_ENV="${FULLTD_ENV:-$HOME/scratch/fulltd/.venv-fulltd}"
export FULLTD_DATA_DIR="${FULLTD_DATA_DIR:-$HOME/scratch/fulltd/data}"
export FULLTD_RESULTS_DIR="${FULLTD_RESULTS_DIR:-$HOME/scratch/fulltd/results}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
