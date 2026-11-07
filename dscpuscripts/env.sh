#!/usr/bin/env bash
# Shared result layout on all machines: ~/scratch/fulltd/<dataset>_results/.
# Only environment setup and launch mechanism differ between servers.
# These are operational paths; no scientific training hash includes this file.
export FULLTD_RESULTS_ROOT="${FULLTD_RESULTS_ROOT:-$HOME/scratch/fulltd}"
export FULLTD_ENV="${FULLTD_ENV:-$FULLTD_RESULTS_ROOT/.venv-fulltd}"
export FULLTD_DATA_DIR="${FULLTD_DATA_DIR:-$FULLTD_RESULTS_ROOT/data}"
export FULLTD_FD001_ROOT="${FULLTD_FD001_ROOT:-$FULLTD_RESULTS_ROOT/fd001_results}"
export FULLTD_ELECTRICITY_ROOT="${FULLTD_ELECTRICITY_ROOT:-$FULLTD_RESULTS_ROOT/electricity_results}"
export FULLTD_METR_LA_ROOT="${FULLTD_METR_LA_ROOT:-$FULLTD_RESULTS_ROOT/metr_la_results}"
# Original 180 Gamma runs, 140-run extension and 60 baseline runs all share
# fd001_results/results/; their run IDs are disjoint.
export FULLTD_FD001_GAMMA_RESULTS_DIR="${FULLTD_FD001_GAMMA_RESULTS_DIR:-$FULLTD_FD001_ROOT/results}"
export FULLTD_FD001_GAMMA_OUTPUT_DIR="${FULLTD_FD001_GAMMA_OUTPUT_DIR:-$FULLTD_FD001_ROOT/outputs}"
export FULLTD_FD001_GAMMA_EXT_RESULTS_DIR="${FULLTD_FD001_GAMMA_EXT_RESULTS_DIR:-$FULLTD_FD001_ROOT/results}"
export FULLTD_FD001_GAMMA_EXT_OUTPUT_DIR="${FULLTD_FD001_GAMMA_EXT_OUTPUT_DIR:-$FULLTD_FD001_ROOT/outputs/extension}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
