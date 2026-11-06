#!/usr/bin/env bash
# Shared defaults for direct Linux CPU servers. Override with environment vars.
export FULLTD_ENV="${FULLTD_ENV:-$HOME/scratch/fulltd/.venv-fulltd}"
export FULLTD_DATA_DIR="${FULLTD_DATA_DIR:-$HOME/scratch/fulltd/data}"
export FULLTD_FD001_GAMMA_RESULTS_DIR="${FULLTD_FD001_GAMMA_RESULTS_DIR:-$HOME/scratch/fulltd/fd001_gamma_v1/results}"
export FULLTD_FD001_GAMMA_OUTPUT_DIR="${FULLTD_FD001_GAMMA_OUTPUT_DIR:-$HOME/scratch/fulltd/fd001_gamma_v1/outputs}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
