#!/usr/bin/env bash
# Every number on the page, from the models on disk, in one command.
#
#   export FN_ROOT=... GS_ROOT=... FN_PY=... FN_PY_SCORE=...
#   bash code/evaluate/run_all.sh              # seven objects, the core set
#   bash code/evaluate/run_all.sh --heavy      # and the resolution and rho sweeps
#
# Results land in $FN_ROOT/measurements: one directory per object holding the raw stdout of
# every tool, and results.json holding what was parsed out of them. Re-running skips what is
# already there, so it resumes; --force re-runs.
#
# FN_PY renders and needs the CUDA rasteriser. FN_PY_SCORE only reads images and needs DreamSim,
# which the render environment does not have; leave it unset and those rows are skipped.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
: "${FN_ROOT:?set FN_ROOT}"
: "${FN_PY:?set FN_PY}"
exec "$FN_PY" "$HERE/measure.py" "$@"
