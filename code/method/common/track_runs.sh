#!/usr/bin/env bash
# Status of the training runs that are actually current, one line each. Called on a timer so a
# run never sits finished-and-unnoticed and a dead one is found in minutes.
#
# Only logs touched in the last two hours count: this directory holds dozens of finished runs
# and listing them all buries the two that matter. The process test excludes this script's own
# shell, which `pgrep -f` otherwise matches through the command line it was given.
cd "${FN_ROOT:-/home/gino/project/FruitNinja_clean}"
for f in $(find . -maxdepth 1 -name 'log_*.log' -mmin -120 | sort); do
  r=$(basename "$f"); r=${r#log_}; r=${r%.log}
  [ -d "$r" ] || continue
  n=$(grep -ac 'Starting iteration' "$f" 2>/dev/null | head -1)
  tot=$(grep -oE 'ABL_ITERS=[0-9]+' "run_$r.sh" 2>/dev/null | head -1 | cut -d= -f2)
  live=$(pgrep -f "train_voxel.py" | while read p; do
           tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "output_path $r " && echo y; done)
  if [ -n "$live" ]; then st=running
  elif ls "$r"/*.ply >/dev/null 2>&1; then st=finished
  else st="DIED at $n"; fi
  printf '  %-6s %4s/%-4s %s\n' "$r" "$n" "${tot:-?}" "$st"
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  gpu /'
