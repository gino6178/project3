#!/usr/bin/env bash
# Stop one training run by its output directory, without stopping the shell that asks.
#
#   report/stop_run.sh wmD
#
# `pkill -f wmD` and hand-rolled `pgrep | kill` loops have taken down the calling shell three
# times in this project, and for the same reason each time: the pattern being searched for is
# itself part of the command line of the process doing the searching. So match only processes
# whose executable is python, read the arguments from /proc rather than from pgrep's own
# formatting, and skip this script, its shell, and every ancestor of it.
set -u
run="${1:?usage: stop_run.sh OUTPUT_DIR}"

skip=" $$ $PPID "
p=$PPID
while [ -r "/proc/$p/stat" ]; do
  p=$(awk '{print $4}' "/proc/$p/stat")
  [ "$p" = 0 ] && break
  skip="$skip$p "
done

n=0
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  case "$skip" in *" $pid "*) continue ;; esac
  [ -r "$d/comm" ] || continue
  case "$(cat "$d/comm" 2>/dev/null)" in python*) ;; *) continue ;; esac
  args=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
  case "$args" in
    *"--output_path $run "*|*"--output_path $run") kill "$pid" && { echo "  stopped $run (pid $pid)"; n=$((n+1)); } ;;
  esac
done
[ "$n" = 0 ] && echo "  no running process for $run"
exit 0
