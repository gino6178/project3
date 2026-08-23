#!/usr/bin/env bash
# Can stratified jitter reach in 200 iterations what independent draws reach in 325?
#
# Two facts have to be held together. The pipeline is largely converged by 40 iterations -- 0.1524
# there against 0.1499 at 325 -- so most of what a schedule could save has already been saved by
# stopping early. And the pipeline's own spread is larger than any schedule's effect: the same arm
# scored 0.1524, 0.1571 and 0.1626 at 40, 80 and 160 iterations, a non-monotonic 3.7% swing, while
# the schedules differ by one or two.
#
# So this runs repeats. Three seeds per cell, and the question is whether strat at 200 lands inside
# the spread of random at 325 rather than whether one number beats another.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-orange_sp}
BASE="SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1"
GPUS="0 1 2 3"

rm -rf "$W/queue11"; mkdir -p "$W/queue11"; : > "$W/ovstrat_status.txt"
i=0
for s in 0 1 2; do
  for cell in "200 strat" "200 random" "325 random"; do
    printf "%s %s" "$cell" "$s" > "$W/queue11/$(printf %02d $i).job"; i=$((i+1))
  done
done

worker () {
  local g=$1 j it sq sd
  while :; do
    j=$(ls "$W"/queue11/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r it sq sd < "$j.taken"
    local tag="st${it}${sq:0:1}s${sd}"
    TAG=$tag bash "$W/objrun.sh" "$OBJ" "$g" ITERS=$it $BASE SEC_SCHED=$sq SEED=$sd \
      > "$W/${tag}_$OBJ.log" 2>&1
    printf "%-5s %-7s seed %s gpu %s exit %s\n" "$it" "$sq" "$sd" "$g" "$?" \
      >> "$W/ovstrat_status.txt"
  done
}

for g in $GPUS; do worker "$g" & done
wait
echo STRAT_DONE >> "$W/ovstrat_status.txt"
