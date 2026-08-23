#!/usr/bin/env bash
# Does settling on a mode beat averaging, when the demands genuinely disagree?
#
# The pixel term is squared error, so a cell that several planes demand different things of ends up
# at their weighted mean. Two photographs of a different orange on the same plane differ by 0.0600
# transverse and 0.0964 longitudinal -- measured -- so the disagreement is a mixture rather than
# noise about one answer, and its mean is a blend of both.
#
# SEC_ROBUST switches the squared error for Geman-McClure, whose weight on a residual falls back
# towards zero as the residual grows: at 0.06 a demand still counts 0.64, at 0.20 it counts 0.02.
# The estimator then sits at a mode. The scale is swept because it is the whole content of the
# method: too large and it is the mean again, too small and nothing is learnt.
#
# Three seeds each, because the pipeline's own spread at 40 iterations is 6.3% and a single run
# cannot see anything smaller than that.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-orange_sp}
BASE="SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1"
GPUS="0 1 2 3"
rm -rf "$W/queue14"; mkdir -p "$W/queue14"; : > "$W/ovrobust_status.txt"
i=0
for c in 0.04 0.08 0.16; do
  for s in 0 1 2; do
    printf "%s %s" "$c" "$s" > "$W/queue14/$(printf %02d $i).job"; i=$((i+1))
  done
done

worker () {
  local g=$1 j c sd
  while :; do
    j=$(ls "$W"/queue14/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r c sd < "$j.taken"
    local tag="rb${c/./}s${sd}"
    TAG=$tag bash "$W/objrun.sh" "$OBJ" "$g" ITERS=40 $BASE SEED=$sd SEC_ROBUST=$c \
      > "$W/${tag}_$OBJ.log" 2>&1
    printf "c=%-5s seed %s gpu %s exit %s\n" "$c" "$sd" "$g" "$?" >> "$W/ovrobust_status.txt"
  done
}
for g in $GPUS; do worker "$g" & done
wait
echo ROBUST_DONE >> "$W/ovrobust_status.txt"
