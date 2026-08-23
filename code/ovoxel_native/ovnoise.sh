#!/usr/bin/env bash
# What a difference has to be, at 40 iterations, before it is a difference.
#
# 40 iterations reached 0.1524 against 325's 0.1499 on one run each, so 40 is where experiments
# should be screened. But the same sweep read 0.1571 at 80 and 0.1626 at 160 -- not monotonic, which
# means the pipeline's own spread is of the same size as the effects being chased. Four seeds of one
# configuration give that spread directly, and every later comparison at 40 iterations is read
# against it.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-orange_sp}
BASE="SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1"
GPUS="0 1 2 3"
rm -rf "$W/queue12"; mkdir -p "$W/queue12"; : > "$W/ovnoise_status.txt"
for s in 0 1 2 3; do printf "%s" "$s" > "$W/queue12/$s.job"; done

worker () {
  local g=$1 j s
  while :; do
    j=$(ls "$W"/queue12/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    s=$(cat "$j.taken")
    TAG=n40s$s bash "$W/objrun.sh" "$OBJ" "$g" ITERS=40 $BASE SEED=$s \
      > "$W/n40s${s}_$OBJ.log" 2>&1
    printf "seed %s gpu %s exit %s\n" "$s" "$g" "$?" >> "$W/ovnoise_status.txt"
  done
}
for g in $GPUS; do worker "$g" & done
wait
echo NOISE_DONE >> "$W/ovnoise_status.txt"
