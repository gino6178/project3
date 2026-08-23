#!/usr/bin/env bash
# The best arm, with the planes walked by a formula instead of drawn.
#
# Drawing a photograph instead of averaging two takes the mean on four of the seven objects and is
# the best arm this project has. The plane schedule is a separate question: the pipeline already
# moves each plane within its own slot, but by an independent draw every time, and independent
# draws clump. Measured on coverage, a low-discrepancy walk reaches 85.6% of the interior in 100
# planes where independent draws reach 77.9%; measured on a noise-start solve of the orange it is
# worth 2.0% of error and 27 points of excess texture.
#
The control is s_rs_*, and it is reused rather than re-run because its settings ARE recorded --
# `rs7.sh` line 17: SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1. That is what made the critic batch
# unsafe to compare against and what makes this one safe. A first version of this file ran its own
# control at REF_SAMPLE=1 alone, which is a different arm: s_rs carries the field prior too.
#
#   s_rs   SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1                  already trained, the control
#   p8     the same, plus SEC_SCHED=cycle                          planes walked by the formula
set -u
W=/workspace/ovoxel_native
OBJS="orange_sp watermelon_sp apple1_sp bread_sp cake2_sp pomegranate2_sp doughnut"
GPUS="0 1 2 3"

rm -rf "$W/queue10"; mkdir -p "$W/queue10"; : > "$W/ovphase7_status.txt"
i=0
for o in $OBJS; do
  printf "%s p8" "$o" > "$W/queue10/$(printf %02d $i).job"; i=$((i+1))
done

worker () {
  local g=$1 j o tag
  while :; do
    j=$(ls "$W"/queue10/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r o tag < "$j.taken"
    TAG=p8 bash "$W/objrun.sh" "$o" "$g" ITERS=325 SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1 \
      SEC_SCHED=cycle > "$W/p8_$o.log" 2>&1
    printf "%-18s %s gpu %s exit %s\n" "$o" "$tag" "$g" "$?" >> "$W/ovphase7_status.txt"
  done
}

for g in $GPUS; do worker "$g" & done
wait
echo PHASE7_DONE >> "$W/ovphase7_status.txt"
