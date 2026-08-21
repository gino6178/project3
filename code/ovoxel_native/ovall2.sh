#!/usr/bin/env bash
# Every object again, with the two changes that survived measurement on the orange.
#
#   SEC_JOINT=1   one transverse and one longitudinal plane in every gradient step, each family's
#                 loss scaled so its total weight over an outer iteration is what it always was.
#                 It is the default now, so it is not set here; ITERS is, because grouping takes
#                 16 steps per outer iteration instead of 26 and 325 * 16 = 5,200 is the budget
#                 ov_<obj> ran at.
#   SEC_TV=0.1    the spatial prior on the interior field, at the weight that measured best.
#
# Not carried: JITTER_V. Sweeping the longitudinal plane along its normal asks the model to
# reproduce a central section at an off-centre depth, because every longitudinal photograph is a
# central section, and it costs the transverse 36%.
#
# On the orange, against ov_orange_sp's 0.0872 rh / 0.2409 rv: joint alone 0.0859 / 0.2300, joint
# with the prior 0.0860 / 0.2256. This is the measurement of whether that holds on the rest.
set -u
W=/workspace/ovoxel_native
OBJS="orange_sp watermelon_sp apple1_sp bread_sp cake2_sp pomegranate2_sp doughnut"
GPUS="0 1 3"

rm -rf "$W/queue2"; mkdir -p "$W/queue2"; : > "$W/ovall2_status.txt"
i=0
for o in $OBJS; do printf "%s" "$o" > "$W/queue2/$(printf %02d $i).job"; i=$((i+1)); done

worker () {
  local g=$1 j o
  while :; do
    j=$(ls "$W"/queue2/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    o=$(cat "$j.taken")
    TAG=ov2 bash "$W/objrun.sh" "$o" "$g" ITERS=325 SEC_TV=0.1 > "$W/ovall2_$o.log" 2>&1
    printf "%-18s gpu %s exit %s\n" "$o" "$g" "$?" >> "$W/ovall2_status.txt"
  done
}

for g in $GPUS; do worker "$g" & done
wait
echo ALLOBJ2_DONE >> "$W/ovall2_status.txt"
