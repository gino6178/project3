#!/usr/bin/env bash
# The block-rule control for every object.
#
# ov_<obj> was run with the continuous depth assignment, which is the default. Without the same
# object under the rule it replaces, its numbers say what the model scores and not what the change
# was worth -- and the only before/after that exists is the orange's, on a different reference set.
# This is that control: identical in every respect except REF_TRANS_BLEND=0.
set -u
W=/workspace/ovoxel_native
OBJS="orange_sp watermelon_sp apple1_sp bread_sp cake2_sp pomegranate2_sp doughnut"
GPUS="0 1 3"

rm -rf "$W/queue0"; mkdir -p "$W/queue0"; : > "$W/ovall0_status.txt"
i=0
for o in $OBJS; do printf "%s" "$o" > "$W/queue0/$(printf %02d $i).job"; i=$((i+1)); done

worker () {
  local g=$1 j o
  while :; do
    j=$(ls "$W"/queue0/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    o=$(cat "$j.taken")
    TAG=ov0 bash "$W/objrun.sh" "$o" "$g" REF_TRANS_BLEND=0 > "$W/ovall0_$o.log" 2>&1
    printf "%-18s gpu %s exit %s\n" "$o" "$g" "$?" >> "$W/ovall0_status.txt"
  done
}

for g in $GPUS; do worker "$g" & done
wait
echo ALLOBJ0_DONE >> "$W/ovall0_status.txt"
