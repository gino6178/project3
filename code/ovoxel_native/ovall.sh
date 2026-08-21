#!/usr/bin/env bash
# Every object on the O-Voxel-native representation, with the transverse family's continuous depth
# assignment on -- which is the default now, so this sets nothing and that is the point.
#
# The seven are named by their objects/<obj>.conf, and each brings its own references, physics
# config and lattice from there. Three at a time: GPU 2 belongs to another tenant.
#
# Not carried: SEC_DIST. Chamfer was measured against r1_tb1 at two weights and is worse at both,
# monotonically -- held-out probe 0.02958 -> 0.03014 (w 0.2) -> 0.03090 (w 1.0), the banding 3.78
# -> 4.12 -> 4.38, and the detail falls with it. sw and JS were ruled out before training, on the
# references themselves.
set -u
W=/workspace/ovoxel_native
OBJS="orange_sp watermelon_sp apple1_sp bread_sp cake2_sp pomegranate2_sp doughnut"
GPUS="0 1 3"

rm -rf "$W/queue"; mkdir -p "$W/queue"; : > "$W/ovall_status.txt"
i=0
for o in $OBJS; do printf "%s" "$o" > "$W/queue/$(printf %02d $i).job"; i=$((i+1)); done

worker () {
  local g=$1 j o
  while :; do
    j=$(ls "$W"/queue/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    # mv is the lock: exactly one worker can rename a given file, and the loser just looks again
    mv "$j" "$j.taken" 2>/dev/null || continue
    o=$(cat "$j.taken")
    bash "$W/objrun.sh" "$o" "$g" > "$W/ovall_$o.log" 2>&1
    printf "%-18s gpu %s exit %s\n" "$o" "$g" "$?" >> "$W/ovall_status.txt"
  done
}

for g in $GPUS; do worker "$g" & done
wait
echo ALLOBJ_DONE >> "$W/ovall_status.txt"
