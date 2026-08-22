#!/usr/bin/env bash
# The best arm on the orange, on every object, with its own control.
#
# The orange says a critic on the planes no photograph corresponds to is worth 17% on the transverse
# family: 0.0910 against 0.1099. One object is not a result -- every conclusion this project drew
# from the orange alone has had to be withdrawn at least once -- so this is the same program on all
# seven, twice: once with the critic and once without, identical otherwise.
#
# The control is run again rather than compared against the existing s0_* runs because the settings
# of the orange's own critic run were never recorded, and a comparison whose two halves were
# configured by inference is not a measurement. These fourteen are configured here, in one file.
#
#   SEC_CRITIC=0.3        the weight that measured best on the orange with the critic on every plane
#   SEC_CRITIC_UNSUP=1.0  and the same critic, at full weight, on a plane drawn fresh each step that
#                         no photograph corresponds to
set -u
W=/workspace/ovoxel_native
OBJS="orange_sp watermelon_sp apple1_sp bread_sp cake2_sp pomegranate2_sp doughnut"
GPUS="0 1 2 3"

rm -rf "$W/queue7"; mkdir -p "$W/queue7"; : > "$W/ovcrit7_status.txt"
i=0
for o in $OBJS; do
  printf "%s c7" "$o" > "$W/queue7/$(printf %02d $i).job"; i=$((i+1))
  printf "%s n7" "$o" > "$W/queue7/$(printf %02d $i).job"; i=$((i+1))
done

worker () {
  local g=$1 j o tag
  while :; do
    j=$(ls "$W"/queue7/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r o tag < "$j.taken"
    if [ "$tag" = c7 ]; then
      TAG=c7 bash "$W/objrun.sh" "$o" "$g" ITERS=325 SEC_CRITIC=0.3 SEC_CRITIC_UNSUP=1.0 \
        > "$W/c7_$o.log" 2>&1
    else
      TAG=n7 bash "$W/objrun.sh" "$o" "$g" ITERS=325 > "$W/n7_$o.log" 2>&1
    fi
    printf "%-18s %s gpu %s exit %s\n" "$o" "$tag" "$g" "$?" >> "$W/ovcrit7_status.txt"
  done
}

for g in $GPUS; do worker "$g" & done
wait
echo CRIT7_DONE >> "$W/ovcrit7_status.txt"
