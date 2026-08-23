#!/usr/bin/env bash
# Is the decoder's capacity what separates the supervised planes from the rest?
#
# The argument against is that capacity per cell is not the constraint: every cell already carries
# its own free latent and emits three numbers, so a least-squares fit at that cell returns the
# weighted average of whatever the two families demand of it, and a wider trunk cannot change an
# average. What a bigger trunk does change is how cells share, which is a different mechanism.
#
# So this is a test that can refute that argument. Three trunks at 40 iterations, two seeds each:
# the pipeline's 128x2, a wider 512x2, and a deeper 128x6.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-orange_sp}
BASE="SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1"
GPUS="0 1 2 3"
rm -rf "$W/queue13"; mkdir -p "$W/queue13"; : > "$W/ovcap_status.txt"
i=0
for cell in "128 2 base" "512 2 wide" "128 6 deep"; do
  for s in 0 1; do
    printf "%s %s" "$cell" "$s" > "$W/queue13/$(printf %02d $i).job"; i=$((i+1))
  done
done

worker () {
  local g=$1 j hid lay name sd
  while :; do
    j=$(ls "$W"/queue13/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r hid lay name sd < "$j.taken"
    TAG=cap${name}s${sd} bash "$W/objrun.sh" "$OBJ" "$g" ITERS=40 $BASE SEED=$sd \
      ANCHOR_HID=$hid ANCHOR_LAYERS=$lay > "$W/cap${name}s${sd}_$OBJ.log" 2>&1
    printf "%-5s %sx%s seed %s gpu %s exit %s\n" "$name" "$hid" "$lay" "$sd" "$g" "$?" \
      >> "$W/ovcap_status.txt"
  done
}
for g in $GPUS; do worker "$g" & done
wait
echo CAP_DONE >> "$W/ovcap_status.txt"
