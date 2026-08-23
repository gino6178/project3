#!/usr/bin/env bash
# Balance the two families where they are unbalanced: in space.
#
# Their total weights are already equal by construction. Their per-cell weights are not: measured,
# a cell in the innermost fifth of the radius hears from the longitudinal family eleven times as
# often as from the transverse one, and a cell at the rim hears from them equally. What comes out
# is a field extruded along the axis -- 0.815 of the across-axis variation on the watermelon, 0.648
# on the orange -- which a longitudinal cut shows as vertical stripes on every plane of that
# family, supervised or not.
#
# SEC_FAM_BAL weights a longitudinal crop by its distance from the axis, raised to that power, and
# renormalises so the family's total say is unchanged. 0 is the pipeline as it stands; 1 flattens
# the 1/r the counts follow.
#
# Two seeds per setting, because the pipeline's own spread at 40 iterations is 6.3%.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-watermelon_sp}
BASE="SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1"
GPUS="0 1 2 3"
rm -rf "$W/queue15"; mkdir -p "$W/queue15"; : > "$W/ovbal_status.txt"
i=0
for b in 0 0.5 1.0; do
  for s in 0 1; do
    printf "%s %s" "$b" "$s" > "$W/queue15/$(printf %02d $i).job"; i=$((i+1))
  done
done

worker () {
  local g=$1 j b sd
  while :; do
    j=$(ls "$W"/queue15/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r b sd < "$j.taken"
    local tag="bal${b/./}s${sd}"
    TAG=$tag bash "$W/objrun.sh" "$OBJ" "$g" ITERS=40 $BASE SEED=$sd SEC_FAM_BAL=$b \
      > "$W/${tag}_$OBJ.log" 2>&1
    printf "bal=%-4s seed %s gpu %s exit %s\n" "$b" "$sd" "$g" "$?" >> "$W/ovbal_status.txt"
  done
}
for g in $GPUS; do worker "$g" & done
wait
echo BAL_DONE >> "$W/ovbal_status.txt"
