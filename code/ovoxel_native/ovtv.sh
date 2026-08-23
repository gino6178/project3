#!/usr/bin/env bash
# The prior meant to break the columns has never been strong enough to be tested.
#
# fieldreg.py already names the defect: columns running along the polar axis, made of disagreement
# across it, and SEC_TV_PERP puts the weight on the two perpendicular directions to break them.
# Measured on the trained field, it has not: across the axis the field still varies 0.069 and along
# it 0.057. If the prior were deciding those modes the ratio would be about 2 in the other
# direction, so at SEC_TV=0.1 the data term simply overwhelms it.
#
# This sweeps the overall strength, which is the lever that was never moved, at two anisotropies.
# What must be watched is the cost: the same weight that flattens the columns also flattens the
# rind ring and the seeds, which are radial too.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-watermelon_sp}
GPUS="0 1 2 3"
rm -rf "$W/queue16"; mkdir -p "$W/queue16"; : > "$W/ovtv_status.txt"
i=0
for tv in 0.4 1.6 6.4; do
  for pp in 4 16; do
    printf "%s %s" "$tv" "$pp" > "$W/queue16/$(printf %02d $i).job"; i=$((i+1))
  done
done

worker () {
  local g=$1 j tv pp
  while :; do
    j=$(ls "$W"/queue16/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r tv pp < "$j.taken"
    local tag="tv${tv/./}p${pp}"
    TAG=$tag bash "$W/objrun.sh" "$OBJ" "$g" ITERS=40 SEC_TV=$tv SEC_TV_PERP=$pp \
      REF_SAMPLE=1 SEED=0 > "$W/${tag}_$OBJ.log" 2>&1
    printf "tv=%-5s perp=%-3s gpu %s exit %s\n" "$tv" "$pp" "$g" "$?" >> "$W/ovtv_status.txt"
  done
}
for g in $GPUS; do worker "$g" & done
wait
echo TV_DONE >> "$W/ovtv_status.txt"
