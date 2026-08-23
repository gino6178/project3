#!/usr/bin/env bash
# Does the formula buy iterations rather than quality?
#
# At 325 outer iterations the two schedules are a wash: each plane's jitter window is about eleven
# cells wide and 325 samples land in it, so the window is oversampled roughly thirty times over and
# an even sequence has nothing left to win. A low-discrepancy walk pays off when the number of
# samples is comparable to the resolution wanted, not when it is far past it.
#
# The prediction that follows is that the formula's advantage grows as the iteration count falls,
# and that is what this measures: the same object at 40, 80 and 160 iterations, both schedules. If
# no advantage appears even at 40, the explanation above is wrong.
set -u
W=/workspace/ovoxel_native
OBJ=${OBJ:-orange_sp}
BASE="SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1"
GPUS="0 1 2 3"

rm -rf "$W/queue9"; mkdir -p "$W/queue9"; : > "$W/ovseqit_status.txt"
i=0
for it in 40 80 160; do
  for sq in cycle random; do
    printf "%s %s" "$it" "$sq" > "$W/queue9/$(printf %02d $i).job"; i=$((i+1))
  done
done

worker () {
  local g=$1 j it sq
  while :; do
    j=$(ls "$W"/queue9/*.job 2>/dev/null | head -1)
    [ -n "$j" ] || break
    mv "$j" "$j.taken" 2>/dev/null || continue
    read -r it sq < "$j.taken"
    TAG=it${it}${sq:0:1} bash "$W/objrun.sh" "$OBJ" "$g" ITERS=$it $BASE SEC_SCHED=$sq \
      > "$W/it${it}${sq:0:1}_$OBJ.log" 2>&1
    printf "%-6s %-7s gpu %s exit %s\n" "$it" "$sq" "$g" "$?" >> "$W/ovseqit_status.txt"
  done
}

for g in $GPUS; do worker "$g" & done
wait
echo SEQIT_DONE >> "$W/ovseqit_status.txt"
