#!/usr/bin/env bash
# How many cross-sections the interior is worth.
#
# Five points on one curve: 1, 3, 5, 10 and all 20 transverse photographs of the watermelon.
# The subsets are *nested* -- every photograph in the 1-set is in the 3-set and so on -- so the
# curve separates how many from which. The 3-set is the one already trained for the equation (7)
# figure and is reused rather than re-run; the 20-set is the model the paper reports.
#
# Sorted order of the twenty files puts watermelon2 at index 11, watermelon6 at 16, watermelon7
# at 17; the rest of the family is spread over the same order.
#
# Only the transverse family is reduced. The longitudinal one is untouched in every arm, so the
# curve is about the transverse count and not about halving the supervision.
set -u
R=/workspace/rebuild; H=$R/project3/code
export FN_ROOT=$R/worktree GS_ROOT=$R/gaussian-splatting
export FN_PY=$R/mc/envs/fn/bin/python FN_PY_SCORE=$R/mc/envs/score/bin/python
unset PYTHONHOME LD_LIBRARY_PATH; export PYTHONPATH= MPLBACKEND=Agg
cd $FN_ROOT
SRC=data_finetune_images/watermelon/horizontal

mk () {  # mk N indices...
  local n=$1; shift
  [ -d "refs_wm${n}_h" ] || PYTHONPATH=$H/src $FN_PY $H/src/refset.py subset "$SRC" "refs_wm${n}_h" "$@"
}
mk 1  16
mk 5  3 11 16 17 19
mk 10 1 3 5 8 11 13 16 17 18 19

arm () {  # arm N gpu
  local n=$1 g=$2 C=$H/objects/wmn$1.conf
  cp $H/objects/watermelon.conf "$C"
  sed -i "1i # the watermelon on $n transverse photographs, for the slice-count curve." "$C"
  sed -i "s|^REF_H=.*|REF_H=refs_wm${n}_h|" "$C"
  # scored against all twenty, so every arm is asked the same question
  sed -i "s|^EVAL_REF=.*|EVAL_REF=$SRC|" "$C"
  rm -rf wmn$n
  LATTICE=build_watermelon/lattice GPU=$g bash $H/run.sh wmn$n train > $R/wmn$n.log 2>&1
  echo "WMN${n}_DONE" >> $R/slice.stat
}

( arm 1 2 ) &
( arm 5 3 ) &
wait
( arm 10 2 ) &
wait
echo SLICE_ALL_DONE >> $R/slice.stat
