#!/usr/bin/env bash
# The baseline, as a script rather than as a memory.
#
#     bash baseline.sh            # train seven, score seven, redraw the page's figures
#     bash baseline.sh score      # score only
#     bash baseline.sh figures    # redraw only
#
# What "baseline" names, exactly:
#
#   cameras   cams_<obj>_v2.npz -- the polar axis each conf names (UP_AXIS, where the object's
#             released model does not stand the way the orange's does), and a distance far enough
#             back that the object is inside the frame. mvcams.py derives both; nothing is per
#             object except the one line in the conf.
#   run       s_v2_<obj>, ITERS=325 SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1
#   scoring   scorefull.py against the same cameras, which it now refuses to do otherwise
#   figures   mapall.py (the seven-object setup sheet), ovcut.py (the sweeps),
#             pairfig.py (supervised above, held out below)
#
# The three defects this baseline exists because of are on the page under #setup. Two of them were
# a setting inherited from the orange by every object; the third was a run trained on one plane
# split and scored on another. All three were invisible in every number and visible in one picture.
set -u
W=/workspace/ovoxel_native
OBJS="${OBJS:-orange_sp watermelon_sp apple1_sp bread_sp cake2_sp pomegranate2_sp doughnut}"
WHAT="${1:-all}"

if [ "$WHAT" = "all" ]; then
  rm -rf "$W/queue_bl"; mkdir -p "$W/queue_bl"; : > "$W/baseline_status.txt"
  i=0
  for o in $OBJS; do printf "%s" "$o" > "$W/queue_bl/$(printf %02d $i).job"; i=$((i+1)); done
  worker () {
    local g=$1 j o
    while :; do
      j=$(ls "$W"/queue_bl/*.job 2>/dev/null | head -1)
      [ -n "$j" ] || break
      mv "$j" "$j.taken" 2>/dev/null || continue
      read -r o < "$j.taken"
      env TAG=s_v2 CAMS_SUFFIX=_v2 bash "$W/objrun.sh" "$o" "$g" ITERS=325 \
        SEC_TV=0.1 SEC_TV_PERP=4 REF_SAMPLE=1 > "$W/s_v2_$o.log" 2>&1
      printf "%-18s gpu %s exit %s\n" "$o" "$g" "$?" >> "$W/baseline_status.txt"
    done
  }
  for g in 0 1 2 3; do worker "$g" & done
  wait
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "score" ]; then
  for o in $OBJS; do
    PYTHONPATH= CUDA_VISIBLE_DEVICES=0 OBJ="$o" SF_RUNS="s_v2_$o" SF_CAMS=_v2 \
      "$W/env/bin/python" "$W/scorefull.py"
  done
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "figures" ]; then
  PYTHONPATH= CUDA_VISIBLE_DEVICES=0 RES=280 "$W/env/bin/python" "$W/mapall.py"
  mkdir -p "$W/vid_v2"
  for o in $OBJS; do
    PYTHONPATH= CUDA_VISIBLE_DEVICES=0 "$W/env/bin/python" "$W/ovcut.py" "$o" "$W/vid_v2" s_v2
    PYTHONPATH= CUDA_VISIBLE_DEVICES=0 OBJ="$o" "$W/env/bin/python" "$W/pairfig.py"
  done
fi
