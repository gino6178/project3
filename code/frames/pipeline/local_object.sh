#!/bin/bash
# One new object, from a line in objects.json to a running job on the GPU machine.  Run it repeatedly:
# every step skips itself when its output exists, and it stops where a person has to look.
#   bash local_object.sh <object>
#
#   1 generate   Gemini: 9 lengthwise, 9 transverse cut faces, 4 exteriors -> aiobj/<o>/raw, exterior
#   2 review     a contact sheet is written; STOP until aiobj/<o>/REVIEWED exists. Record rejections in
#                aiobj/REJECT.txt first (a second object, perspective, the wrong cut, text). EXT_PICK=k
#                chooses the exterior (default 0), written into REVIEWED.
#   3 shape      TRELLIS.2 on p3-fast: exterior -> mesh -> grid, height/width matched to the lengthwise
#                photographs; a thin object (aspect > 2) gets a 256^3 grid
#   4 prepare    curved: the bend turned into the theta=0 plane, the centreline written, lengthwise photographs
#                straightened; all: photographs framed to the grid, cross-sections centred, polar strips
#   5 submit     uploaded to p3-train-4g and appended to the queue that sched.sh's workers drain
set -eu
o=$1; H=$(cd "$(dirname "$0")" && pwd); FR=$H/..; ROOT=${AIROOT:-$HOME/project/Fruit3D_Fusion/aiobj}; EXTR=${EXTROOT:-$HOME/project/Fruit3D_Fusion/aiobj_ext}
PYL=${PYL:-$HOME/miniconda3/envs/fruitninja/bin/python}; export PATH=$HOME/google-cloud-sdk/bin:$PATH
ZF=us-central1-b; ZG=us-central1-a
CURVED=$(python3 -c "import json;print(int(json.load(open('$H/objects.json'))['$o']['curved']))")
say(){ echo "== $o: $*"; }
# 1 generate
if [ ! -f $ROOT/$o/raw/trans_8.png ] || [ ! -f $ROOT/$o/exterior/ext_3.png ]; then
  say "generating with Gemini"; python3 $H/gen.py $ROOT $o
fi
# 2 review
if [ ! -f $ROOT/$o/REVIEWED ]; then
  $PYL $H/sheet.py $ROOT $o; say "review $ROOT/$o/sheet.png, add rejections to $ROOT/REJECT.txt, then: echo \${EXT_PICK:-0} > $ROOT/$o/REVIEWED"; exit 0
fi
PICK=$(cat $ROOT/$o/REVIEWED); PICK=${PICK:-0}
# 3 shape
mkdir -p $EXTR/$o
if [ ! -f $EXTR/$o/ext.pt ]; then
  ASP=$($PYL $FR/photo_aspect.py $ROOT $o); N=128; python3 -c "import sys;sys.exit(0 if $ASP>2 else 1)" && N=256
  say "TRELLIS.2 (aspect $ASP, grid $N)"
  gcloud compute instances start p3-fast --zone $ZF >/dev/null 2>&1 || true
  until timeout 60 gcloud compute ssh p3-fast --zone $ZF --command true >/dev/null 2>&1; do sleep 15; done
  gcloud compute scp $ROOT/$o/exterior/ext_$PICK.png p3-fast:~/ext_in_$o.png --zone $ZF >/dev/null 2>&1
  gcloud compute scp $FR/glb2grid.py p3-fast:~/repro/frames/ --zone $ZF >/dev/null 2>&1
  timeout 3000 gcloud compute ssh p3-fast --zone $ZF --command "mkdir -p ~/ext_in && rm -f ~/ext_in/*.png && mv ~/ext_in_$o.png ~/ext_in/$o.png && source ~/mf/etc/profile.d/conda.sh && conda activate trellis2 && cd ~ && ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers python run_trellis_batch.py > ~/trellis_$o.log 2>&1; cd ~/repro/frames && ASPECT=$ASP python glb2grid.py ~/glb/$o.glb ~/${o}_ext.pt $N > ~/grid_$o.log 2>&1" >/dev/null 2>&1 || true
  gcloud compute scp p3-fast:~/${o}_ext.pt $EXTR/$o/ext.pt --zone $ZF >/dev/null 2>&1
  [ -f $EXTR/$o/ext.pt ] || { say "TRELLIS.2 or glb2grid failed: see ~/trellis_$o.log, ~/grid_$o.log on p3-fast"; exit 1; }
fi
# 4 prepare
G=$EXTR/$o/ext.pt; ST=""
if [ "$CURVED" = 1 ]; then $PYL $FR/orient_bend.py $EXTR/$o/ext.pt $EXTR/$o/ext_o.pt $EXTR/$o/centerline.json; G=$EXTR/$o/ext_o.pt; ST=1; fi
rm -rf $EXTR/$o/raw; cp -r $ROOT/$o/raw $EXTR/$o/raw; cp $ROOT/REJECT.txt $EXTR/REJECT.txt
CUDA_VISIBLE_DEVICES= GRID=$G STRAIGHTEN=$ST $PYL $FR/prep_ai.py $EXTR $o
[ "$CURVED" = 1 ] && cp $EXTR/$o/centerline.json $EXTR/$o/centerline.json 2>/dev/null || true
# 5 submit
R=ai_$o                                     # the remote name: never the same as a synthetic solid's
TGZ=/tmp/${R}_obj.tgz; rm -rf /tmp/$R && mkdir -p /tmp/$R && (cd $EXTR/$o && cp -r $(ls -d grid.pt meta.json spl_* hld_* polar_spl_trans centerline.json 2>/dev/null) /tmp/$R/) && (cd /tmp && tar czf $TGZ $R)
gcloud compute instances start p3-train-4g --zone $ZG >/dev/null 2>&1 || true
until timeout 60 gcloud compute ssh p3-train-4g --zone $ZG --command true >/dev/null 2>&1; do sleep 15; done
gcloud compute scp $TGZ $H/remote_object.sh $FR/eval_curved.py $FR/x3dcurv.py $FR/x3dcurv2.py $FR/x3dcart.py $FR/train_prior.py $FR/make_gsply.py $FR/ov2grid.py $FR/grid2ov.py $FR/showcase.sh $H/sched_ai.sh p3-train-4g:~/ --zone $ZG >/dev/null 2>&1
timeout 120 gcloud compute ssh p3-train-4g --zone $ZG --command "mkdir -p ~/ov ~/repro/frames && cp ~/remote_object.sh ~/eval_curved.py ~/x3dcurv.py ~/x3dcurv2.py ~/x3dcart.py ~/train_prior.py ~/make_gsply.py ~/ov2grid.py ~/grid2ov.py ~/showcase.sh ~/sched_ai.sh ~/repro/frames/ && cp ~/make_gsply.py ~/ov2grid.py ~/ov/ && tar xzf ~/${R}_obj.tgz -C ~/ov/ && until mkdir ~/ov/.lock 2>/dev/null; do sleep 1; done; echo $R >> ~/ov/queue.txt; rmdir ~/ov/.lock; ( [ -f ~/ov/sched_ai.pid ] && kill -0 \$(cat ~/ov/sched_ai.pid) 2>/dev/null ) || (setsid nohup bash ~/repro/frames/sched_ai.sh > /dev/null 2>&1 < /dev/null &)" >/dev/null 2>&1 || true
say "queued on p3-train-4g as $R; progress in ~/ov/$R/pipeline.log, results in ~/ov/$R/score.log (and curved_scores.json), the cut animation in ~/show/"
