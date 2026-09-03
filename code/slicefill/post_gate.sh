#!/bin/bash
SP=/tmp/claude-1000/-home-gino-project-FruitNinja-clean/e5030f17-1844-4a79-8a91-3194645ab588/scratchpad
PY=/home/gino/miniconda3/envs/fruitninja/bin/python
cd $SP
while ! grep -q POST_DONE post_all.log; do sleep 60; done
echo "object       gate    supervised O-Voxel long/trans -> ours"
for o in apple bread cake doughnut pomegranate watermelon; do
  env OBJDIR=$SP/obj/$o scoreenv3/bin/python dssup.py 2>&1 | grep -v Warning > obj/$o/sup.log
  g=$($PY gate.py obj/$o/sup.log); echo "$o: $g"
  case "$g" in FITTED*) cp wt/trained/$o.ply obj/$o/${o}_asset.ply;; *) cp obj/$o/${o}_x3d.ply obj/$o/${o}_asset.ply;; esac
done
echo GATE_DONE
