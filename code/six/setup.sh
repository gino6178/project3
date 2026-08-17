#!/usr/bin/env bash
# Build a working tree from this repository plus a FruitNinja checkout.
#
#   bash code/six/setup.sh WORKTREE [FRUITNINJA_CHECKOUT]
#
# The confs address their inputs relative to FN_ROOT -- `config/orange_physics.json`,
# `secref_orraw_hsep`, `data_finetune_images/...` -- so code/ and data/ have to be laid out as one
# tree before anything runs. Everything here is a symlink, so it costs nothing and editing a file
# in the tree edits it in the repository.
#
# The renderer and the solver are FruitNinja's, not ours, and are not vendored: give the checkout
# as the second argument and every top-level entry this repository does not already provide is
# linked in. That is deliberately not a list -- the first attempt named four directories and the
# render died on a fifth, mpm_solver_warp, that the list had not thought of.
set -eu

HERE=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${1:-$HERE/worktree}
FN=${2:-${FN_CHECKOUT:-}}
mkdir -p "$DEST"

for p in "$HERE"/code/*; do
  b=$(basename "$p")
  [ "$b" = "site_tools" ] && continue
  ln -sfn "$p" "$DEST/$b"
done
for p in "$HERE"/data/*; do
  ln -sfn "$p" "$DEST/$(basename "$p")"
done
ours=$(ls "$DEST")

if [ -n "$FN" ]; then
  [ -d "$FN" ] || { echo "no such checkout: $FN"; exit 1; }
  n=0
  for p in "$FN"/* "$FN"/.??*; do
    [ -e "$p" ] || continue
    b=$(basename "$p")
    case " $ours " in *" $b "*) continue;; esac
    case "$b" in .git|.gitmodules|worktree) continue;; esac
    ln -sfn "$p" "$DEST/$b"
    n=$((n + 1))
  done
  echo "linked $n entries from $FN"
else
  echo "no FruitNinja checkout given; the renderer and solver will be missing"
  echo "  usage: bash code/six/setup.sh $DEST /path/to/FruitNinja3DInterior"
fi

echo "worktree at $DEST"
missing=0
# What the render actually resolves at import time. gaussian_renderer is not among them: it lives
# inside the gaussian-splatting build, and naming it here reported a failure on a tree that ran.
# `scene` is the same and was on this list anyway: every script appends `$FN_ROOT/gaussian-splatting`
# to sys.path, and `scene/` is a directory of that build, so `from scene.gaussian_model import ...`
# resolves whether or not anything links it at the top level. A FruitNinja3DInterior checkout has
# no top-level scene/, so the list was telling anyone who followed the README that a working tree
# was broken and sending them to fetch.sh, which does not carry it either. The development tree
# this was written on happened to have a hand-made scene -> gaussian-splatting/scene link, which
# is exactly why the false alarm was invisible here.
#
# particle_filling is FruitNinja's and is checked here because `train_voxel.py` imports it at
# module level, on line 25, before anything is parsed -- so without it the trainer dies at import
# with no clue as to why. It used to be a symlink committed into code/ pointing at an absolute
# path on the machine this was developed on, which was worse than missing: the name then existed
# in `ours` above, so the loop below skipped it and the real one in the checkout was never linked.
for d in prefilled/trained_gs utils mpm_solver_warp particle_filling gaussian-splatting; do
  if [ -e "$DEST/$d" ]; then echo "  have    $d"; else echo "  MISSING $d"; missing=1; fi
done
[ "$missing" = 1 ] && { echo; echo "Fetch the reconstructions:  bash code/six/fetch.sh $DEST"; }
echo
echo "Then:  export FN_ROOT=$DEST"
