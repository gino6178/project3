#!/usr/bin/env bash
# Download the binaries that cannot live in a git repository.
#
#   bash code/fetch.sh /path/to/worktree
#
# GitHub rejects any file over 100 MB on push, and five of the six released reconstructions are
# 158 to 541 MB. They are published as release assets instead, which have a 2 GB limit and do not
# enter the clone. Everything else the pipeline needs is in the repository already.
#
#   released.tar        the six reconstructions FruitNinja published, ~1.6 GB. Needed to build
#                       a lattice from scratch; run.sh quantises them.
#   trained_v3.tar      the seven models this work trained, ~430 MB, and the two small tensors
#                       per object that say what lattice each was trained on. With these,
#                       `run.sh OBJECT eval` scores without retraining and
#                       `figures/draw_cuts.sh` redraws the animation at the top of the page.
#                       These are the models the page shows.
#   trained_v2.tar      superseded: the same models on a lattice whose levels came from a
#                       radius rather than from the occupancy.
#   trained.tar         superseded: before the exterior was fixed at all. Both are kept so the
#                       earlier numbers on the page can still be checked against the models
#                       that produced them.
#   gfluent             the GaussianFluent watermelon and cake, from the authors'
#                       Hugging Face release rather than from here. Needed only by
#                       code/evaluate/compare.py, and not fetched by default.
#   lattices.tar        the six quantised lattices, ~640 MB. Only needed to skip the minutes
#                       run.sh spends rebuilding them; trained_v2 carries the metadata alone.
set -eu

REPO=${REPO:-gino6178/project3}
TAG=${TAG:-v1-inputs}
DEST=${1:-$(cd "$(dirname "$0")/../.." && pwd)/worktree}
WANT=${WANT:-released trained_v3}

# Absolute, before anything cds: the untar below names $DEST as its target directory *after*
# this cd, so a relative DEST -- which is what the README's own example passes -- resolves
# against the wrong place and tar exits with "worktree: Cannot open".
DEST=$(mkdir -p "$DEST" && cd "$DEST" && pwd)
cd "$DEST"
for a in $WANT; do
  # Not ours to redistribute: the comparison arms come from the GaussianFluent authors' own
  # release. Two of the eighteen objects they publish are also ours -- the watermelon and the
  # cake -- and code/evaluate/arms.json addresses them here. arms.json used to name an absolute
  # /workspace path that existed on one machine, which is no comparison anyone else could run.
  if [ "$a" = gfluent ]; then
    B=https://huggingface.co/hbpencil01/GaussianFluent/resolve/main
    for o in watermelon cake; do
      p="gfluent/model/$o/point_cloud/iteration_30000/point_cloud.ply"
      [ -s "$DEST/$p" ] && { echo "== gfluent/$o already here"; continue; }
      echo "== gfluent/$o"
      mkdir -p "$DEST/$(dirname "$p")"
      curl -fL --progress-bar "$B/model/$o/point_cloud/iteration_30000/point_cloud.ply" \
           -o "$DEST/$p"
    done
    continue
  fi
  url="https://github.com/$REPO/releases/download/$TAG/$a.tar"
  echo "== $a"
  if command -v curl >/dev/null; then
    curl -fL --progress-bar "$url" -o "/tmp/$a.tar"
  else
    wget -q --show-progress "$url" -O "/tmp/$a.tar"
  fi
  tar xf "/tmp/$a.tar" -C "$DEST"
  rm -f "/tmp/$a.tar"
done
echo
echo "in $DEST:"
for d in prefilled/trained_gs orange watermelon apple pomegranate bread cake; do
  [ -e "$DEST/$d" ] && echo "  have $d" || echo "  still missing $d"
done
