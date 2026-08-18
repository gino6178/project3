#!/usr/bin/env bash
# Download the binaries that cannot live in a git repository.
#
#   bash code/six/fetch.sh /path/to/worktree
#
# GitHub rejects any file over 100 MB on push, and five of the six released reconstructions are
# 158 to 541 MB. They are published as release assets instead, which have a 2 GB limit and do not
# enter the clone. Everything else the pipeline needs is in the repository already.
#
#   released.tar        the six reconstructions FruitNinja published, ~1.6 GB
#   trained.tar         the six models this work trained, ~320 MB. Optional: with them
#                       six/eval.sh reproduces the table without retraining, which is hours.
#   lattices.tar        the six quantised lattices, ~640 MB. Optional: method/run.sh rebuilds
#                       them from the reconstructions, which is minutes per object.
set -eu

REPO=${REPO:-gino6178/project3}
TAG=${TAG:-v1-inputs}
DEST=${1:-$(cd "$(dirname "$0")/../.." && pwd)/worktree}
WANT=${WANT:-released trained}

mkdir -p "$DEST"
cd "$DEST"
for a in $WANT; do
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
