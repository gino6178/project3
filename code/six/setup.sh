#!/usr/bin/env bash
# Build a working tree from this repository, so the pipeline can run.
#
#   bash code/six/setup.sh /path/to/worktree
#
# The confs address their inputs relative to FN_ROOT -- `config/orange_physics.json`,
# `secref_orraw_hsep`, `data_finetune_images/...` -- so the repository's code/ and data/ have to be
# laid out as one tree before anything runs. This makes that tree out of symlinks, so it costs
# nothing and editing a file in the tree edits it in the repository.
#
# The released reconstructions are not in the repository: five of the six are 158 to 541 MB and
# GitHub refuses any file over 100 MB. Run six/fetch.sh to get them.
set -eu

HERE=$(cd "$(dirname "$0")/../.." && pwd)          # the repository root
DEST=${1:-$HERE/worktree}
mkdir -p "$DEST"

for p in "$HERE"/code/*; do
  b=$(basename "$p")
  [ "$b" = "site_tools" ] && continue
  ln -sfn "$p" "$DEST/$b"
done
for p in "$HERE"/data/*; do
  ln -sfn "$p" "$DEST/$(basename "$p")"
done

echo "worktree at $DEST"
echo "  code:  $(ls "$DEST" | grep -c . ) entries"
missing=0
for d in prefilled/trained_gs utils scene gaussian-splatting; do
  if [ -e "$DEST/$d" ]; then echo "  have   $d"; else echo "  MISSING $d"; missing=1; fi
done
if [ "$missing" = 1 ]; then
  echo
  echo "Fetch what is missing:"
  echo "  bash code/six/fetch.sh $DEST"
fi
echo
echo "Then:  export FN_ROOT=$DEST"
