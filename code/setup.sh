#!/usr/bin/env bash
# Lay code/ and data/ out as one tree, which is what FN_ROOT has to be.
#
#   bash code/setup.sh WORKTREE
#
# The confs address their inputs relative to FN_ROOT -- `config/orange_physics.json`,
# `secref_orraw_hsep`, `data_finetune_images/...` -- so the inputs have to sit beside each other
# before anything runs. Everything here is a symlink, so it costs nothing and editing a file in
# the tree edits it in the repository.
#
# Unlike the version this replaces, no second checkout is linked in: the solver and the filling
# are vendored under code/inherited, and the only outside dependency left is a built
# gaussian-splatting, which GS_ROOT points at.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$HERE/worktree}
mkdir -p "$DEST"
for p in "$HERE"/data/*; do ln -sfn "$p" "$DEST/$(basename "$p")"; done
echo "linked $(ls "$DEST" | wc -l) inputs into $DEST"
echo
echo "next:"
echo "  bash code/fetch.sh $DEST      # the released reconstructions"
echo "  export FN_ROOT=$(cd "$DEST" && pwd)"
echo "  export GS_ROOT=/path/to/gaussian-splatting"
echo "  export FN_PY=/path/to/python"
echo "  bash code/run.sh orange"
