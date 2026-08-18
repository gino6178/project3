# The animation at the top of the page

Three planes cut an object, the pieces slide apart along their own outward directions and back,
and the loop closes on itself. Nothing is tinted: a face that was always on the outside keeps the
model's own colour and a face the cut made shows what the volume behind it holds, which is the
whole claim the picture is there to support.

```bash
export FN_ROOT=$PWD/worktree GS_ROOT=... FN_PY=...
bash code/figures/draw_cuts.sh orange orange.mp4
ffmpeg -framerate 14 -i orange_frames/%04d.png -c:v libx264 -crf 20 -pix_fmt yuv420p orange.mp4
```

`MODEL=path/to.ply` overrides which model the appearance comes from; by default it is the one
`run.sh` writes. The seven files here are not on either route of the method — they draw a figure —
which is why they sit apart from `src/`.

## What it needs

The trained model, and `lattice.pt` plus `cell_level.pt` from the lattice it was trained on. It
does **not** need that lattice's own `gs_fill.ply`: the model is row-aligned with it, so positions
come from the model when the lattice's copy is absent. That is the difference between 375 MB and
1.1 GB for anyone who only wants to redraw these.

## Two things that were wrong here and are worth not repeating

**Faces took their colour by distance.** Every face looked up the nearest lattice point to the
face *centre*, which sits half a cell off the cell the face belongs to and can match an interior
one. The orange came out blotched white where peel picked up pith, the apple mottled where skin
and flesh swapped, and the doughnut fluorescent magenta. A face now reads the primitive in its own
cell, by index.

**The projection flipped y.** `ndc2Pix` in the Gaussian rasteriser is `((v + 1) S - 1) / 2` on
both axes; this file used `(1 - (v + 1)/2) S` on y alone, so every object it drew was a vertical
mirror of the same model rendered anywhere else. `skin_project.py` records the identical trap
costing four side views of every shell.
