# Cube Interior + O-Voxel Cut Surface

The hidden interior of a cuttable 3D asset is stored as a two-level structured cube lattice holding
occupancy and an appearance latent, rather than as opaque Gaussian primitives. Cells are subdivided
only where a cut passes, and only the newly exposed cross-section becomes an O-Voxel surface. The
interior is supervised directly by cross-section photographs — no per-object fine-tuning and no
score distillation.

**Page:** <https://gino6178.github.io/project3/> — `index.html` is the paper, `supplement.html` the
detail behind it. Numbers on the page are measured under one protocol on one machine and include
the results that did not work.

This repository holds three things and nothing else: the page, the code that produced it, and this
file.

```
index.html supplement.html assets/   the page. Serving it is a git push; there is no build step.
code/                               the method
README.md                           this
```

## What is in `code/`

```
code/method/run.sh                 one entry point, any object:  bash method/run.sh orange
code/method/objects/*.conf         one file per object. Adding an object is writing one of these.
code/method/common/cube/           the lattice: subdivision, cutting, dual grid, compositing
code/method/common/eval/           held-out cuts, DreamSim/FID/CLIP scoring
code/method/common/train/          the training loop's pieces
code/six/                          the six-object pipeline, five steps (its own README)
code/train_voxel.py                the trainer
code/sds_demo.py                   section supervision: photographs, warp, plane schedule
code/fid_eval.py code/clip_eval.py scoring
code/site_tools/                   scripts that build page assets, not part of the method
```

## Running it

The method is a stage on top of FruitNinja's released reconstructions and its renderer, which are
not vendored here. You need:

- [FruitNinja3DInterior](https://github.com/fanguw/FruitNinja3DInterior) and its
  `gaussian-splatting` submodule, for `utils/`, `scene/` and the rasteriser
- its released `.ply` reconstructions, in `prefilled/trained_gs/`
- one CUDA GPU. Peak device memory during training is 5.2 to 5.9 GB.

Put `code/` alongside that checkout so `method/`, `six/` and `train_voxel.py` sit at its root, then:

```bash
export FN_ROOT=/path/to/checkout
bash method/run.sh orange            # lattice, train, score
GPU=1 bash method/run.sh watermelon  # on the second card
bash method/run.sh orange eval       # re-score an existing model
```

Two interpreters are in play and mixing them is the most common way to lose an afternoon. The
renderer needs the CUDA rasteriser and runs on the project's own Python; the analysis scripts need
DreamSim, FID and CLIP and run on the system Python. Scripts that need one or the other launch it
themselves. When calling one by hand, clear the inherited environment:

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME python3 six/prep.py
```

### The six-object pipeline

`code/six/` takes all six released objects from `.ply` to the comparison table, in five steps that
run in order. `code/six/README.md` has the detail, including what `HELDOUT_BAND` does and does not
hold out.

```bash
python3 six/objects.py    # write method/objects/<name>.conf for each object
python3 six/prep.py       # put every reference on white, repoint the confs
bash    six/train.sh      # train all six, one per GPU
bash    six/eval.sh       # render both arms and score, one batch
python3 six/figure.py     # the comparison figure
```

Step 2 is not optional. Section supervision finds a reference's slice by thresholding against
white, so eight of the released photographs — shot on a dark or textured backdrop — read as one
frame-filling disc and align the render to the frame's radius rather than the slice's. Nothing
errors without it: the training log prints `dark 66.56%` and the run proceeds.

## Adding an object

Write `method/objects/<name>.conf`. Nothing in the code changes. The fields that differ between
objects are all parameters rather than branches: `SRC` (where the lattice comes from — a directory
holding one, a mesh, or a point cloud; the kind is read off the file), `COARSE_DX`, the reference
directories, and `SCORE` (`fid` for objects with several references, `topology` for objects with
too few).

## What the numbers were measured on

One machine, one protocol, one render batch per table. DreamSim drifts by up to 0.006 between
batches, so a value is comparable within its own table and the tables say which batch they came
from. The leave-one-out study is the one place something is genuinely unseen: it holds out a
photograph. `HELDOUT_BAND` holds out the plane, not the depth.
