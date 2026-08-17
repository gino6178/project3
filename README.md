# Cube Interior + O-Voxel Cut Surface

The hidden interior of a cuttable 3D asset is stored as a two-level structured cube lattice holding
occupancy and an appearance latent, rather than as opaque Gaussian primitives. Cells are subdivided
only where a cut passes, and only the newly exposed cross-section becomes an O-Voxel surface. The
interior is supervised directly by cross-section photographs — no per-object fine-tuning and no
score distillation.

**Page:** <https://gino6178.github.io/project3/> — `index.html` is the paper, `supplement.html` the
detail behind it. Numbers on the page are measured under one protocol on one machine and include
the results that did not work.

This repository holds the page, the code that produced it, the inputs that fit, and this file.

```
index.html supplement.html assets/   the page. Serving it is a git push; there is no build step.
code/                               the method
data/                               every input small enough to keep in a git repository
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

```bash
git clone https://github.com/gino6178/project3.git
cd project3
bash code/six/setup.sh ./worktree /path/to/FruitNinja3DInterior
bash code/six/fetch.sh ./worktree       # the binaries too large for a git repo
export FN_ROOT=$PWD/worktree
cd worktree
bash six/eval.sh                        # reproduces the six-object table
```

`data/` holds every input small enough to keep in the repository: the solver and demo configs, all
84 reference photographs, the whitened set, and the six exterior views. Nothing has to be hunted
down elsewhere.

`fetch.sh` gets what cannot be kept here. GitHub rejects any file over 100 MB on push and five of
the six released reconstructions are 158 to 541 MB, so they are release assets instead. It pulls
two archives by default:

- `released.tar` — the six reconstructions FruitNinja published, ~1.6 GB. Needed for the baseline
  column and, if you retrain, for the lattice.
- `trained.tar` — the six models this work trained, ~320 MB. With them `six/eval.sh` reproduces
  the table without retraining. Without them run `six/train.sh` first, which is hours on one card.

Verified: a clean clone, `setup.sh` against a FruitNinja checkout, those two archives and nothing
else reproduces every row of the table below.

`WANT="released trained lattices" bash code/six/fetch.sh` adds the quantised lattices, ~640 MB,
which `method/run.sh` would otherwise rebuild from the reconstructions in minutes per object.

The renderer and the MPM solver are FruitNinja's and are not vendored here, so `setup.sh` takes a
[FruitNinja3DInterior](https://github.com/fanguw/FruitNinja3DInterior) checkout as its second
argument and links in every top-level entry this repository does not already provide — `utils/`,
`scene/`, `mpm_solver_warp/`, `gaussian_renderer/`, the built `gaussian-splatting` rasteriser and
whatever else it carries. It reports what is still missing rather than failing later inside a
render. You also need one CUDA GPU; peak device memory during training is 5.2 to 5.9 GB.

Any object on its own:

```bash
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

## The six-object table

| object | this work | FruitNinja released | photographs to each other |
|---|---|---|---|
| orange | **0.0700** | 0.1290 | 0.0424 |
| watermelon | **0.1783** | 0.2328 | 0.0670 |
| apple | **0.2951** | 0.3823 | 0.1653 |
| pomegranate | **0.2508** | 0.2985 | 0.1324 |
| bread | **0.3685** | 0.4949 | 0.1873 |
| cake | 0.2816 | **0.2669** | 0.2677 |

DreamSim to the nearest reference, lower is better. Five of six, by 1.19x to 1.84x. The cake loses
by 0.015, to a baseline that on that object sits at the photographs' own agreement with each other.

## What the numbers were measured on

One machine, one protocol, one render batch per table. DreamSim drifts by up to 0.006 between
batches, so a value is comparable within its own table and the tables say which batch they came
from. The leave-one-out study is the one place something is genuinely unseen: it holds out a
photograph. `HELDOUT_BAND` holds out the plane, not the depth.
