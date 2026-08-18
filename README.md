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

One entry point, four stages, and the seven objects the page reports. Nothing that is not on one
of the two routes is in here.

```
code/run.sh              bash code/run.sh orange   -- and the same for every other object
code/stages.sh           lattice -> exterior -> train -> eval
code/objects/*.conf      one file per object: source, spacing, configs, references, prompts
code/src/                every line of the method, seventeen files
code/inherited/          PhysGaussian's solver and filling, vendored unmodified
code/setup.sh            lays code/ and data/ out as the one tree FN_ROOT has to be
code/fetch.sh            the released reconstructions, too large for a git repository
code/README.md           the two routes, and what each constant does
```

Which route an object takes is a property of its source, not a flag: a `.ply` brings its own
appearance, and anything else has six references projected onto it. Six objects take the first,
the doughnut the second.

## Running it

```bash
git clone https://github.com/gino6178/project3.git && cd project3
bash code/setup.sh ./worktree
bash code/fetch.sh ./worktree

export FN_ROOT=$PWD/worktree
export GS_ROOT=/path/to/gaussian-splatting   # built, with diff_gaussian_rasterization
export FN_PY=/path/to/python                 # torch, taichi, warp

bash code/run.sh orange
```

Stages skip themselves when their output exists, so a run resumes by repeating the command, and
`bash code/run.sh orange eval` scores a model that is already trained.

## The six-object pipeline

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

## A correction to the commit of 18 Aug

That commit reported an interior FID for every object, including
`pomegranate FID 318.3 -> 124.1`. The exterior figures beside them stand: they
are per-pixel distances over 240,000 to 640,000 pixels of the object.

The interior figures do not, and the pomegranate's least of all. It has three
reference photographs. FID estimates a covariance in 2048 dimensions, which
three samples cannot support, and KID reported its own uncertainty as +-75.3 on
a value of 247.7. Re-scoring the same model a second time moved it by more than
the difference the commit attributed to the change. References per object:
watermelon 20, apple 8, orange 6, bread 5, pomegranate 3, cake 3, doughnut 1.

This is not a new limitation -- section 2.4 states that both distribution
metrics run "at sample sizes far below where they are reliable", and the results
use DreamSim and a manifold precision/recall for exactly that reason. The error
was mine, in reaching for FID because it was the number `stages.sh` prints.

What the exterior change is supported by, and what it is not:

  the exterior          per-pixel distance to each object's own ply render, and
                        counts of stray bright specks. 5,471 specks to 35.
  the interior          reference-free measures only, where the reference set is
                        too small: slicing consistency, which needs no photograph
                        at all.
