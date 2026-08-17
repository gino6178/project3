# The method, and only what it needs

Everything here produced the published models.

Paths inside the scripts resolve against `FN_ROOT`, which is the worktree `six/setup.sh` builds —
this repository's `code/` and `data/` laid out as one tree beside a FruitNinja checkout. They used
to be absolute and point at one machine; each such path was written three times per script, two
`sys.path` entries and a `chdir`, and every one of them silently pinned the file. `FN_ROOT`
defaults to the development machine's path so nothing changes there, and setting it is the whole
of what porting a script now costs.

The shared machinery they import — `utils/`, `scene/`, `mpm_solver_warp/`, `particle_filling/`,
`gaussian-splatting/` — is FruitNinja's and arrives through that checkout, as do the released
reconstructions. The configurations in `config/` and the references in `secref_*/` and
`refs_white/` are this repository's `data/`.

The trainer and the five modules it imports live at the top of `code/`, not under this folder:
`stages.sh` `cd`s to `FN_ROOT` and calls `train_voxel.py`, `fid_eval.py` and `clip_eval.py` by
bare name. There were second copies of all eight under `common/train/` and `common/eval/`, and two
of them had fallen a revision behind the files that actually run; they are gone.

## Running it

One path, any object:

```
bash method/run.sh orange              # ~1 hour
bash method/run.sh watermelon          # ~1.5 hours
bash method/run.sh doughnut            # its lattice is not in this repository -- see below
GPU=1 bash method/run.sh orange        # on the second card
bash method/run.sh orange eval         # just re-score a model that already exists
```

**Adding an object is writing `objects/<name>.conf`.** Nothing else changes. Seventeen exist and
they differ only in that file — the six the table reports, the doughnut, and the ten fold and
ablation variants of the orange that sections 4.1.2 and 4.1.5 measure. The three the method README
was first written around:

| | orange | watermelon | doughnut |
|---|---|---|---|
| `SRC` | released model | released model | a lattice |
| `SCORE` | `fid` | `fid` | `topology` |
| `ITERS` | 200 | 200 | 200 |

Two kinds of conf here will not run from a fresh clone, and both are published anyway because the
paper's numbers are read off them. The ten variants of the orange point at reference directories
that are not in this repository — `fold0_htr`, `fewn2_h` and their siblings are subsets and
perturbations of `secref_orraw_*sep`, built for one study each. And the doughnut's `SRC=INIT_dn5`
is a lattice whose source mesh is no longer on disk, so it is neither in the repository nor in a
release asset. Each stops at the missing input rather than producing a wrong number.

Everything runs on a lattice. `SRC` says where that lattice comes from, and **its kind is read
off the file rather than chosen**: a directory that already holds one is used as it is, a mesh
goes through `mesh_to_voxel.py`, a point cloud through `voxelize.py`. There is no mode.

There used to be a second geometry route — recover the surface from the released model, re-run
the paper's fill, then quantise — because quantising the released watermelon directly left holes
in its skin, which covers only 84.5% of its own silhouette. That was measured on a lattice with
no skin refinement. Refined, direct quantisation covers **97.9%**, against the recover-and-refill
route's 97.0%, with a solid core either way. So the fill is gone, and with it the slowest stage
and the only one that fails silently: a hollow fill trains to a white hole through the middle and
scores FID 693.

`SCORE` says what the object can be judged on. `fid` is held-out cuts against the photographs.
`topology` is for objects with too few references for a distribution metric: the doughnut, which
has one image per family and is here to show that every held-out transverse section still reads as
an annulus, and the cake, whose three photographs are all longitudinal. Note that `six/eval.sh`
ignores `SCORE` entirely — it scores every object with DreamSim, which is a distance between pairs
and does not need a distribution. `SCORE` only decides what `method/run.sh <object> eval` does.

Every stage skips itself if its output is already there, so re-running resumes rather than
restarts. Intermediates go to `build_<object>/`, the model to `<object>/`, the held-out renders
to `eval_<object>/`.

## The stages, in order

| step | script | what it does |
|---|---|---|
| 1 | `common/pipeline/voxelize.py` or `mesh_to_voxel.py` | Make the two-level lattice. `refine` **must be 2**: the published models were built at 1 — coarse and fine spacing equal, no refinement at all — and that alone cost the watermelon 43 points of FID. |
| 2 | `train_voxel.py` | Training. Run from `FN_ROOT`, which is where `stages.sh` calls it from. |
| 3 | `common/eval/random_cuts.py` → `fid_eval.py` | Held-out cuts at depths training never sampled, scored against the photographs. |

Three stages, not five. `extract_shell.py`, `internal_filling.py` and `close_shell.py` are kept
in `common/pipeline/` because they are what the published models were built with and the paper
describes them, but no object calls them any more.

## Two things in the evaluation that were wrong for a long time

**Load the whole model.** `load_ply_zero_sh` reads the degree-zero colour and discards higher
spherical-harmonic bands. The watermelon lattice carries 24 of them; the released models carry
none. Measured with them discarded it scored FID 278.0, and whole, 260.6. `eval/random_cuts.py`
loads at the degree the file actually has.

**Filter the reference directory.** The trainer and the DreamBooth script cache depth maps beside
the photographs they were computed from, so `*.png` in a reference directory is half greyscale
depth. `fid_eval` has always filtered them by name; the figure script did not, and published a
"Photographs" row with two depth maps in it.

## Where the models ended up

| | ours | FruitNinja released | our cells |
|---|---|---|---|
| Orange | **FID 79.1** | 102.6 | 877,495 against 3,928,330 |
| Watermelon | FID 195.5 | **190.0** | 1,520,021 against 7,959,789 |
| Doughnut | 6/6 held-out sections report a hole | no released model | — |

On the orange the refinement costs nothing — 877,495 cells against the 898,563 the unrefined
lattice used, because a coarser interior pays for the finer skin. On the watermelon it costs
cells and closes a 43-point gap.

## What was tried on the watermelon and did not work

Recorded because each is a plausible idea that costs an hour to re-test. All of them left the
transverse FID between 238 and 266, against the 195.5 that refining the skin produced.

| | FID |
|---|---|
| regenerating the references from the model's own render, once the mask bug below was fixed | 260.6 |
| capping the primitive size at half, then a quarter, of the cell spacing | 240.2, 255.5 |
| rendering the section from a thinner slab | 266.4, 281.0 |
| estimating the map's radii robustly instead of smoothing them | 251.3 |
| giving the appearance a view direction and letting the sections supervise the shell | 266.0 |

Two defects found along the way are fixed in `section_match.py` and are worth knowing about,
because both were invisible in the configuration and both only ever hurt *generated* references:

* the flood fill that finds a reference's silhouette had no `FIXED_RANGE`, so its tolerance was a
  step size and it walked the soft edge of a generated image into the fruit — a mask of 842 pixels
  for a disc 223 across, whose radii were then stretched over the whole section;
* the per-ray radii were estimated on 1440 bins, under a pixel of arc each, so the estimate was
  pixel noise; it is now estimated on 90 bins and interpolated back.

## Two probes that were built, failed, and should not be rebuilt

Both looked convincing on the object they were developed on and inverted on the next one.

* **Shell cells inside a 3-D slab.** Estimates the radius from a random subsample, then counts
  inside a slab a few percent thick. Three lattices of the same family reported 7,796, 3,090 and
  455.
* **Rind-to-interior colour separation in the rendered rim.** Ordered the watermelon exactly as FID
  did, and on the orange put our winning model last and the model with a hole through its middle
  near the references. The script is not in this repository; it is described rather than shipped
  precisely because it should not be rebuilt.

The only trustworthy signals here are held-out FID/KID against the photographs, and the image.
