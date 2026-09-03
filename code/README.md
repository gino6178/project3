# Two routes to an object with an interior

> The interior the paper reports no longer comes from this pipeline alone: it is fitted here and
> then re-sampled by `code/slicefill/`, which has its own README. Everything below builds the
> carrier that `slicefill` starts from, and its exterior, cut operator and evaluation are unchanged.


Everything here builds a voxel lattice, gives it an exterior, trains only its interior against
cross-sections, and scores held-out cuts. Seven objects run through it and the command is the
same for all of them.

```bash
git clone https://github.com/gino6178/project3.git && cd project3
bash code/setup.sh ./worktree         # lays code/ and data/ out as one tree
bash code/fetch.sh ./worktree         # the released reconstructions, too large for a git repo

bash code/bootstrap.sh ./build         # both interpreters and the two CUDA extensions

export FN_ROOT=$PWD/worktree
export GS_ROOT=$PWD/build/gaussian-splatting
export FN_PY=$PWD/build/mc/envs/fn/bin/python
export FN_PY_SCORE=$PWD/build/mc/envs/score/bin/python   # only code/evaluate needs this
export TRELLIS2_ROOT=$PWD/build/TRELLIS.2                # only the dual-grid exterior needs this

bash code/run.sh orange               # or: watermelon apple bread cake pomegranate doughnut
```

`run.sh` writes its lattice under `build_<object>/`, its model under `<object>/`, and its scores
under `eval_<object>/`, all inside `FN_ROOT`.

`run.sh OBJECT [stage]` where stage is `geometry`, `exterior`, `train`, `eval`, or `all`
(the default). Every stage skips itself when its output already exists, so a run resumes by
re-running the same command.

## The two routes

Which route an object takes is a property of its source file, not a flag anyone sets.

| `SRC` in the object's conf | where the exterior comes from |
|---|---|
| a `.ply` | the object was captured; its appearance came in with the model and is quantised onto the lattice |
| anything else | the shape was generated and carries no appearance, so the six references in `REFS6` are projected onto its surface by their own cameras |

Six objects take the first route: `apple`, `bread`, `cake`, `orange`, `pomegranate`,
`watermelon`, all quantised from the models FruitNinja released. The `doughnut` takes the
second: its topology is generated, and `skin_project.py` paints it from six views.

Both end in the same place — the exterior sitting on the lattice's surface cells — and the
trainer pins it there. Only the interior learns.

## What the references are put through before they supervise

A reference set is many photographs of one kind of fruit. Each was cut at its own angle and
each is a different fruit, so two of them disagree in two ways, and both are removed before
training rather than averaged away by it.

| | what disagrees | what is done |
|---|---|---|
| around the section | the cut angle each photograph happened to have | `stage_phases` solves the rotations and the assignment of photographs to depths together, equation (27), and writes the result beside the references. `sds_demo` falls back to the greedy per-family alignment of (11) when it is absent, so the stage is safe to skip. |
| across the section | where each fruit's own peel ended | nothing. Two ways of aligning it were written and measured and neither is general -- the numbers are in `src/section_match.py`, and both are off. |

The second looked like it should matter -- the peel runs from 3.9% to 19.5% of the radius
across the watermelon's twenty transverse photographs -- and measurement says it does not. Both
attempts made the references agree less, not more, and the orange got 17% worse. What these
references disagree about is colour, everywhere at once and by 0.07, because they are
photographs of different fruit; nothing radial reaches that. The first row works because the
cut angle really is the only thing wrong with a rotated photograph.

## What counts as the surface

One function, `occupancy.surface_cells`, used by the painter and again by the trainer that pins
what it painted. It seals the stored occupancy, takes the boundary of the solid two cells deep,
and unions that with the furthest cell along each of 26 directions.

It replaces `cell_level != 0`, which asked whether a cell sits beyond `skin_frac * R` of the
centroid. That is a sphere, and none of these objects is one: measured against the cells that
actually are outermost, it missed 0.2% of the watermelon's surface, 41.8% of the orange's, 57%
of the loaf's and 98.5% of the pomegranate's. Whatever it missed was never pinned, so the
cross-section supervision repainted it — the pale mottling that used to appear on peels.

The three constants below are the same for every object. Nothing here is tuned per object, and
no object conf sets any of them.

| variable | default | what it does |
|---|---|---|
| `SHELL_PIN_LAYERS` | 2 | how deep the pinned band goes, in cells |
| `POS_FREEZE` | 1 | holds interior cells on the lattice so the visibility test stays true |
| `ANCHOR_CAP_FIX` | 1 | drops a nearest-neighbour self-match by index rather than by distance |

## Layout

```
run.sh          the only entry point
stages.sh       lattice -> exterior -> train -> eval
objects/        one conf per object: source, spacing, configs, references, prompts
src/            every line of the pipeline that is ours
inherited/      utils, particle_filling, mpm_solver_warp, unmodified from PhysGaussian
```

`src/` holds seventeen files and nothing that is not on one of the two routes.

`inherited/` is PhysGaussian's solver and filling, vendored so that reproducing this
needs only a built `gaussian-splatting` beside it and not a second checkout.

## What the doughnut needs

Its references (`data/cube_donut3_prep`), its section photographs (`data/secref_dn2_*`) and its
physics config are in this repository. Its lattice is generated rather than captured, so it is
not: `lattices.tar` from the release carries the one every number here was measured on, and

```bash
$FN_PY code/src/make_shape.py torus INIT_dn5 --radius 0.55 --tube 0.18 --dx 0.00832
```

builds one from nothing but the shape.

## What is not here

`gaussian-splatting` supplies `scene` and `gaussian_renderer`, and it has to be **FruitNinja's
fork**, `fanguw/gaussian-splatting`: the pipeline calls `GaussianModel.load_ply_zero_sh`, which
is only there. That fork carries no submodules, so `diff_gaussian_rasterization` and `simple_knn`
are built from upstream, whose rasteriser API its renderer is written against. `bootstrap.sh`
does both, along with the parts that are not guessable -- torch 2.0.1 on **cu118**, because CUDA
11.7 cannot target sm_89 and an L40 is Ada, and `setuptools<70`, because the rasteriser's
`setup.py` imports `pkg_resources`. The released reconstructions are the six models FruitNinja
published and come from `fetch.sh`.

## Scoring

`stage_eval` renders twelve cuts at depths training never sampled and scores them against the
photographs with FID and CLIP. Objects with too few references for a distribution use
`SCORE=topology` instead, which asks what a section still *is* — every transverse cut of a torus
should read as an annulus.

To check an exterior against the model it was taken from:

```bash
$FN_PY src/exterior_views.py MODEL.ply CFG DEMO out.png 384
```

## Redrawing the animation at the top of the page

```bash
bash code/fetch.sh ./worktree            # trained_v3.tar: the seven models, 430 MB
bash code/figures/draw_cuts.sh orange orange.mp4
ffmpeg -framerate 14 -i orange_frames/%04d.png -c:v libx264 -crf 20 -pix_fmt yuv420p orange.mp4
```

`code/figures/` is not on either route — it draws a figure — and has its own README. It needs the
trained model and two small tensors per object, not the lattice's own point cloud, so redrawing
all seven is a 430 MB download rather than 1.1 GB.
