# Every number, from the models on disk

```bash
export FN_ROOT=$PWD/worktree GS_ROOT=... FN_PY=... FN_PY_SCORE=...
bash code/evaluate/run_all.sh          # this work, seven objects
$FN_PY  code/evaluate/compare.py       # the released models, same cuts, same tools
$FN_PY_SCORE code/evaluate/matrix.py   # every object's renders against every object's references
```

Results land in `$FN_ROOT/measurements`: the verbatim stdout of every tool under
`<object>/<tool>.log`, and `results.json`, `compare.json`, `matrix.json` holding what was parsed
out of them. The logs are the record and the JSON is the convenience. A measurement that has run
is skipped, so this resumes; `--force` re-runs.

Two interpreters, because no single one has everything. `FN_PY` renders and needs the CUDA
rasteriser. `FN_PY_SCORE` only reads images and needs DreamSim, which the render environment does
not have; leave it unset and those rows are skipped and say so.

## Three things this harness got wrong once, and now cannot

**The cut has to go through the object.** `random_cuts.py` with no `HELDOUT_BAND` puts the plane
at 0.04–0.15 or 0.85–0.96 of the axis — the two ends — and what comes back is very nearly the
exterior. Scored against cross-section photographs that reads as a DreamSim of 0.40 where the
middle of the object reads 0.07, and the mistake is invisible in a number and obvious in a
picture. `HELDOUT_BAND=0.30,0.70` and `FULL_SH=1` are set here, as `stages.sh` sets them. **Look
at the cuts before believing any number computed from them.**

**FID needs a distribution.** It estimates a covariance in 2048 dimensions; six references cannot
support one and three certainly cannot. Below ten references this harness does not compute it and
records why. DreamSim is pairwise and survives the sample size, which is why it is the column to
read.

**A number is only comparable to one measured the same way.** `compare.py` re-renders the
released models through the same `random_cuts` at the same depths and scores them with the same
tools, in the same run. Scoring one arm now and quoting another from an older run puts two
pipelines in one table.

## What each object trains on, and what it is scored against

This is a property of the object's conf and it decides how a number should be read.

| object | train h / v | scored against | is the score a fit or a generalisation |
|---|---|---|---|
| orange | 6 / 6 | the same 6 | a fit |
| watermelon | 20 / 23 | the same 20 | a fit |
| apple | 8 / 3 | the same 8 | a fit |
| bread | 5 / 5 | the same 5 | a fit |
| cake | 3 / 3 | the same 3 | a fit |
| pomegranate | 3 / 5 | the same 3 | a fit |
| doughnut | 1 / 1 | the same 1 | a fit |

Every one of them scores a model against the images it was trained on. None of these numbers is
evidence of generalisation, and a table that does not say so invites the opposite reading. The
watermelon used to be the exception — one reference in, twenty out, nineteen never seen — and
that arm is preserved as `trained_v2.tar`'s watermelon so the one held-out result this project
had can still be checked.

## ovox_cuts draws half a transverse section

Measured 2026-08-19 on the orange, twelve held-out planes, same depths and camera as the Gaussian
renderer: the longitudinal six agree with it to within 2 to 6% of silhouette area and DreamSim
0.04 to 0.09 between the two pictures, and the transverse six lose 22% to 86% of the area. Since
`realism.py` globs `rh*` only, a DreamSim computed on this tool's output is a measurement of the
tool.

Excluded already: the exterior layer (`OVOX_LAYERS=face,ext` moves the area 0.153 to 0.154, so
the missing half is not the peel), and resolution (2048 and a 512 downsample agree to 0.0004).
The occupancy lookup is axis-agnostic. What is left is the ray-plane construction near the
equator.

Do not quote a figure from this file as a property of the dual grid until that is closed.

## The four tools section 4 gained, and what each answers

None of these is in `measure.py`'s battery: each answers one claim, on every object at once, and
is run on its own.

```bash
$FN_PY       code/evaluate/cutface.py orange=build_orange/lattice watermelon=... apple=... \
                                      bread=... cake=... doughnut=...
$FN_PY       code/figures/bandfrac.py assets/bandfrac.png orange=build_orange/lattice ...
$FN_PY       code/evaluate/detail.py "the photographs=secref_orraw_hsep" "orange=<cuts>" ...
$FN_PY       code/evaluate/material_segment.py MODEL.ply LATTICE OUT.png 4 ANCHOR.pt
$FN_PY       code/evaluate/material.py LATTICE MODEL.ply CFG OUT_labels.npy DEMO OUT.png
```

`cutface.py` drives `figures/cutmesh.py` itself, so the polygon counts and the geometry cannot
come apart, and it prints both band counts side by side. **They differ by a factor of four and the
difference is not an error.** `crossed` returns the cells of the stored lattice that the plane
straddles; the operator then refines every one of them, and the leaves that carry a polygon are
four times as many. A figure of 4.0 to 10.7% was carried for this and matches neither: on the six
objects the first is 0.51 to 3.03% and the second 2.04 to 12.12%.

`bandfrac.py` coarsens each object's own occupancy to get four resolutions of one shape, which is
what turns that fraction into a scaling result rather than a property of six fruit.

`material.py`'s class table and stiffness ordering are equation (13). Its resolvability verdict is
computed against the particle-filling grid and section 4.2.1 of the page uses the transfer grid,
which is coarser; the file says so where it prints it. Read the ordering, not the verdict.

## Two ways a sweep over depth measures the harness

Both were hit, both are cheap to hit again.

**The cut view quantises the plane.** `random_cuts.build_renderer` indexes a fixed list of
`n_depth` plane centres, 24 by default, so any sweep denser than that list returns the same image
several times and then jumps. Driving it through `random_cuts.main` and `HELDOUT_BAND`, two
different models came back with 38 of 47 steps at exactly zero and the same nine spikes in the
same places — a step function of the indexing, identical for both, and it would have read as a
result. Use `random_cuts.sweep`, which takes `n_depth`; `figures/depthassign.py` passes four times
the sweep density.

**A per-step ratio against the median divides by nothing.** Under the block assignment most steps
are exactly zero, so `worst / median` is whatever the epsilon was. The count of steps below 1e-6
says the same thing and survives.
