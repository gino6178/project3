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
