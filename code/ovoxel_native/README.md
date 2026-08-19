# O-Voxel-native interior

The representation is an O-Voxel dual grid and nothing else: two per-cell feature tensors decoded
by two small MLPs, plus the dual grid's own geometry. No opacity, no covariance, no spherical
harmonics; rendering is nvdiffrast only. Everything else -- initialisation, schedule, references,
losses -- is `project3/code/stages.sh`'s `stage_train`, carried across.

## One program, three variables

    ROUTE=1|2       1 = the released ply quantised; 2 = shape from make_shape.py's ellipsoid SDF
                    and exterior from skin_project.py's six-view projection
    FLAT_INIT=0.5   the interior does not start from the released model's colours
    SHELL_PIN=0|1   whether the exterior is trained

## Files

    ovnative.py     the representation: occupancy -> dual grid -> interior field, the closed-form
                    cut, and the two renderers (section and exterior)
    anchor.py       the anchor decoder's colour path and prefit, plus voxel_smooth_anchors
    secloss.py      the section loss stage_train trains on: 0.7(1-SSIM)+0.3MSE on SEC_PATCH crops
                    plus the band term
    refsel.py       which photograph supervises which plane: equation (27) solved, equation (11)
                    greedy as the fallback, equation (14) for the longitudinal family
    mvcams.py       every camera and plane, from the pipeline's own machinery
    mvtrain.py      the training loop
    build_state.py  build the state for a route
    evalmv3.py      DreamSim on both families, from evaluate/realism.py
    figmv3.py       the held-out sheets and crops
    streaks*.py     what the oriented structure in a cut face is
    extviews.py     each arm's exterior, from the six cameras that supervised it
    figext.py       the exterior sheet

State is rebuilt from the cached `state_r1.pt` / `state_r2.pt` (built 03:51 from
`build_orange/lattice` and `build_orange_r2/skin`) plus each arm's `params.pt` -- NOT from
`build_orange/lattice` as it now stands, which another tenant's rerun.sh rebuilt at 05:15 with a
flat interior.

Logs and per-arm `eval_final` renders are beside the code; `out/` holds the figures.

## The grid blocking, and what reduces it

A cut through a per-cell field on a cubic lattice shows the lattice, and how much depends on the
cut's orientation. Transverse cuts show thin dark chords whose 3-D directions are low-order
rational directions of the coarse grid -- (1,0,-1), (2,0,-1), (1,0,2), the same three in every arm
including route 2, which shares no data with route 1. Longitudinal cuts run along the axes and show
whole cell faces, which is the blocking visible in `out/route1_longitudinal.png`.

It is not a rendering artefact: a constant field with no exterior and no antialias renders exactly
flat (residual rms 0.00000), the effect is resolution-invariant, and it is not the supervision --
the ten longitudinal planes' intersections with a transverse cut carry 0.80x the frame's average
change, below it, not above.

**SEC_PATCH's band term is what reduces it.** That term compares low frequencies where they sit and
finer octaves only in quantity, so no single view can impose its own pattern on a cell -- which is
exactly the failure mode here. Measured: longitudinal DreamSim 0.2535 -> 0.2230 with SEC_PATCH on,
the largest single improvement in that column, and `full parity` is visibly the least blocky row of
the longitudinal sheet. It costs transverse DreamSim (0.0504 -> 0.0609), so it is a trade and not a
free win.

## The exterior

`out/exterior_views.png` draws every arm from the six cameras that supervised it (`e_mvp` in
cams_mv.npz). Rows marked (O-Voxel) are the dual surface through nvdiffrast; the two rows marked
(Gaussian) are the pipeline's own exterior through `project3/code/src/exterior_views.py`. Two
renderers, labelled, not blended.

DreamSim of each exterior against the six reference views:

| exterior | DreamSim |
|---|---|
| route 1, as built from the released ply | 0.2974 |
| r1_pin, r1_pin_full (SHELL_PIN=1) | 0.2974 |
| r1_free, exterior trained | **0.0880** |
| route 2, projected from six photographs | 0.1866 |
| r2_pin_full (SHELL_PIN=1) | 0.1866 |
| r2_free, exterior trained | 0.1241 |
| existing pipeline, route 1 (orange_b, Gaussian) | 0.3053 |
| existing pipeline, route 2 (orange_r2, Gaussian) | 0.1952 |

Three things follow. SHELL_PIN is exact -- a pinned arm scores its seed's number to four decimal
places, and `dual_v` and `split_w` come back bit-identical to what went in. Route 2's exterior is
far closer to the photographs it was projected from than route 1's is to photographs it never saw,
and the pipeline shows the same split (0.1952 against 0.3053). And SHELL_PIN has opposite signs on
the two measurements: pinning gives the better transverse section score (0.0504 against 0.0550) and
a 3.4x worse exterior.

## The reconstruction filter: slab against plane

The pipeline's cut face is a slab and this one was a plane. `train_voxel.py:2007` and
`random_cuts.py` both call `plane_filter(..., surf_dis=avg_dis/2, include_double=True)` and splat
every primitive in the band, so what they draw is an integral over depth; `cut_polygons` +
`sample_interior` is a trilinear sample on a zero-thickness plane.

Measured on this orange rather than assumed. `avg` = 0.04348 in the transformed frame the plane is
stated in, so `surf_dis` = 0.02174 there -- but that frame is not the lattice's. The
transformed-to-lattice scale is **1.5281**, so `surf_dis` is 0.03322 in lattice units = **2.82
coarse cells**, and the slab is **5.63 cells thick**, not the 3.68 that comparing the transformed
value against the lattice cell size gives.

`render_section` now takes a `thickness`: M sub-planes over +-surf_dis, uniform weight. Uniform is
defensible here rather than merely simple -- OPACITY_FREEZE holds interior opacity at 1.0 and
SCALE_FREEZE holds every footprint the same size -- and it is still an approximation of a
front-to-back composite.

### Both directions

O-Voxel arms, rendered at both filters (they were *trained* at zero thickness; only the drawing
changes):

| arm | filter | DS rh | DS rv | probe L1 |
|---|---|---|---|---|
| r1_pin | plane | 0.0504 | 0.2535 | 0.03197 |
| r1_pin | slab | 0.0527 | 0.2292 | 0.02929 |
| r1_pin_full | plane | 0.0599 | 0.2315 | 0.03081 |
| r1_pin_full | slab | 0.0613 | **0.1945** | 0.02848 |
| r1flat_pin | plane | 0.0504 | 0.2808 | 0.03235 |
| r1flat_pin | slab | 0.0524 | 0.2475 | 0.02943 |
| r2_pin_full | plane | 0.1018 | 0.2731 | 0.02621 |
| r2_pin_full | slab | 0.0993 | 0.2357 | 0.02448 |

The pipeline's own model (`baseline/orange_b.ply`, copied 06:05 before rerun.sh could touch it),
with `surf_dis` scaled by rebinding `plane_filter` in `random_cuts`'s namespace -- the plane
sequence, seed, band, camera and rasteriser are all the repository's:

| x avg/2 | cells | primitives in the band | DS rh | DS rv |
|---|---|---|---|---|
| 1 (as shipped) | 2.82 | 68,861 | 0.0707 | 0.2171 |
| 0.5 | 1.41 | 33,604 | 0.0780 | 0.2053 |
| 0.25 | 0.70 | 18,566 | 0.0803 | 0.2035 |
| 0.125 | 0.35 | 8,315 | 0.2455 | 0.2948 |
| 0.0625 | 0.18 | 4,270 | 0.4670 | 0.4104 |

A point cloud cannot render a thin plane: below about 0.7 cells the band stops containing a
primitive in every column and the render degenerates into a stipple, which is what the last two
rows are. The floor is a fact about the representation, not a setting.

### What follows

**The slab is not what hides the blocking.** `out/slab_longitudinal.png` shows the same
longitudinal cut at both filters: `r1_pin` blocks up at 0 cells and still blocks up at 5.6, so
integrating over four cells does not wash it out. The rv metric does improve (0.2535 -> 0.2292),
but far less than the hypothesis predicts, and the picture is unambiguous.

**The pipeline does not block up when the slab is removed.** Its rv *improves* as the slab thins,
0.2171 -> 0.2053 -> 0.2035, right down to the sampling floor. So the blocking is a genuine property
of this representation and not something the pipeline's filter was hiding on its own side.

**Part of the pipeline's transverse advantage is the filter.** Its rh degrades 0.0707 -> 0.0803 as
the slab thins, while every O-Voxel arm beats it at either filter.

**At a matched filter, `r1_pin_full` wins both columns**: rh 0.0613 against 0.0707, rv 0.1945
against 0.2171. That is the first time anything here has beaten the pipeline longitudinally, and
what makes it possible is SEC_PATCH's band term plus the slab, not the slab alone.

## The spherical harmonics could not be costed

`FULL_SH=0` is a no-op on every model on this box. Checked with plyfile: `orange`, `orange_b`,
`orange_r2`, `orange_f0..f5`, `watermelon`, `apple`, `bread`, `pomegranate_r2`, `abl_*` -- **every
one carries f_rest 0**. `random_cuts.main` only takes the higher-band branch `if FULL_SH and _n`,
so with `_n = 0` both settings load the same degree-0 model. The comparison against this
representation, which has no view-dependent term, is therefore already like-for-like on these
models, and the 24-coefficient figure in `random_cuts.py`'s comment does not describe anything
currently on disk.

## Cross-family reconciliation: measured, and it does not work

`SEC_XCONS` reconciles the two families' section targets along the lines their planes share. It
is off in the pipeline, set nowhere, and the page reports only its averaging mode, which failed.
`mode="copy"` is documented in `section_consistency.py` as the answer to that failure -- one
family wins and the other adopts, so structure survives with a definite position -- and had never
been run. It has now been run, on both representations.

| O-Voxel arm | DreamSim rh | DreamSim rv |
|---|---|---|
| r1_pin_full (full parity) | 0.0599 | **0.2315** |
| r1_xc_copy (+ reconcile) | 0.0588 | 0.2901 |
| r2_pin_full (full parity) | 0.1018 | **0.2731** |
| r2_xc_copy (+ reconcile) | 0.1019 | 0.3092 |

| pipeline arm (orange) | DreamSim |
|---|---|
| control | **0.0743** |
| SEC_XCONS_MODE=mean | 0.0826 |
| SEC_XCONS_MODE=copy | 0.0867 |

The transverse column does not move (0.0599 to 0.0588, 0.1018 to 0.1019) and the longitudinal
gets substantially worse on both routes, by 0.059 and 0.036. The mechanism is not mysterious in
hindsight: `copy` overwrites the longitudinal targets with what the transverse family says along
the shared lines, so the families do agree afterwards, but the longitudinal family is then
supervised in part by another family's photographs -- and `rv` is scored against the longitudinal
photographs. The agreement is bought by making one family give up its own evidence, and the
measurement asks whether it kept it.

On the pipeline both modes cost, and copy costs more than mean. That also reproduces the page's
own figure independently: it reports 0.012 for mean and this run measures 0.0083, same order and
same sign. So the page can make its claim more strongly than it does -- both modes were tried and
the non-averaging one is worse -- rather than reporting the averaging mode alone.

The longitudinal defect is therefore not a missing consistency term. The only thing measured to
reduce it so far is SEC_PATCH's band term (rv 0.2535 to 0.2230), and it does not remove it. The
next place to look is the geometry rather than the supervision: the streak directions recovered
by Hough and back-projection are (1,0,-1), (2,0,-1) and (1,0,2) in the lattice frame, and whether
those correspond to particular dual-grid topologies has not been checked.
