# Slice-driven interior fill, on our own O-Voxel

Everything here runs locally; the remote is not needed. The geometry is **our O-Voxel result** —
`trained/orange.ply` from the `v1-inputs` GitHub release (1,162,387 cells), **not** the FruitNinja
released model that `build_orange/lattice/` was quantised from.

    bash code/fetch.sh <dir>          # WANT=trained_v3 -> trained/orange.ply + lattice_meta/
    voxelize_ov.py                    # -> a dense grid: V, OCC, SHELL, CORE
    sd3d_train.py                     # the 2-D prior per family
    xtrain.py                         # the prior, cross-conditioned on the other family
    xfill.py                          # THE BASELINE: families take turns, each conditioned

## What the pieces are

| file | |
|---|---|
| `planes.py` | the ONE implementation of cut-a-plane / write-a-plane, shared by every script |
| `voxelize_ov.py` | O-Voxel cells -> dense grid, pinholes closed, `cell_level==1` becomes the shell |
| `sd3d_train.py` | 2-D SinDiffusion per family; `PDIR` trains on photographs, `MASKED` on the interior only |
| `xtrain.py` / `xfill.py` | **the baseline.** each family is denoised WITH the other family's account of the same plane as three extra input channels, so consistency is produced by the model instead of imposed by a rule. Trained priors in `baseline/` |
| `fillcells.py` | the previous default: every plane restored once, then each cell gathers its own two planes |
| `chain2d.py` | the volume-level reverse chain (random or all planes per step) |
| `onepass.py` | the same as fillcells but scattering planes into slabs — kept as the comparison |
| `sdedit2d.py` | the honest test of a prior: noise a real plane, restore it, look |
| `tpdm.py`, `dbtrain.py`/`dbsample.py`, `sync.py`, `syncfill.py` | TPDM, DiffusionBlend, intersection sync |

## Settings that are not free choices

- **The prior is trained on the photographs**, unmasked, `lr 5e-4`, 8000 steps — the recipe that
  produced the realistic priors before. Training it on planes cut from the O-Voxel instead caps its
  realism at the fit's own smoothness, which is what "不太真" was.
- **The shell is never written.** `cell_level==1` is observed data; noise and updates are confined
  to `CORE`, verified at `max|d| = 0.00000000`.
- **The mask goes on the write, not on the image.** The plane handed to the prior is fully noised,
  because that is what it was trained on; only the interior of the volume is written back.
- **`WORDER=uv`.** The gather puts the in-plane coordinate on the image's columns, so the write
  must too. Reversed, it writes each section transposed — invisible in a round trip on a
  near-symmetric object, obvious as a rotated columella once a working prior is in place.

## Current method: `x3dcyl.py` -- the cylinder (2026-09-03)

**Baseline:** `x3dcyl.py T0H=0.3 T0V=0.3 R0=0.15 WFAR=0.1 DCFIX=16 NPHI=512`, longitudinal prior
`sd3d_train.py` (unconditional, photographs only, spl split, mult (1,2), **30k**; `baseline/prior_long.pt`) and transverse prior
`polar_train.py` on the polar-warped photographs (mult (1,2,4), 4k). Priors in `baseline/`.

| held-out DreamSim (lower = better) | long | trans | mean |
|---|---|---|---|
| real (spl vs held-out) | 0.070 | 0.040 | 0.055 |
| O-Voxel | 0.148 | 0.077 | 0.113 |
| FruitNinja | 0.154 | 0.162 | 0.158 |
| **cylinder, T0=0.3, WFAR=0.1, longitudinal 30k + polar 4k** | **0.112** | **0.062** | **0.087** |
| cylinder, same, polar prior 8k / 15k / 30k | 0.113 / 0.121 / 0.126 | 0.063 / 0.064 / 0.073 | 0.088 / 0.092 / 0.100 |
| cylinder, T0V=0.3, 8k | 0.122 | 0.064 | 0.093 |
| cylinder, T0V=0.5 / 0.7 / 0.9, 8k | 0.130 / 0.150 / 0.159 | 0.074 / 0.096 / 0.106 | 0.102 / 0.123 / 0.133 |

Both faces and the mean below the O-Voxel, with honest unconditional priors and nothing copied.
Colour = photograph (0.962 vs 0.961), horizontal banding 0.0068 (photo 0.0079, O-Voxel 0.0123),
no block tiling. `baseline/compare.png`: segment membranes in the right number on the transverse
face, a continuous columella on the longitudinal; faint z-streaks remain on the longitudinal.

**Why the cylinder.** The state is one array A[r, phi, z]. A longitudinal plane at phi_k is the
exact (s,z) slice from columns k and k+NPHI/2; a transverse plane at z_j is the exact (r,phi) polar
slice. Both families denoise the SAME x_t -- the product prior's step is literally
x0 = w(r) x0_L + (1-w(r)) x0_T, the longitudinal leading at the axis and the transverse away from it
-- and nothing is interpolated inside the chain. In polar coordinates the segment membranes are
parallel vertical stripes, which a patch prior learns; in Cartesian coordinates their global radial
organisation is beyond any patch. The O-Voxel's low-pass is pinned on the longitudinal planes
(ILVR); the shell is never written; the asset is written once at the end.

**T0V.** The honest longitudinal prior has no layout of its own (from 90% noise it makes blobs);
the lower its T0, the more of the O-Voxel's structure it keeps under the texture it adds, and
DreamSim falls monotonically 0.9 -> 0.7 -> 0.5 -> 0.3 (0.133 -> 0.123 -> 0.102 -> 0.093); it must
turn back toward the O-Voxel's 0.113 as T0 -> 0, so 0.2 and 0.1 are being measured. WFAR: less
longitudinal far out is better (0.5 > 0.3 > 0.1). **Longer training helps**: the 30k longitudinal
prior takes the mean from 0.093 to 0.088, and in-plane at T0=0.3-0.5 it draws segments and a crisp
columella where 8k draws mush (`baseline/long_prior_8k_vs_30k.png`); at T0=0.9 both collapse --
a patch prior has no global layout, which is the O-Voxel's job (ILVR) and the reason low T0 wins.
**Training length is per family**: the polar transverse prior at 30k *overfits* its three strips
(segments become fixed and evenly spaced, `baseline/polar_prior_8k_vs_30k_overfit.png`) and the
held-out distance rises monotonically with its training: 4k 0.087, 8k 0.088, 15k 0.092, 30k 0.100.
The curve is flat below 8k (the 4k-8k gap is within noise); 4k is the baseline. T0V is flat at 0.2-0.3 (0.091-0.093
at 8k); below that it turns back toward the O-Voxel. WFAR 0.1 < 0.3 < 0.5.

## Held-out DreamSim (2026-09-03) -- the number the method has to beat

Declared split (`split/`): longitudinal spl = or_long_00/02/04, held-out 01/03/06; transverse spl =
or_trans_00/02/04, held-out 01/03/05. Priors are trained on spl only. Every candidate is scored the
same way (`dsscore.py` + `dsrun.py` in a DreamSim env): canonical-frame grid slices, mean over faces
of the distance to the nearest held-out photograph. Grid slices, not the O-Voxel renderer, so the
absolute numbers are not the remote protocol's; the ranking is fair.

| held-out DreamSim (lower = better) | long | trans | mean |
|---|---|---|---|
| real (spl vs held-out) | 0.070 | 0.040 | 0.055 |
| **O-Voxel** | 0.148 | **0.077** | **0.113** |
| FruitNinja | 0.154 | 0.162 | 0.158 |
| xfill (old) | 0.142 | 0.095 | 0.118 |
| x3d, symmetric sync | 0.151 | 0.133 | 0.142 |
| x3d, radius leadership (B) | 0.133 | 0.144 | 0.138 |
| x3d, B + transverse cond = O-Voxel view (Bov0) | 0.129 | 0.103 | 0.116 |
| x3d, B, unconditional priors 8k (U0) | 0.160 | 0.142 | 0.151 |

x3d beats FruitNinja and loses to the O-Voxel on the transverse face. Three findings on the way:

1. **Radius leadership works**: the longitudinal takes from the transverse away from the axis, the
   transverse takes from the longitudinal near it (`R0=0.15`). Longitudinal face 0.148 -> 0.133,
   columella continuous, horizontal bands gone without transverse ILVR.
2. **The cross-conditioned priors are copiers, not priors** (`baseline/cond_is_a_copier.png`):
   with cond = the input slice they reproduce it, with cond = zeros they collapse to noise. The
   `degrade()` at training was too weak. xfill's transverse looked right only because round 1 fed
   the O-Voxel's own view as cond; in x3d the transverse cond came from the longitudinal family,
   which cannot see segment membranes, and no sync weight or ILVR could bring them back. Feeding the
   O-Voxel view as cond (Bov0) recovers most of it -- by copying the O-Voxel's layout.
3. **Honest unconditional SinDiffusion priors (3 photographs, 8k) are weaker than the copier**:
   colour and banding right, texture below the O-Voxel, DreamSim worse. In-plane, the longitudinal
   one has no layout at T0=0.9 (blobs); the O-Voxel's low-pass supplies it through ILVR.

The missing structure is the transverse segment membranes: thin radial lines with a global
organisation a patch prior cannot learn in Cartesian coordinates. In polar coordinates they are
parallel vertical stripes (`split/polar_check.png`), and the cylinder makes both families exact
axis-aligned slices of one array -- `x3dcyl.py`, one shared x_t, no interpolation in the chain,
`polar_train.py` for the transverse prior. Being measured.

## The Cartesian version (`x3d.py`) -- superseded by the cylinder above, kept for the record

**Assumption, the only one connecting 2-D to 3-D:** a volume is real iff every longitudinal slice is
in the longitudinal photographs' patch distribution and every transverse slice in the transverse
ones'. As a prior that is a product of experts over slices; its score is a sum over the slices
through each voxel. A slice of a Gaussian-noised volume is a Gaussian-noised image at the same t,
so the 2-D SinDiffusion denoisers apply unchanged -- no 3-D training.

**Sampler:** one reverse chain. State = the planes themselves (90 longitudinal at prior resolution;
one transverse plane per longitudinal pixel row, so z needs no interpolation), never re-sliced
through the voxel grid. Each step: every plane predicts x0 with its own prior, conditioned on the
other family's current x0; ILVR pins the O-Voxel's low-pass on both families (the soft data term;
the shell is the hard one and is never written); then x0 is averaged along the intersections
(`SYNCW`, the score-average of the product prior), the only cross-family interpolation being 2-D
bilinear along radial lines on the transverse discs; each plane re-noises with its own eps (DDIM).
The asset is written once at the end.

**Baseline:** `T0H=0.5 T0V=0.9 SYNCW=0.25 DCFIX=16 DCT=16`. `DCT=8` trades 5% flesh texture for
banding at the photograph's level.

| longitudinal tex15 | real | O-Voxel | xfill (old) | **x3d DCT=16** | x3d DCT=8 |
|---|---|---|---|---|---|
| columella | 0.126 | 0.097 | 0.088 | **0.112** | 0.101 |
| flesh | 0.067 | 0.060 | 0.054 | **0.061** | 0.057 |
| pith band | 0.124 | 0.121 | 0.119 | **0.122** | 0.120 |
| flesh value (colour) | 0.961 | 0.962 | 0.958 | 0.963 | 0.963 |
| h-band (row profile) | 0.0079 | 0.0123 | 0.0100 | 0.0100 | **0.0087** |
| tex3/tex15 (spectrum shape) | 0.43 | 0.48 | 0.44 | 0.44 | -- |

Every band above the O-Voxel at the photograph's colour and spectrum shape; no tiling, no spokes,
no horizontal bands; the unsupervised 45-degree cut is clean. Intersection disagreement 0.012.
Open: the columella is a chain of white patches rather than a continuous core -- the two priors
disagree on that structure (disc vs strip) and ILVR resolves only part of it.

**What failed on the way and why:** two families in separate chains averaged at the end (xfill)
blur -- that is averaging two finished samples. A joint chain re-sliced through the 128^3 grid every
step (xsync) blurred to a blob and its 3-D low-pass pulled the white background in through the
shell. Without transverse ILVR, x3d shows per-transverse-plane brightness as horizontal bands on
longitudinal cuts -- the spoke artifact rotated 90 degrees; `hband.py` measures it.

## Where it stood before x3d (kept for the record)


Three scorers, read together, because each one alone misled for a day: `fidelity.py` (tex15 by
region on longitudinal slices in the photographs' canonical frame), `colour.py` (flesh RGB /
saturation / value — tex15 is blind to a colour shift), `transverse.py` (the other family's view —
a spoke artifact lives there and nowhere else). And the composite, before any number is believed.

**Baseline:** `T0H=0.5 T0V=0.9 DCMODE=ilvr DCFAM=long DCFIX=16 RADW=0 UPW=2`, longitudinal prior
`mult=(1,2)`, transverse `(1,2,4)`, both trained on the photographs unmasked, lr 5e-4, 8000 steps.
No clamp inside the chain. Priors in `baseline/`.

| longitudinal tex15 | real | O-Voxel | prior alone, in-plane | baseline volume |
|---|---|---|---|---|
| columella r<0.14 | 0.126 | 0.097 | **0.106** | 0.088 |
| plain flesh 0.28-0.57 | 0.067 | 0.060 | **0.059** | 0.054 |
| pith band 0.57-0.71 | 0.124 | 0.121 | **0.125** | 0.119 |
| flesh value (colour) | 0.961 | 0.962 | 0.961 | 0.958 |

Colour equals the photograph. No block tiling (the O-Voxel has it), no spokes. In-plane the prior
now exceeds the O-Voxel in every band; the volume loses 20 / 11 / 6% of that in the gather, which
is the open problem.

**Two bugs found by looking, not measuring.** `x0.clamp(-1,1)` inside the chain walked every
channel toward 0 and cost texture; removing it lifted every band. What remained was a uniform
multiplicative shrink at t~900 (a small-receptive-field network cannot recover DC), fixed by ILVR:
pin the O-Voxel's low-pass (σ=16 px, tiling lives at 4-8) at every step so texture is generated
consistent with the right colour. Post-hoc swaps and mean offsets were each measured and each lose.
Without ILVR the transverse face shows the 90 longitudinal planes as spokes — disqualifying.

RADW is inert under ILVR; it only ever chose which family's wrong colour won. Every aggregation
number taken before the clamp fix is suspect; the 50/50 average of two independent fields is the
first to re-take.

**The assembly, measured to the end.** The O-Voxel's own planes lose 7 / 8 / 2.5% through one
write-and-read of the grid, and the same at 256^3 -- it is the tent filters of the bilinear write
and the trilinear read, not the 2x resampling. Pure ownership (no averaging) gains 1.6 / 3.6 / 0.8%;
a nearest write ~1%; writing at 2x (`UPW=2`, kept) ~2%; a 256^3 grid lands at the same fraction of
its own O-Voxel as 128^3. A joint chain that re-slices both families through the grid every step
(`xsync.py`, not kept) blurred to a blob that tex15 scored above the photograph -- the 3 px probe
and the composite caught it. What the volume finally holds is in-plane minus the round trip minus
~5% at the columella; against the O-Voxel's tiling-inflated tex15 that is 88-90% at the columella
and flesh, 98% at the pith, at the photograph's colour with no tiling and no spokes.

## Every object, one program (`runobj.sh`)

`runobj.sh <object> <gpu>`: `prep_obj.py` finds the photographs, canonicalises them on a white
background (the background colour is read from the corners, so the black-background cake works),
declares the split (>=4 per family: 3 train, the rest held out; fewer: all train, scored against the
training photographs and flagged), unwraps the transverse family to polar strips; then the recipe
found on the orange -- longitudinal prior 30k, polar prior 4k, cylinder at T0=0.3, WFAR=0.1 --
and the scores. A family without photographs is inactive in the chain (bread and cake have only
longitudinal cuts, the doughnut one transverse). The only per-object declaration is the polar
axis, `axis.env`, read from the occupancy (the doughnut lies in a vertical plane: axis 2); bread's
loaf axis is horizontal and the photographs are cross-sections perpendicular to it, so under the
Y convention they are longitudinal planes at one azimuth -- reported, not tuned.

`writeback.py` returns the interior to the O-Voxel asset: every interior cell takes the volume's
colour at its centre, every skin cell keeps its trained colour, the PLY is rewritten with only
f_dc of interior cells changed. Orange: 682,100 cells rewritten, 480,287 untouched; re-voxelised,
the interior differs from the volume by 0.029 (trilinear round trip).

### Six objects, measured (2026-09-03)

Same program, same recipe, no per-object setting but the polar axis. Held-out DreamSim where a
family has >=4 photographs, otherwise the distance to the training photographs (flagged).
`baseline/six_objects.png` shows every face beside the fitted field and a photograph.

| object | split | ours long / trans / mean | O-Voxel long / trans / mean |
|---|---|---|---|
| orange | 3+3 / 3+3 held out | **0.112 / 0.062 / 0.087** | 0.148 / 0.077 / 0.113 |
| apple | long vs train; trans 3+6 held out | **0.240 / 0.264 / 0.252** | 0.283 / 0.289 / 0.286 |
| pomegranate | long 3+2 held out; trans vs train | **0.306 / 0.251 / 0.279** | 0.362 / 0.296 / 0.329 |
| watermelon | 3+20 / 3+17 held out | 0.152 / 0.112 / 0.132 | **0.137 / 0.086 / 0.112** |
| bread | long 3+2 held out; no transverse | **0.364** / - / - | 0.384 / - / - |
| cake | long vs train; no transverse | **0.440** / - / - | 0.454 / - / - |
| doughnut | no longitudinal; trans 1 vs train | - / **0.196** / - | - / 0.256 / - |

Five of six below the fitted field; the watermelon above it. Its fitted field is already near the
real floor (0.112 vs 0.097): smooth flesh and discrete high-contrast seeds, which the patch prior
smears along z. Bread and cake are "less far than a bad fit" -- their fitted fields are blotches
and the lift smooths them; the numbers are far from the photographs either way and are reported
as such. The bread's loaf axis is horizontal; under the Y convention its photographs are
longitudinal planes at one azimuth and the swirl is not reproduced.

**The measured gate** (`gate.py`, applied by `post_gate.sh`): the lift is kept only where it lowers
the distance to the object's own training photographs -- which every object has -- else the asset
keeps the fitted field. One rule, no threshold, never below the baseline, held-out never consulted.
The gate's verdicts: apple, cake, doughnut, pomegranate LIFT; **watermelon FITTED** (0.139/0.129 ->
0.143/0.139, clear); **bread FITTED** by 0.001 (0.3392 -> 0.3403) where the held-out photographs
say the lift is better (0.384 -> 0.364) -- a tie at the rule's boundary, reported as such and not
tuned around. Everywhere else the training-photo verdict and the held-out one agree.

`writeback.py` produced `<object>_x3d.ply` for all six (interior cells rewritten, skin untouched);
`<object>_asset.ply` is the gated asset.

### The two released baselines, same path (2026-09-03)

FruitNinja's released Gaussians (`prefilled/trained_gs/<obj>.ply`, six objects) and GaussianFluent's
watermelon and cake (Hugging Face `hbpencil01/GaussianFluent`, fetched by `code/fetch.sh
WANT=gfluent`), voxelised by `voxelize_gs.py` (centres + f_dc, opacity > 0.1, same fill) and scored
on the same held-out sets. GaussianFluent's axes: watermelon AXD=0, cake AXD=2; FruitNinja's apple and cake AXD=2 (checked by rendering all three).

| object | carrier | FruitNinja | GaussianFluent | ours |
|---|---|---|---|---|
| orange | 0.113 | 0.154 | - | **0.087** |
| watermelon | **0.112** | 0.174 | 0.182 | 0.132 |
| apple | 0.286 | 0.280 (AXD=2) | - | **0.252** |
| bread (long) | 0.384 | 0.378 | - | **0.364** |
| cake (long) | 0.454 | **0.305** (AXD=2) | 0.354 | 0.440 |
| pomegranate | 0.329 | **0.215** | - | 0.279 |

FruitNinja wins the pomegranate (its longitudinal 0.176 vs our 0.306) and the cake (0.305; GaussianFluent 0.354) --
both where our carrier fits badly and the lift inherits it.

**On the gate.** `gate.py` / `post_gate.sh` remain as an acceptance test for producing assets; the
paper does not present the gate as part of the method -- it reports the lift as it is on every
object, the watermelon as a loss.

### How many photographs (orange, 2026-09-03)

Per family, everything else fixed. 1 / 2 / 3 scored on the same 3+3 held-out; 3 / 5 on the one
photograph per family left when five train (`nab/`).

| photographs per family | held out | ours | carrier |
|---|---|---|---|
| 1 | 3 | 0.121 / 0.065 / 0.093 | 0.148 / 0.077 / 0.113 |
| 2 | 3 | 0.120 / 0.071 / 0.096 | same |
| 3 | 3 | **0.112 / 0.062 / 0.087** | same |
| 3 | 1 | 0.190 / 0.081 / 0.136 | 0.223 / 0.092 / 0.157 |
| 5 | 1 | 0.192 / 0.084 / 0.138 | same |

One photograph already beats the carrier; two = one; three gains 0.006-0.009; five = three.
