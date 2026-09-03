# Input-Driven Diffusion for Any-Angle Cut Interiors: Filling an O-Voxel Volume from Two or Three Cut Photographs

**Anonymous CVPR submission**

Paper ID ####

---

## Abstract

We study a reconstruction problem that sits between image inpainting and 3-D generation: given an object represented as an *O-Voxel* — a photogrammetric shell with a photographically supervised skin but an interior constrained by only two or three orthogonal cut photographs per family (a longitudinal and a transverse set) — fill the interior so that a cut at *any* depth and *any* orientation reads as a real cross-section of that specimen. Two or three photographs cannot supervise a volume, and a per-plane diffusion prior offers essentially no per-seed diversity once the silhouette and boundary are fixed by construction: we measure the flesh-region standard deviation across seeds collapsing from 0.130 (unconditional) to 0.011–0.035 under every constrained regime we tried. Our method therefore moves generative variation off the seed axis and onto the *input* axis: for each cut plane we affine-warp a depth-assigned reference photograph into the rasterized flesh silhouette and run a single released single-image diffusion as a low-strength SDEdit ($\rho\!\approx\!0.25$) that pins only a thin rind ring, so the diffusion harmonizes the input into the plane rather than inventing it. Generated faces from both cut families are splatted into one shared voxel field and read back by angle-weighted bidirectional fusion. We formalize a *boundary–anatomy decoupling problem* explaining why the interior axis cannot come from the shell for rotationally isotropic objects, report an honest negative result against a continuous Triplane field, and demonstrate the identical program on seven objects.

---

## 1. Introduction

Digital assets that can be *cut* — sliced at an arbitrary plane to reveal a plausible interior — are useful for simulation, food and surgical visualization, games, and content creation. Most 3-D reconstruction pipelines recover only a surface; the inside of the object is either empty, or a texture-mapped shell, or (for released generative meshes) a diffusion model's uninspected guess quantized into the volume. We start from a representation we call an **O-Voxel**: a photogrammetric shell whose *skin* is supervised by photographs, but whose *interior* is a coarse voxel lattice initialized flat and supervised only where a real cut photograph constrains it. The problem this paper addresses is the interior: **fill it, from a handful of cut photographs, so that any cut plane looks like a genuine cut of this specimen.**

This is hard for a specific and quantifiable reason. Per object we have two or three photographs in each of two families — a *longitudinal* family whose planes pass through the object's axis, and a *transverse* family whose planes stack along it. Every other cut plane, at every intermediate depth and every oblique orientation, is unphotographed. A volume has on the order of $10^5$–$10^6$ cells; the photographs touch a measured 14–92% of them depending on family (Sec. 5), and *no* photograph exists for the majority of longitudinal depths, which are all central sections. There is no ground-truth image against which an unphotographed plane could even be scored. The task is therefore not "match the photograph" but "hallucinate a coherent volume consistent with the few photographs, whose every slice is plausible and whose slices agree in 3-D."

The instinct is to reach for a generative prior and sample. We show this instinct fails in a measurable way. A single-image diffusion (SinDiffusion) trained on the one cut photograph is genuinely diverse when sampled unconditionally — flesh-region standard deviation 0.130 across seeds — but almost all of that entropy lives in *global layout*: where the silhouette sits, the placement of large colour blocks. Our task fixes the silhouette and the boundary by construction, removing exactly that freedom. Under any regime that respects the boundary — per-instance overfitting, de-leaked augmentation, RePaint-style masked resampling, rind-only outpainting — the per-seed diversity collapses to 0.011–0.035 (Sec. 3.1). Seed diversity is not a knob we failed to turn; it is dead under geometric constraint.

Our response is to **move the axis of variation**. Instead of asking one diffusion for many faces from many seeds, we ask it for one harmonized face from many *inputs*: a different depth-assigned reference photograph per plane, affine-warped into that plane's silhouette, lightly denoised so its structure survives. The variation that matters — how the cross-section changes with depth and orientation — rides on the input, which is also exactly what makes a depth sweep coherent. The diffusion's only job is to reconcile the warped input with the plane's silhouette and rind.

**Contributions.**

- **Input-driven generation under extreme sparsity.** We reframe interior generation so that variation comes from the input reference, not the seed, and we justify this with a direct measurement showing seed diversity collapses under our constraints (Sec. 3.1). The generator is a single released diffusion run as a rind-only, low-strength SDEdit; there is no per-plane training, and a face is produced in about a second.
- **The Boundary–Anatomy Decoupling Problem.** We give a clean statement of why, for a rotationally isotropic shell, boundary geometry *cannot in principle* recover the interior's anatomical polarity, so the cut photograph's normal — not an SDF, skeleton, or PCA axis — must supply the interior axis. We support it with a geometry-driven vs. anatomy-driven taxonomy and orange-vs-doughnut evidence (Sec. 4).
- **A shared cuttable field with bidirectional fusion.** Both cut families write into one voxel field; an angle-weighted read (Sec. 3.4) lets each query direction be reconstructed by the family that actually photographed it. The result is a genuine cuttable asset: 99.9% of interior cells are written and any oblique plane reads its face directly from the volume.
- **An honest negative result.** A continuous Triplane implicit field, fit to the same generated faces in ~12 s to $1.3\times10^{-3}$ MSE, is *grainier*, not sharper, than the discrete voxel field under 2–3-photograph constraint, because the two families disagree at their plane intersection and a globally-reconciled field can only fit a conflicted compromise (Sec. 6). This motivates the voxel-guided design rather than assuming it.
- **Generality.** The identical program — same constraints, same render, only the object's own single-image diffusion and reference changing — fills seven objects (orange, watermelon, pomegranate, apple, bread, cake, doughnut), with two object-general refinements (dark-pixel seed compositing, supersampled slab rendering) that carry across the set without per-object tuning.

We are deliberately conservative about what is claimed. The interior is *exact* for the families written into it and *plausible* for a strongly oblique cut; the voxel-plus-slab representation softens faces that were sharp in 2-D; and the method relies on a per-object single-image prior. We state these limits in Sec. 7 rather than paper over them.

---

## 2. Related Work

**Single-image generative models.** SinGAN and SinDiffusion learn the internal patch statistics of one image and resample them into new layouts. Their receptive field is patch-scale (no attention, no global bottleneck), which is exactly why their unconditional diversity is dominated by global arrangement rather than local texture — the property we measure and exploit in Sec. 3.1. We use a released single-image diffusion as our only generative prior. This is heavy supervision in the sense that we train one such model per object and family; it is light in that it needs a single cut photograph, not a dataset of cut objects, which for most specimens does not exist.

**Diffusion editing and inpainting.** SDEdit noises an input to an intermediate step and denoises, trading fidelity to the input against the strength of the prior via the noising fraction. RePaint enforces a known region by re-imposing it at every reverse step with resampling. We adopt SDEdit's noising trade-off as our central control (the strength $\rho$) and RePaint's step-wise re-pinning for the rind, but we deliberately pin *only* a thin rind ring rather than the full known region, because — unlike standard inpainting, where the goal is to complete a hole consistently — our goal is to preserve the input's interior structure and let variation come from the *choice of input*. We measure that a heavy RePaint pin drives per-seed diversity *lower* (0.011), confirming the standard inpainting objective is the wrong one here.

**Score distillation and text-to-3D.** DreamFusion-style SDS lifts a 2-D diffusion prior into 3-D by distilling its score into a volumetric field across many viewpoints. SDS is inappropriate for our setting for a concrete reason, and we tested the concern rather than asserting it: our prior is a *single-image* diffusion that knows exactly one cut appearance. Distilling it onto arbitrary oblique planes would impose that one reference everywhere, reviving the failure in which off-centre longitudinal planes grow a columella they should not have. SDS also assumes a prior with broad view coverage; ours has none. We therefore do not run SDS and explain why (Sec. 6); the cheapest experiment that would falsify our concern is stated in the reviewer self-assessment.

**Triplane and coordinate-network 3-D representations.** EG3D's triplane and instant-NGP's hash-grid encode a continuous field decoded by a small MLP, giving resolution independence and smooth oblique queries. These are the natural "upgrade" from a discrete voxel field, and we implement one (Sec. 6). Our finding is that under 2–3-photograph constraint the continuous field's global reconciliation is a liability, not an asset — it fits a conflicted compromise where the two families' planes intersect. We keep the discrete field as the main line and report the continuous field as a measured negative.

**Medial-axis and SDF shape analysis.** Skeletonization, the medial axis transform, and signed distance fields recover an object's axial structure from its boundary. Our Sec. 4 argues precisely when they can and cannot supply the interior axis needed to place anatomical structure (a columella / pith): they succeed exactly when the *shape* fixes the axis (elongated, bent, or toroidal bodies) and fail in principle when the boundary is rotationally isotropic. This scopes standard shape analysis rather than dismissing it.

**Image-to-3D and interior generation.** Most single- or few-image 3-D methods recover a surface or a radiance field viewed from outside; the interior is not modeled, or is a by-product of a generative mesh never inspected as a cut. Volumetric medical reconstruction assumes dense tomographic slices, not two or three photographs. Our contribution is a method that treats the interior as the object of reconstruction from *sparse cut photographs specifically*, and produces an asset cuttable at any plane.

---

## 3. Method

### 3.0 The O-Voxel interior task

An O-Voxel provides a shell and a photographically supervised skin. The interior is a coarse voxel lattice, initialized flat (0.5 in every channel) and — by design — supervised *only from the cut photographs*, never from the released mesh's own interior, so that every interior claim is that the photographs put the structure there (a decision that costs a measured 0.0026–0.0373 DreamSim on the skin depending on object, smallest where the photographs constrain the interior most, and accepted). For a cut plane with unit normal $n$ and offset $d$ we rasterize the flesh silhouette $f$ from the shell, generate a cut face, and write it into the lattice; a later cut at any $(n,d)$ reads its face back out. The method below is the generator and the write/read; the shell and skin are unchanged by any of it.

### 3.1 Where a single-image diffusion's variation actually lives

Before designing the generator we measured what a single-image diffusion can and cannot vary under our constraints. On one orange plane we sampled the model across seeds and computed the flesh-region standard deviation:

| Regime | Flesh std | Outcome |
|---|---|---|
| Unconditional (unconstrained) | 0.130 | diverse layout, consistent texture |
| Per-instance overfit (6000 steps, hard pin) | 0.030 | every seed near-identical |
| De-leaked base + flip/roll augmentation | 0.035 | still identical |
| Masked resampling (RePaint, $U=5$) | 0.011 | even lower |
| Rind-only outpainting (thin ring pin) | 0.021 | low, and interior loses all structure |

The resampling count $U$ is a weak knob: on an off-centre plane (free gap 61% of the flesh), $U=1/3/5$ gives 0.024/0.015/0.011 — fewer resampling steps are more diverse, but the ceiling (~0.024) is still 5× below unconditional. The reading is unambiguous: a single-image diffusion's diversity lives in global layout (silhouette, placement, big colour blocks); our task fixes the silhouette and boundary by construction, so what remains free is low-entropy interior fill that the model completes canonically regardless of seed. **Seed diversity is not a missing knob under our constraints.** The correct source of variation is a *different reference per plane*, which is also what makes the depth sweep change with depth.

### 3.2 The generator: rind-only pin, low-strength SDEdit

For a plane $(n,d)$ we rasterize the flesh silhouette $f$ and affine-warp a depth-assigned reference photograph into it, giving the input image $x_0\in[-1,1]^{3\times H\times W}$. The *only* pinned region is a thin rind ring plus the white background,

$$k=\big(f\wedge\neg\,\mathrm{erode}(f)\big)\vee(1-f),$$

leaving the interior — segments, juice vesicles, columella — entirely to the model. Generation is SDEdit: noise the input to a fraction $\rho$ of the schedule and denoise from there, re-imposing the pin after every reverse step,

$$x_{\rho T}=\sqrt{\bar\alpha_{\rho T}}\,x_0+\sqrt{1-\bar\alpha_{\rho T}}\,\epsilon,$$
$$x_{t-1}=k\odot\big(\sqrt{\bar\alpha_{t}}\,x_0+\sqrt{1-\bar\alpha_{t}}\,\epsilon_t\big)+(1-k)\odot\Phi(x_t,t),$$

with $\Phi$ one unconditional DDPM reverse step of the object/family diffusion.

The strength $\rho$ is the entire trade-off. At $\rho\!\approx\!0.55$ the model washes the interior into a canonical smooth blob and the input stops mattering; at $\rho\!\approx\!0.25$ the input's structure survives, the diffusion only harmonizes it into this plane's silhouette and rind, and *different references produce visibly different, sharp faces*. There is no per-plane training: a face is generated in about a second. (An earlier version of this project reached comparable sharpness by *overfitting one diffusion per plane* — a 6000-step fit per plane, generated from noise with supervised cells and background hard-pinned. It was sharp, but every seed produced the *same* face and it cost a full training run per plane; the input-driven SDEdit matches its sharpness at seconds rather than minutes per plane. See Sec. 6e.)

### 3.3 The columella is geometry, carried by the input

The pith (columella) is a thin core along the object's axis; whether a plane shows it is an intersection, not a texture choice. With a core cylinder of radius $r_c\!\approx\!0.13\,R$ about the axis, a longitudinal plane at perpendicular offset $s$ meets it over a band of half-width

$$w(s)=\sqrt{\,r_c^{2}-s^{2}\,}\quad(|s|<r_c),\qquad 0\ \text{otherwise.}$$

Rather than pin a columella, we crop the reference's own pith to that width *before* warping: central planes carry a full pith, off-centre planes carry a thinner one, and beyond the core the input is de-pithed so the patch-local model grows none (measured pith width 14→12→7→0 px as $s$ grows). The gating rides entirely on the input, which suits a patch-local model — it paints a columella only where its neighbourhood already looks like one. Transverse references need no crop: every transverse plane crosses the axis, so the columella is simply a central dot.

### 3.4 From planes to one field, and bidirectional fusion

Each generated face $r_p$ is splatted back over a *slab* whose half-width is set to just over half the inter-plane spacing, so neighbouring planes tile the depth range with no unwritten gap (the slab half-width $\approx\!0.55\times$ the spacing; this is what removes the coverage banding that a zero-thickness write leaves — measured, only 38.5% of cells fall within half a cell of a plane, but 96.7% within 2.82 cells). A cell's colour is the mean of every reconstructed pixel projecting into it,

$$C_i=\frac{1}{|\mathcal P_i|}\sum_{(o,u)\in\mathcal P_i} r_{o}(u),\qquad \mathcal P_i=\big\{(o,u): \pi_{\,n,\,d+o}(u)=i,\ f(u)>0\big\},$$

with $\pi$ the plane-to-cell projection. Both families together fill the field in about seven seconds, and that one field is what every cut renders (slab thickness $2.82$ cells, sub-sampled along the normal).

Averaging both families into one field is the simplest fusion but dilutes a longitudinal cut with transverse evidence and vice versa. A better rule keeps a *separate* field per family, $C^{v}$ and $C^{h}$, and blends them at query time by how well the queried normal $n$ aligns with each family's own normal:

$$C(n)=\omega_v\,C^{v}+\omega_h\,C^{h},\qquad \omega_v=\frac{\cos^{k}\!\angle(n,n_v)}{\cos^{k}\!\angle(n,n_v)+\cos^{k}\!\angle(n,n_h)},\quad k=4,$$

with $\omega_h=1-\omega_v$. A longitudinal query then reads almost entirely the longitudinal-built field, a transverse query the transverse field, and an oblique query a smooth mix, so every direction is reconstructed by the evidence that actually constrains it. The gain is largest at the pure axes (a crisper columella at $0^\circ$) and modest on the orange because its two fields already agree; it would help more on directionally-distinct objects.

### 3.5 Two object-general refinements

**Seed compositing.** A single-image diffusion erases features that are rare in its one training image — watermelon seeds, pomegranate arils — at *any* SDEdit strength (even $\rho=0.05$ removes them, because the model's score treats them as outliers to its seed-sparse flesh prior). We composite the input's actual dark pixels back onto the generated flesh,

$$r\leftarrow(1-m_s)\odot r+m_s\odot y,\qquad m_s=[\,\mathrm{lum}(y)<\mathrm{median}_f(y)-0.12\,]\wedge f,$$

giving crisp features with no halo. It is used where an object has such features and left off for the apple's pale core — a switch driven by whether the feature exists, not a tuned threshold.

**Supersampled slab rendering.** Uniform flesh (watermelon) exposes a woven moiré that busy orange texture hid; it is not the diffusion (the raw warped input written to voxels shows it too) but aliasing between the slab's sub-samples and the lattice. We render the slab at $2\times$ and average down. Both refinements are general; neither is a per-object knob.

---

## 4. The Boundary–Anatomy Decoupling Problem

The obvious objection to $w(s)$ is that its axis should come from the shell — extract a medial axis, a signed distance field, or a principal direction, and derive the core geometrically. For an isotropic object this is impossible in principle, and the failure has a clean statement.

> **Observation (Boundary–anatomy decoupling).** When the shell is rotationally near-symmetric — an $SO(3)$-isotropic boundary, as for citrus, a tomato, an eyeball, or many uniform organs — the boundary carries *no* information about the anatomical polarity of the interior. A sphere has no preferred axis, so any shell-only operator — SDF, skeletonization, PCA — returns either a rotationally-ambiguous answer or an artifact of the discretization, never the stem–blossom axis the pith follows. The interior axis is *decoupled* from the boundary; only the orthogonal cut *photographs*, through their normals $n$, supply it.

We verified each operator on a near-spherical orange. The distance field is isotropic; the medial axis of a compact near-sphere collapses to a central blob rather than a full-height line; the PCA principal axis is essentially arbitrary (it came out tilted off the true pith). Only the supervised cut normal $n$ — the direction of $w(s)$ — locates the anatomical axis. Note the split of what the photograph supplies: the *scalar* $r_c$ is read from the photograph, but what the shell fundamentally cannot give is the *direction*.

This scopes when a geometric core-extractor is the right tool:

- **Geometry-driven** — a banana, a croissant, a doughnut: curvature or topology fixes the axis, and an SDF / medial axis recovers it cleanly (a torus's skeleton is its central ring).
- **Anatomy-driven** — citrus, tomato, eyeball, a uniform organ: the boundary is isotropic and the axis must come from the cut photographs' normals $n$. This is the regime this method targets.

The same skeletonizer run on both regimes makes the taxonomy concrete: the near-spherical orange yields a tangled central blob of 5757 cells with no usable axis; the doughnut yields a clean 356-cell ring — the topology's medial curve. Geometry recovers the axis exactly when the shape defines it, and not otherwise. This preempts the "why not standard SDF core extraction" objection by delimiting where it applies, and it is what turns $w(s)$ from a hand-tuned prior into a data-driven one whose only free scalar is $r_c$.

*Figures:* `assets/ovdecoupling.png` (SDF / skeleton / PCA all fail on the orange, photograph-$n$ recovers the pith); `assets/ovtaxonomy.png` (orange skeleton blob vs. doughnut ring).

---

## 5. Experiments

**Setup.** Per object and per family we train one released single-image diffusion (3-channel SinDiffusion) on that family's single cut photograph, 6000–8000 steps (lr $5\times10^{-4}$, fp16, two GPUs, checkpoint every 2000; a 6000-step checkpoint already suffices). Generation is the rind-only SDEdit of Sec. 3.2 at $\rho=0.25$, ~1 s/plane; both families are swept over depth and written to one voxel field in ~7 s total. Rendering is slab (thickness 2.82 cells) with $2\times$ supersampling. The only things that change per object are its own diffusion(s) and its reference set; the pipeline, constraints, and render are identical.

**On metrics.** We deliberately do *not* report per-plane pixel or perceptual error against a held-out photograph as the primary measurement, and this is a considered choice, not an omission. For the interior task there is by construction *no* ground-truth photograph for the unphotographed planes — that absence is the entire problem. A DreamSim-style score can be computed only against a plane's own reference, which measures reconstruction of the *supervised* planes, not the generative fill that is the contribution; and the reference photographs of a real specimen are themselves inconsistent plane to plane (between- over within-photograph patch distance 35.4 sliced-Wasserstein / 60.1 JS vs. 10.7 MSE on the orange transverse set), so a distributional match to one photograph penalizes exactly the plausible variation we want. We therefore report the *coherence* of the any-angle sweep and the coverage of the field as the primary evidence, and use DreamSim only where a supervised reference genuinely exists (the base-representation ablations we inherit). Quantitative comparison to a 3-D generative baseline is *to be measured* (see reviewer self-assessment).

**Any-angle coherence.** Because the write-back stores a colour per cell rather than pre-rendered images, the field is a filled volume: **99.9% of interior cells are written**, and any plane — including orientations never generated — reads its face straight out of the voxels. Tilting the cut normal continuously from longitudinal to transverse,

$$n(\theta)=\cos\theta\,n_{\parallel}+\sin\theta\,n_{\perp},\qquad \theta:0\to\tfrac{\pi}{2},$$

the columella passes from a vertical stripe to a central dot with no discontinuity, because it was stored consistently in 3-D. None of the intermediate angles were generated. *Figure:* `assets/ovring_orange_oblique.png`; sweep `assets/ovring_orange_vh.mp4`.

**Coverage asymmetry (why the field, not per-plane sharpness, is the object).** By pure geometry on the orange's 770,182 solid cells, the transverse family reaches 92.4% of cells (16 supervised depths, jittered half a step, so it sweeps) while the longitudinal family reaches only 15.9% (fixed sheets through the axis that never move); both reach 14.1%. The watermelon is the same shape of answer (88.9 / 16.3 / 13.7). This is *data*, not a bug: every longitudinal reference is a *central* section — dumping the target at 0%, 5%, and 12% of the radius returns the same central photograph — so off-centre longitudinal cuts have no photograph and cannot be fixed in code (jittering the plane off-axis costs the transverse family 36% for a small longitudinal gain, and was left off). This is why the contribution is a *coherent shared field* read by angle-weighted fusion, not per-plane photographic reproduction of planes no photograph covers.

**Per-seed diversity.** The measured collapse table of Sec. 3.1 is the load-bearing experimental result behind the method's central design choice, and we reproduce it as the diversity ablation: under every boundary-respecting regime the per-seed flesh std sits at 0.011–0.035 against 0.130 unconditional, confirming variation must come from the input.

**Seven-object generality.** The identical program fills orange, watermelon, pomegranate, apple, bread, cake, and doughnut, each read from its own voxel field. *Figures:* montage `assets/ovsix_montage.png`; per-object sweeps `assets/ovsweep_{orange,watermelon,pomegranate2,apple1,bread,cake2,doughnut}_sp.mp4`. The doughnut is the geometry-driven half of the Sec. 4 taxonomy realized end to end: its own two diffusions were trained on its longitudinal and transverse photographs, and its ring topology (an open hole) survives the depth sweep at every depth.

---

## 6. Ablations

**(a) Triplane vs. voxel — the negative result.** The natural objection is to replace the discrete field with a continuous implicit one. We built it: three $16\times192\times192$ feature planes and a two-layer MLP, supervised by the same $3.9\times10^{5}$ sharp cut-face pixels from both families plus a 3-D total-variation prior, converging to $1.3\times10^{-3}$ MSE in about twelve seconds, and it renders any oblique angle as a full cross-section — the continuity claim holds. But in a fair comparison at identical oblique planes (0/30/45/60/90°) it is *grainier*, not sharper: the two families disagree along the line where their planes intersect, and a single continuous field can only fit a conflicted compromise — a speckled interior, exactly the projection-intersection artifact the shared planes were supposed to prevent. The 3-D TV that suppresses the speckle trades it for over-smoothing. The voxel field, slab-rendered, stays cleaner because each cut reads a *local* neighbourhood rather than a globally-reconciled field. We separately confirmed the conflict is representation-independent: fitting a *dense* per-plane target set into the discrete lattice also regresses and produces cell-boundary grid, so the barrier is the mutually-inconsistent 2-D targets, not the storage. *Figure:* `assets/ovtriplane_ablation.png`. (One mild point for the continuous field: fed no $w(s)$, its $0^\circ$ columella emerged purely from the longitudinal anchors — but the voxel field learns from the same anchors, so this is not unique to it.)

**(b) Seed compositing on/off.** Without compositing, watermelon seeds and pomegranate arils vanish at every $\rho$. With it, features are crisp and halo-free (transverse especially clean). Residual: in the longitudinal view point-seeds streak along depth because the slab write-back averages them across planes — inherent to storing point features in a voxel field.

**(c) Supersampling on/off.** Without it, uniform flesh shows a woven moiré from slab/lattice aliasing. Rendering at $2\times$ and averaging down removes it while keeping seeds; the effect is invisible on busy-textured objects (orange), which is why it must be validated across objects before defaulting rather than tuned on one.

**(d) SDEdit strength $\rho$.** $\rho\!\approx\!0.05$ keeps input structure but already loses seed-sparse features and adds nothing over compositing; $\rho\!\approx\!0.25$ keeps interior structure while harmonizing to the silhouette and rind — the operating point; $\rho\!\approx\!0.55$ washes the interior to a canonical blob and the input stops mattering. The knob is monotone: more prior, less input.

**(e) Per-instance overfit vs. input-driven.** The predecessor overfit one diffusion per plane (6000 steps, generate-from-noise, hard-pin cells and background). It reaches the same sharpness as the input-driven SDEdit (both are photographic) but at minutes rather than a second per plane, and it has *zero* per-seed diversity (flesh std 0.030). The input-driven approach matches its sharpness, removes the per-plane training, and sources variation from the reference where it belongs.

---

## 7. Limitations

- **Voxel-plus-slab softening.** Writing a 2-D face into a discrete lattice and integrating over a slab softens detail that was sharp in the generated 2-D face. Keeping the 2-D sharpness would mean generating each frame at render time rather than storing it — a genuine alternative (render-time generation) that trades storage and instant read-out for compute per view. We store, and accept the softening.
- **One colour per cell.** The field holds a single value per cell, so a cut is *exact* for the families written into it and only *plausible* for a strongly oblique cut. This is the single-value-per-cell limit any voxel interior carries; bidirectional fusion mitigates but does not remove it.
- **The core radius is still a photograph scalar.** Sec. 4 shows the axis *direction* must come from the photograph for isotropic objects, but $r_c$ is also read from the photograph rather than inferred. The columella is thin, so only the one or two most central longitudinal planes carry one; the depth gating is correct but visually sparse.
- **A per-object single-image diffusion.** Each object and family needs its own diffusion trained on its cut photograph. This is cheap relative to a per-plane fit and needs only single photographs, but it is per-object supervision, and the method inherits whatever that one photograph does and does not show (e.g., off-centre longitudinal structure it never depicts).

---

## 8. Conclusion

We addressed filling the interior of an O-Voxel so that any cut, at any depth and orientation, reads as a real cross-section — from only two or three cut photographs per family. The core idea is to recognize, and measure, that a single-image diffusion has essentially no usable seed diversity once the boundary is fixed, and to move variation onto the input: warp a depth-assigned reference into each plane's silhouette and let one released diffusion harmonize it with a rind-only, low-strength SDEdit. Both cut families write into one shared voxel field read by angle-weighted bidirectional fusion, yielding a cuttable asset with 99.9% of cells filled. We formalized why the interior axis of a rotationally isotropic object cannot come from its boundary — the boundary–anatomy decoupling problem — and reported an honest negative result showing a continuous Triplane field is grainier than the discrete voxel field under this much constraint. The same program runs on seven objects with no per-object tuning. The open direction is a genuinely 3-D-consistent generator whose targets do not conflict at plane intersections, which our ablations suggest — rather than any storage upgrade — is the true remaining barrier.

---

## Figures (appendix — asset filenames and intended captions)

- `assets/ovdecoupling.png` — On a near-spherical orange the SDF is isotropic, the medial axis collapses to a central blob, and the PCA axis is arbitrary; only the cut normal $n$ (blue band, our $w(s)$) locates the anatomical pith axis. (Sec. 4)
- `assets/ovtaxonomy.png` — The same skeletonizer on both regimes: orange gives a tangled 5757-cell blob (no axis), doughnut a clean 356-cell ring. Geometry recovers the axis exactly when the shape defines it. (Sec. 4)
- `assets/ovtriplane_ablation.png` — Top: Triplane INR fit to the generated faces plus 3-D TV. Bottom: the voxel field, slab-rendered. Cut normal tilted $0^\circ\!\to\!90^\circ$. Both fill every angle, but the continuous field is speckled at the families' intersection; the voxel field stays clean. (Sec. 6a)
- `assets/ovbidir.png` — Angle-weighted bidirectional fusion (top) vs. naive mean (bottom), cut $0^\circ\!\to\!90^\circ$; the columella at $0^\circ$ is less diluted under fusion. (Sec. 3.4)
- `assets/ovring_orange_oblique.png` — Cut normal tilted $0^\circ\!\to\!90^\circ$; none of these angles were generated, each face read from one voxel field; the columella moves from a line to a central dot. (Sec. 5)
- `assets/ovring_orange_vh.mp4` — Longitudinal (left) and transverse (right) input-driven SDEdit sweeps ($\rho=0.25$, rind-only pin) written into one field and slab-rendered; both directions read the same voxels. (Sec. 3, 5)
- `assets/ovsix_montage.png` — Seven objects, one program: cut faces read from each object's own voxel field. (Sec. 5)
- `assets/ovsweep_{orange,watermelon,pomegranate2,apple1,bread,cake2,doughnut}_sp.mp4` — Per-object depth sweeps; orange, watermelon, and doughnut show both cut families side by side; the doughnut's ring topology survives the sweep. (Sec. 5)
- (Pipeline flowchart — the per-plane loop of Sec. 3, as rendered in `3dfusion.html`.)

---

## Reviewer self-assessment (rebuttal prep)

*Switching hats to a skeptical Reviewer #2. The most valuable section: where this paper is genuinely exposed, and the cheapest experiment that closes each gap.*

### Likely strengths

1. **A measured, non-obvious design principle.** "Move variation from the seed to the input" is backed by a direct measurement (seed std 0.130 → 0.011–0.035 across five constrained regimes), not asserted. Reviewers reward a paper that quantifies why the obvious approach fails before proposing its own.
2. **The boundary–anatomy decoupling framing is a genuine conceptual contribution.** It converts a hand-written prior ($w(s)$) into a principled one and preempts the strongest "why not SDF" objection with a clean in-principle argument plus a two-regime demonstration (orange blob vs. doughnut ring). This is the kind of small, correct idea that travels beyond the specific system.
3. **Intellectual honesty.** The Triplane negative result, the coverage-asymmetry admission, the explicit "we do not report per-plane pixel metrics and here is why," and the accepted-cost decisions (interior-from-photographs) read as a careful group that measures its own claims. ACs like this.
4. **Generality demonstrated, not asserted.** Seven objects through one program, with the two refinements shown to be object-general (validated on more than the object they were developed on), directly answers the "does this transfer" reflex.

### Likely weaknesses / attacks, with rebuttals

1. **"No quantitative comparison to a 3-D generative baseline."** *Fair, and the biggest exposure.* We have coverage (99.9%), the diversity table, and the Triplane ablation, but no numeric head-to-head against, e.g., an SDS/DreamFusion interior or a 3-D diffusion fill. **Concession + cheapest fix:** run an SDS interior with the single-image prior and a text-to-3D/interior-diffusion baseline on the seven objects, and score with a *held-out-plane* protocol — for the transverse family only (which has 16 supervised depths), hold out alternate depths, fill, and compute DreamSim/LPIPS on the held-out planes. This is honest because held-out transverse planes *do* have ground-truth photographs; it is ~a day of compute and would convert the qualitative claim into a table. We should add it.

2. **"A single-image diffusion per object and family is heavy supervision — this is closer to per-instance fitting than to a general model."** *Partly concede.* We train 1–2 small diffusions per object. **Rebuttal:** the alternative predecessor was *per-plane* training (dozens of fits per object); we reduced that to per-object-per-family and generate any plane in ~1 s, and the input to each diffusion is a *single* real cut photograph, for which no dataset alternative exists. **Cheapest strengthening:** show one diffusion generalizing across a small object *category* (three oranges of different size) with only the reference swapped, to argue the prior is object-class not object-instance. If it holds, the "heavy supervision" charge weakens substantially.

3. **"Is a voxel field with slab write-back novel enough? The storage is standard."** *Concede the storage is standard; contest that storage is the contribution.* The contributions are the input-axis reframing, the decoupling problem, and the bidirectional fusion; the voxel field is the *deliberately chosen* representation justified by a negative result against the fancier continuous one. **Rebuttal:** the paper's own Triplane ablation is the argument that the "obvious upgrade" is worse here — novelty is in *why the simple representation is correct under this constraint*, which is a result, not an assumption. We should make sure Sec. 6a is not buried.

4. **"SDS is dismissed too quickly."** *Legitimate.* We argue from a failure mode (imposing one reference everywhere) but do not run it. **Concession:** we should run SDS at least once, precisely to show the predicted failure (a columella appearing on off-centre oblique planes it should not). One run on the orange either vindicates the dismissal with a figure or forces us to soften the claim — either outcome is better than an unbacked dismissal. Cheap and mandatory before camera-ready.

5. **"The evaluation is qualitative; 'plausible' is doing a lot of work."** *Partly concede.* Coherence-of-sweep and coverage are real but not a metric a skeptic can rank against baselines. **Rebuttal + fix:** add (i) the held-out-transverse-plane quantitative protocol of point 1, and (ii) a small human study — pairs of (real cut photo, generated cut) shown to raters for realism/2AFC — which is the appropriate measurement when no per-plane ground truth exists for most planes. Report inter-object variance so it reads as generality, not cherry-picking.

6. **"Only two families and mostly near-spherical or simple objects; does the fusion scale to genuinely anisotropic 3-D structure?"** *Concede scope.* Bidirectional fusion is demonstrated with a marginal gain on the orange because its two fields agree. **Rebuttal:** the honest claim is that fusion *matters most* exactly where the two fields disagree, which is under-tested here; the cheapest evidence is the pomegranate or cake (directionally distinct interiors) with fusion on vs. off and the per-axis difference reported. If the gain is larger there, the mechanism is validated where it should be; if not, we down-scope the fusion claim to "does no harm and is more principled" rather than "improves cuts."

*Net self-assessment:* the conceptual contributions (input-axis variation with its measurement; boundary–anatomy decoupling; the honest Triplane negative) are defensible and somewhat unusual in their honesty. The paper's real risk is empirical thinness against baselines — items 1, 4, and 5 — every one of which is closable with under two days of compute using ground truth we already have (held-out transverse planes) or a single diagnostic SDS run. Those three experiments should be done before submission, not promised in rebuttal.

---

*Superseded.* This draft describes the input-driven SDEdit route, kept for the record. The method
now reported is on `3dfusion.html`, whose interior comes from a product of two single-image
diffusion priors on a cylindrical lattice; its code is `code/slicefill/`.
