# v0.1 experiments: canonical coordinates, 3-D truth, generated objects

What `3dfusion_v0.1.html` section 3.3 and sections 4.6-4.11 report, and the code that produced it.
The method itself is `code/slicefill/` and `code/src/` unchanged; everything here either generates
inputs, adds a lattice, or measures.

| file | |
|---|---|
| `synth2.py` | the solids generated whole: `wood1`-`wood4`, `honeycomb`, `hose`, `cake` (and `wood2x`, cross-specimen); inputs, truth, growth field G, declared centreline |
| `make_gsply.py` | a solid into the paper's carrier by its first route: a coloured Gaussian set, exterior coloured, interior ONE mean colour |
| `ov2grid.py`, `grid2ov.py` | the fitted O-Voxel grid to the object's grid and back, by the affine `make_gsply.py` placed it with |
| `x3dcart.py` | Algorithm 1 on a Cartesian lattice: three axis families, equal weights, 3-D low-pass |
| `x3dcurv2.py` | Algorithm 1 on a curvilinear lattice with parallel-transported frames about a declared centreline |
| `x3dcurv.py` | the translation-only curvilinear form (exact for an axis bent by shear) |
| `train_prior.py` | the Cartesian transverse prior (square canonical frames, the longitudinal recipe) |
| `eval_gt.py` | held-out faces against the truth: DreamSim, PSNR, SSIM, LPIPS, orientation; from G: ring alignment, grad-G orientation, ring centre, knot IoU |
| `eval_cross.py` | cross-specimen: DreamSim to the nearest held-out specimen on the same plane |
| `ovobj.sh`, `sched.sh` | one object through the paper's carrier (`code/run.sh`), the priors alongside it, every lift, evaluation; one worker per GPU |
| `showcase.sh` | the lift baked into the trained O-Voxel PLY (`slicefill/writeback.py`) and cut by `figures/draw_cuts.sh` |
| `gen_ai.py`, `prep_ai.py`, `glb2grid.py` | the generated log of section 4.10: cut faces and an exterior from an image model, the exterior to a mesh by TRELLIS.2, the mesh to a grid |
| `make_overview.py` | Figure 12 |
| `carrier.py` | the weaker single-level RGB carrier of section 4.11, kept only as that ablation |
| `synth.py`, `run_obj.sh`, `faces.py`, `score.py`, `render_cut.py` | the first, superseded procedural objects and their light-carrier route, kept for the record |
