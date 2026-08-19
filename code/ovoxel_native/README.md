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

Logs and per-arm `eval_final` renders are beside the code; `out/` holds the figures.
