"""Cross-family target reconciliation, on the O-Voxel state.

The pipeline's own `section_consistency.reconcile`, called rather than copied: the two families
of section targets are written without reference to each other, and where their planes meet they
describe the same cells. This makes them agree along that line before either is used.

It is off in the pipeline and has been since it was written, and the page reports only the
averaging mode, which failed -- 0.012 DreamSim against a control, because the average of a seed
and no seed is a smudge. `mode="copy"` is in the same file, is documented as the answer to that
failure, and has never been measured anywhere. Here the transverse family wins and the
longitudinal adopts, which is the right way round: equation (27)'s solved phases are applied to
the transverse family only -- `sds_demo` applies them when `view_cut == "horizontal"` and not
otherwise -- so the family that is currently unaligned is the one that should be doing the
adopting.

Two shims and nothing else. `reconcile` projects through a camera object; this representation
carries a bare 4x4 mvp and a resolution, so `Cam` presents one as the other. And the pipeline's
planes live in a frame its cameras do not, which is what `to_world` is for; here both are the
same frame, so it is None, passed explicitly to say so rather than left to the default.
"""
import sys

sys.path.insert(0, "/workspace/rebuild/project3/code/src")
import section_consistency as _sc                                      # noqa: E402


class Cam:
    """What `section_consistency.project` asks of a camera, and nothing more."""

    def __init__(self, mvp, res):
        self.full_proj_transform = mvp
        self.image_width = res
        self.image_height = res


def reconcile(tgt_h, mvp_h, n_h, d_h, tgt_v, mvp_v, n_v, d_v, points, res,
              band=None, weight=1.0, mode="copy"):
    """Make one longitudinal target agree with one transverse target along their shared line.

    Both targets are (3, H, W) and are written in place. The transverse one is passed as
    `tgt_a`, so under `copy` it is the one that wins and the longitudinal adopts.
    """
    pa = [float(n_h[0]), float(n_h[1]), float(n_h[2]), float(d_h)]
    pb = [float(n_v[0]), float(n_v[1]), float(n_v[2]), float(d_v)]
    return _sc.reconcile(tgt_h, Cam(mvp_h, res), pa, tgt_v, Cam(mvp_v, res), pb, points,
                         to_world=None, band=band, weight=weight, mode=mode)
