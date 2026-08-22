"""Rasterise many cut faces in one pass.

`render_section` draws one plane per call: one `cut_polygons`, one `sample_interior`, one
`dr.rasterize` whose batch dimension is 1. Drawing N planes therefore costs N launches of
everything, which is why the training step could only afford four planes and why its gradient was
decided by four planes.

nvdiffrast's range mode takes one vertex array, one triangle array and a table saying which block
of triangles belongs to which image. Every plane has different geometry, which rules out the
ordinary batch mode, but range mode does not care: the planes are concatenated, each with its own
vertices already through its own matrix, and one call rasterises all of them.

Only the cut faces. The exterior behind the plane is a large shared mesh clipped differently for
every plane, and duplicating it per plane costs more memory than the batch saves -- so the exterior
stays on the single-plane path, and `selftest` compares against `render_section(exterior=False)`,
which is the same picture this draws.
"""
import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import ovnative as ON


def render_batch(st, glctx, mvps, ns, ds, res, bg=1.0, aa=True):
    """N cut faces, one rasterise. Returns (N,3,res,res) and (N,1,res,res), as render_section does."""
    dev = st["interior"].device
    pos, tri, attr, ranges, voff, toff = [], [], [], [], 0, 0
    for i in range(len(ds)):
        P, T, _ = ON.cut_polygons(st, ns[i], float(ds[i]), device=dev)
        C = ON.sample_interior(st, P)
        ph = torch.cat([P, torch.ones_like(P[:, :1])], 1) @ mvps[i]
        pos.append(ph)
        tri.append(T.int() + voff)
        attr.append(torch.cat([P, C.clamp(0, 1)], 1))
        ranges.append([toff, len(T)])
        voff += len(P); toff += len(T)
    pos = torch.cat(pos).contiguous()
    tri = torch.cat(tri).contiguous().int()
    attr = torch.cat(attr).contiguous()
    rng = torch.as_tensor(ranges, dtype=torch.int32)          # ranges live on the host

    rast, _ = dr.rasterize(glctx, pos, tri, resolution=[res, res], ranges=rng)
    it, _ = dr.interpolate(attr, rast, tri)
    p3, vcol = it[..., :3], it[..., 3:6]
    img = vcol
    if ON.DEFERRED:
        hit = rast[..., 3:4] > 0
        if bool(hit.any()):
            idx = hit[..., 0].nonzero(as_tuple=True)
            img = img.clone()
            img[idx] = ON.sample_interior(st, p3[idx]).clamp(0, 1)
    if aa:
        img = dr.antialias(img.contiguous(), rast.contiguous(), pos, tri)
    alpha = (rast[..., 3:4] > 0).float()
    img = img * alpha + bg * (1 - alpha)
    return img.permute(0, 3, 1, 2), alpha.permute(0, 3, 1, 2)


def selftest(st, glctx, mvps, ns, ds, res=512):
    """The batch must draw exactly what the one-at-a-time path draws. Not nearly: exactly."""
    bi, ba = render_batch(st, glctx, mvps, ns, ds, res)
    worst_i = worst_a = 0.
    for i in range(len(ds)):
        im, al, _, _ = ON.render_section(st, glctx, mvps[i], ns[i], float(ds[i]), res,
                                         exterior=False)
        worst_i = max(worst_i, float((im - bi[i]).abs().max()))
        worst_a = max(worst_a, float((al - ba[i]).abs().max()))
    print(f"  batch of {len(ds)} against {len(ds)} single renders: "
          f"largest pixel difference {worst_i:.3e}, alpha {worst_a:.3e}")
    return worst_i, worst_a
