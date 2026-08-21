"""Each object on a turntable, with the lattice's own axes drawn on it.

    TAG=ov2 python axisviews.py OUTDIR

Which way an object is stored is not something the paper argues; it is something to be looked at
and written down. This renders every object's exterior over a grid of azimuths and elevations and
draws the three lattice axes through its centre, so the answer to "which way is up" can be read off
and put in objects/<obj>.conf.

The camera is built here rather than taken from the pipeline's `cam_at`, on purpose: the axes drawn
have to be the lattice's own, and borrowing a camera that was defined through two frame changes
would leave which frame they are in open to argument. A look-at in the lattice frame has no such
question.

One sprite sheet per object, azimuth across and elevation down, so the viewer needs one request.
"""
import os
import sys

import numpy as np
import torch
import cv2
import nvdiffrast.torch as dr

sys.path.insert(0, "/workspace/ovoxel_native")
import ovcut
import ovnative as ON

W = "/workspace/ovoxel_native"
TAG = os.environ.get("TAG", "ov2")
RES = int(os.environ.get("AX_RES", "256"))
NAZ = int(os.environ.get("AX_NAZ", "24"))
ELS = [float(x) for x in os.environ.get("AX_ELS", "-25,0,25").split(",")]
OBJS = ["orange_sp", "watermelon_sp", "apple1_sp", "bread_sp", "cake2_sp",
        "pomegranate2_sp", "doughnut"]
AXES = [("x", (0.85, 0.15, 0.15)), ("y", (0.15, 0.65, 0.15)), ("z", (0.15, 0.35, 0.9))]


def look_at(eye, at, up, fov, aspect=1.0, near=0.01, far=100.0):
    """A row-vector mvp: clip = [x, 1] @ mvp, which is what the rasteriser here is fed."""
    f = at - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    V = np.eye(4)
    V[:3, :3] = np.stack([s, u, -f])
    V[:3, 3] = -V[:3, :3] @ eye
    t = 1.0 / np.tan(np.radians(fov) / 2)
    P = np.zeros((4, 4))
    P[0, 0] = t / aspect; P[1, 1] = t
    P[2, 2] = (far + near) / (near - far); P[2, 3] = 2 * far * near / (near - far)
    P[3, 2] = -1.0
    return (P @ V).T.astype(np.float32)


def project(mvp, pts):
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ mvp
    w = np.where(np.abs(h[:, 3:4]) < 1e-9, 1e-9, h[:, 3:4])
    n = h[:, :3] / w
    return np.stack([(n[:, 0] * 0.5 + 0.5) * RES, (0.5 - n[:, 1] * 0.5) * RES], 1), h[:, 3]


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    ON.FDG = ON._load_ovoxel()
    glctx = dr.RasterizeCudaContext(device="cuda")
    meta = {}
    for obj in OBJS:
        st = ovcut.load(obj, TAG)
        hc = float(st["hc"])
        org = np.asarray(st["org"], np.float64)
        lo = st["solid"].min(0).values.cpu().numpy() * hc + org
        hi = (st["solid"].max(0).values.cpu().numpy() + 1) * hc + org
        cen = (lo + hi) / 2
        rad = float(np.linalg.norm(hi - lo)) / 2
        L = rad * 1.35                                    # the axis arms, just past the object
        sheet = np.full((len(ELS) * RES, NAZ * RES, 3), 255, np.uint8)
        for ei, el in enumerate(ELS):
            for ai in range(NAZ):
                az = 360.0 * ai / NAZ
                e, a = np.radians(el), np.radians(az)
                d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                mvp = look_at(cen + d * rad * 3.2, cen, np.array([0.0, 0.0, 1.0]), 38.0)
                with torch.no_grad():
                    img, _ = ON.render_exterior(
                        st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"),
                        RES)[:2]
                fr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                fr = np.ascontiguousarray(fr[:, :, ::-1])          # cv2 draws in BGR
                ends = np.stack([cen] + [cen + L * np.eye(3)[k] for k in range(3)])
                uv, wclip = project(mvp, ends)
                for k, (nm, col) in enumerate(AXES):
                    if wclip[0] <= 0 or wclip[k + 1] <= 0:
                        continue
                    p0 = tuple(np.round(uv[0]).astype(int))
                    p1 = tuple(np.round(uv[k + 1]).astype(int))
                    bgr = tuple(int(255 * c) for c in col[::-1])
                    cv2.arrowedLine(fr, p0, p1, bgr, 2, cv2.LINE_AA, tipLength=0.12)
                    cv2.putText(fr, "+" + nm, (p1[0] + 3, p1[1] - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
                sheet[ei * RES:(ei + 1) * RES, ai * RES:(ai + 1) * RES] = fr
        f = os.path.join(outdir, f"axis_{obj}.jpg")
        cv2.imwrite(f, sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
        meta[obj] = dict(naz=NAZ, els=ELS, res=RES)
        print(f"  {obj}: {NAZ}x{len(ELS)} views -> {f} "
              f"({os.path.getsize(f) / 1e6:.1f} MB)", flush=True)
    import json
    json.dump(meta, open(os.path.join(outdir, "axis_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
