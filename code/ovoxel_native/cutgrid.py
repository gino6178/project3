"""Every object cut in every direction, so a cut can be paired with a photograph by eye.

    TAG=ov2 python cutgrid.py OUTDIR

Naming an axis has turned out to be the wrong thing to ask: the arrows are read one way and the
sections another, and two of the answers that came back disagreed with what the sections show. This
asks nothing to be named. The plane's normal is swept over a hemisphere -- a plane and its opposite
normal cut the same face, so half is all there is -- and each cut is drawn face-on. Put the
photograph beside it and turn until they are the same cut.

The plane always passes through the object's centre, so what changes between frames is only its
direction. The camera looks along the normal, which is what makes the face readable rather than
foreshortened.
"""
import json
import os
import sys

import numpy as np
import torch
import cv2
import nvdiffrast.torch as dr

sys.path.insert(0, "/workspace/ovoxel_native")
import axisviews as AV
import ovcut
import ovnative as ON

W = "/workspace/ovoxel_native"
TAG = os.environ.get("TAG", "ov2")
RES = int(os.environ.get("CG_RES", "224"))
NAZ = int(os.environ.get("CG_NAZ", "16"))
ELS = [float(x) for x in os.environ.get("CG_ELS", "0,22.5,45,67.5,90").split(",")]
OBJS = ["orange_sp", "watermelon_sp", "apple1_sp", "bread_sp", "cake2_sp",
        "pomegranate2_sp", "doughnut"]


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
        cen, rad = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
        sheet = np.full((len(ELS) * RES, NAZ * RES, 3), 255, np.uint8)
        for ei, el in enumerate(ELS):
            for ai in range(NAZ):
                az = 360.0 * ai / NAZ
                e, a = np.radians(el), np.radians(az)
                n = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                # the plane through the centre, and a camera on the normal looking back down it
                d = -float(cen @ n)
                up = np.array([0.0, 0.0, 1.0])
                if abs(float(up @ n)) > 0.95:
                    up = np.array([1.0, 0.0, 0.0])
                up = up - float(up @ n) * n
                up /= np.linalg.norm(up)
                mvp = AV.look_at(cen - n * rad * 3.2, cen, up, 38.0)
                with torch.no_grad():
                    img, _, _, _ = ON.render_section(
                        st, glctx, torch.as_tensor(mvp, dtype=torch.float32, device="cuda"),
                        torch.as_tensor(n, dtype=torch.float32, device="cuda"), d, RES)
                fr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                sheet[ei * RES:(ei + 1) * RES, ai * RES:(ai + 1) * RES] = fr[:, :, ::-1]
        f = os.path.join(outdir, f"cut_{obj}.jpg")
        cv2.imwrite(f, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        meta[obj] = dict(naz=NAZ, els=ELS, res=RES)
        print(f"  {obj}: {NAZ}x{len(ELS)} directions -> {f} "
              f"({os.path.getsize(f) / 1e6:.1f} MB)", flush=True)
    json.dump(meta, open(os.path.join(outdir, "cut_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
