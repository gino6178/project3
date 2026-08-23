"""Are the stripes the seams between the longitudinal planes?

fieldreg.py describes the defect as columns along the polar axis made of disagreement across it.
Each longitudinal plane paints a whole meridian half-plane with its own photograph, its neighbour
ten degrees away paints another, and where two wedges meet the field has to choose -- which leaves
a discontinuity running along the axis. Seen in a transverse cut that is a radial spoke, and there
should be exactly as many spokes as there are longitudinal planes.

So: take a transverse render, resample it in polar coordinates about the axis, and take the power
spectrum around the angle. A peak at the number of longitudinal planes is the seam; a flat spectrum
means the stripes come from somewhere else.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import anchor
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "watermelon_sp")
RUN = os.environ.get("SP_RUN", f"s_rs_{OBJ}")
RES = int(os.environ.get("RES", "512"))
dev = "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NV = len(C["v_planes"])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
p = torch.load(f"{W}/{RUN}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
if "dec_i" in p:
    w = p["dec_i"]["stage1.0.weight"].shape[0]
    n = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
    anchor.W_HID, anchor.N_HID = w, n
    di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
    di.load_state_dict(p["dec_i"])
    ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
    ds.load_state_dict(p["dec_s"])
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()

NA, NR = 360, 64


def spectrum(img, mask):
    ys, xs = mask.nonzero(as_tuple=True)
    cy, cx = float(ys.float().mean()), float(xs.float().mean())
    rmax = float(((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).sqrt().max())
    th = torch.linspace(0, 2 * np.pi, NA + 1, device=dev)[:-1]
    rr = torch.linspace(0.25, 0.8, NR, device=dev) * rmax
    yy = cy + rr[None, :] * torch.sin(th)[:, None]
    xx = cx + rr[None, :] * torch.cos(th)[:, None]
    g = torch.stack([xx / (img.shape[-1] - 1) * 2 - 1, yy / (img.shape[-2] - 1) * 2 - 1], -1)
    s = torch.nn.functional.grid_sample(img[None], g[None], align_corners=True)[0].mean(0)
    s = s - s.mean(0, keepdim=True)                    # per radius, so only the angle is left
    f = torch.fft.rfft(s, dim=0).abs() ** 2
    return f.mean(1)                                   # power at each angular frequency


with torch.no_grad():
    acc = None
    for i in range(H_LO, H_HI):
        img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, float(hd[i]), RES, exterior=False)
        f = spectrum(img, al[0] > 0.5)
        acc = f if acc is None else acc + f
    acc = acc / (H_HI - H_LO)

base = float(acc[1:60].median())
print(f"{OBJ} ({RUN}): angular power of the transverse renders, {NV} longitudinal planes")
print(f"  {'frequency':>10}{'power / median':>16}")
for k in sorted({NV, 2 * NV, NV - 1, NV + 1, 6, 12, 24, 36}):
    if 1 <= k < len(acc):
        mark = "  <- the number of longitudinal planes" if k == NV else (
            "  <- twice it" if k == 2 * NV else "")
        print(f"  {k:>10}{float(acc[k]) / base:>16.2f}{mark}")
top = int(acc[1:60].argmax()) + 1
print(f"  strongest between 1 and 59: frequency {top}, {float(acc[top]) / base:.2f} of the median")
