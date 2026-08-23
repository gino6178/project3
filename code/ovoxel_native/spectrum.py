"""The angular power spectrum of a transverse render, against the comb the longitudinal family is.

The sampling argument predicts a peak at m = P and nothing distinguished elsewhere. This measures
it: resample the render in polar coordinates about the axis, take the power spectrum around theta,
average over radius, and normalise by the median. Two objects, and the baseline against the arm
that band-limits the reconstruction.
"""
import os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON, anchor
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
RES = 512
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
NA, NR = 720, 160


def spectrum(OBJ, run):
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
    C = np.load(f"{W}/cams_{OBJ}_v2.npz")
    p = torch.load(f"{W}/{run}/params.pt", map_location=dev)
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
    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
    acc = None
    for i in range(H_HI - H_LO):
        mv = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
        nn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
        with torch.no_grad():
            img, al, _, _ = ON.render_section(st, glctx, mv, nn,
                                              float(C["h_planes"][H_LO + i, 3]), RES,
                                              exterior=False)
        g = img.mean(0).cpu().numpy()
        m = (al[0] > 0.5).cpu().numpy()
        ys, xs = np.where(m)
        if len(ys) < 100:
            continue
        cy, cx = ys.mean(), xs.mean()
        rmax = 0.85 * np.sqrt(m.sum() / np.pi)
        th = np.linspace(0, 2 * np.pi, NA, endpoint=False)
        rr = np.linspace(0.25 * rmax, rmax, NR)
        Y = np.clip((cy + rr[:, None] * np.sin(th)[None]).astype(int), 0, RES - 1)
        X = np.clip((cx + rr[:, None] * np.cos(th)[None]).astype(int), 0, RES - 1)
        pol = g[Y, X]
        pol = pol - pol.mean(1, keepdims=True)
        f = np.abs(np.fft.rfft(pol, axis=1)) ** 2
        acc = f.mean(0) if acc is None else acc + f.mean(0)
    acc /= max(H_HI - H_LO, 1)
    return acc / np.median(acc[1:60]), len(C["v_planes"])


fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
for a_, (OBJ, base, alt, nm) in zip(ax, (("orange_sp", "s_v2_orange_sp", "q_cub_orange_sp", "orange"),
                                         ("watermelon_sp", "s_v2_watermelon_sp", None, "watermelon"))):
    sp, P = spectrum(OBJ, base)
    m = np.arange(len(sp))
    a_.plot(m[1:60], sp[1:60], lw=1.4, c="#2c6fbb", label="baseline")
    if alt and os.path.exists(f"{W}/{alt}/params.pt"):
        sp2, _ = spectrum(OBJ, alt)
        a_.plot(m[1:60], sp2[1:60], lw=1.4, c="#c0392b", label="cubic B-spline")
    a_.axvline(P, color="#888", ls="--", lw=1.0)
    a_.annotate(f"$m=P={P}$", (P, a_.get_ylim()[1] * 0.82), fontsize=10, color="#555",
                xytext=(6, 0), textcoords="offset points")
    a_.axhline(1.0, color="#bbb", lw=0.8)
    a_.set_xlabel("angular frequency $m$"); a_.set_title(nm, fontsize=11)
    a_.set_ylabel("power / median"); a_.legend(fontsize=9, frameon=False)
    a_.set_yscale("log")
fig.tight_layout()
fig.savefig(f"{W}/spectrum.jpg", dpi=118, bbox_inches="tight")
print("spectrum.jpg")
