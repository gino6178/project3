"""Is there a peak at m = P, or is 1.39 a point on a monotone decay?

Global median is the wrong baseline for a peak test: the spectrum falls from low m, so any m below
the crossing sits above the median whether or not it is distinguished. Local prominence -- power at
m against the median of a window excluding m -- answers the question the comb argument poses.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W, dev, RES = "/workspace/ovoxel_native", "cuda", 512
ON.FDG = ON._load_ovoxel(); glctx = dr.RasterizeCudaContext(device=dev)
NA, NR = 720, 160
for OBJ, run in (("orange_sp", "s_v2_orange_sp"), ("watermelon_sp", "s_v2_watermelon_sp")):
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
        with torch.no_grad(): st["interior"], st["surf_rgb"] = di(), ds()
    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0]); P = len(C["v_planes"])
    acc = None
    for i in range(H_HI - H_LO):
        with torch.no_grad():
            img, al, _, _ = ON.render_section(
                st, glctx, torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev),
                torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev),
                float(C["h_planes"][H_LO + i, 3]), RES, exterior=False)
        g = img.mean(0).cpu().numpy(); m = (al[0] > 0.5).cpu().numpy()
        ys, xs = np.where(m)
        if len(ys) < 100: continue
        cy, cx = ys.mean(), xs.mean(); rmax = 0.85 * np.sqrt(m.sum() / np.pi)
        th = np.linspace(0, 2*np.pi, NA, endpoint=False); rr = np.linspace(.25*rmax, rmax, NR)
        Y = np.clip((cy + rr[:, None]*np.sin(th)[None]).astype(int), 0, RES-1)
        X = np.clip((cx + rr[:, None]*np.cos(th)[None]).astype(int), 0, RES-1)
        pol = g[Y, X]; pol = pol - pol.mean(1, keepdims=True)
        f = (np.abs(np.fft.rfft(pol, axis=1))**2).mean(0)
        acc = f if acc is None else acc + f
    sp = acc / np.median(acc[1:60])
    win = [k for k in range(P-6, P+7) if 0 < k < len(sp) and k != P]
    loc = float(sp[P] / np.median(sp[win]))
    best = int(np.argmax([sp[k] / np.median(sp[[j for j in range(k-6, k+7)
                        if 0 < j < len(sp) and j != k]]) for k in range(7, 55)]) + 7)
    bp = float(sp[best] / np.median(sp[[j for j in range(best-6, best+7)
                                        if 0 < j < len(sp) and j != best]]))
    print(f"{OBJ:16s} P={P:2d}  global {sp[P]:.2f}x median  |  LOCAL prominence at P {loc:.2f}x"
          f"  |  most prominent m in 7..54: m={best} at {bp:.2f}x")
