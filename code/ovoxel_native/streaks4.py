"""Find the dark chords directly, then ask what they are in 3-D.

Everything so far has been elimination: not the tessellation (a constant field renders exactly
flat), not antialias, not resolution, not the longitudinal planes (the change on their predicted
intersections is 0.80x the average, below it). So: detect the lines in the image and back out what
they correspond to, rather than guessing at another candidate.

The radial membranes are the confound -- they are real structure and they are lines through the
centre. They are removed by subtracting an angular median filter, which keeps a spoke (constant in
angle over its length) and leaves a chord (crossing many angles).
"""
import os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON, nvdiffrast.torch as dr, anchor

W = "/workspace/ovoxel_native"; OUT = W + "/out/streaks"
dev = "cuda"; RES = 512
ROUTE = os.environ.get("ROUTE", "1"); ARM = os.environ.get("ARM", "r1_pin"); PL = 0

st = torch.load(f"{W}/state_r{ROUTE}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel(); glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_mv.npz" if ROUTE == "1" else f"{W}/cams_mv_r2.npz")
mvp_np = C["eh_mvp"][PL].astype(np.float64)
n1 = C["eh_planes"][PL, :3].astype(np.float64); d1 = float(C["eh_planes"][PL, 3])
P = torch.load(f"{W}/{ARM}/params.pt", map_location=dev)
di = anchor.ColourDecoder(len(st["interior"])).to(dev); di.load_state_dict(P["dec_i"])
ds = anchor.ColourDecoder(len(st["surf_rgb"])).to(dev); ds.load_state_dict(P["dec_s"])
with torch.no_grad():
    st["interior"], st["surf_rgb"] = di(), ds()
st["dual_v"], st["split_w"] = P["dual_v"].to(dev), P["split_w"].to(dev)
img, _, _, _ = ON.render_section(st, glctx, torch.as_tensor(C["eh_mvp"][PL], dtype=torch.float32, device=dev),
                                 torch.as_tensor(n1, dtype=torch.float32, device=dev), d1, RES)
A = img.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
g = A.mean(2)
fg = (np.abs(A - 1).max(2) > 0.03)
ys, xs = np.nonzero(fg); cy, cx = ys.mean(), xs.mean()

# polar resample, median-filter along radius at fixed angle -> the spokes; subtract to leave chords
NA, NR = 720, 256
R = np.sqrt(fg.sum() / np.pi) * 0.92
th = np.linspace(0, 2 * np.pi, NA, endpoint=False)
rr = np.linspace(0, R, NR)
TH, RR = np.meshgrid(th, rr, indexing="ij")
mx = (cx + RR * np.cos(TH)).astype(np.float32)
my = (cy + RR * np.sin(TH)).astype(np.float32)
pol = cv2.remap(g.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=1.0)
spokes = cv2.medianBlur(pol.astype(np.float32), 1)
# a spoke is constant along radius at a given angle: model it by the per-angle median
model = np.median(pol, axis=1, keepdims=True) * np.ones((1, NR))
# and by a smooth radial profile
model = model + (np.median(pol, axis=0, keepdims=True) - np.median(pol))
res_pol = pol - model
back = cv2.remap(res_pol.astype(np.float32),
                 ((np.arctan2(np.mgrid[0:RES, 0:RES][0] - cy,
                              np.mgrid[0:RES, 0:RES][1] - cx) % (2 * np.pi)) / (2 * np.pi) * NA).astype(np.float32),
                 (np.hypot(np.mgrid[0:RES, 0:RES][0] - cy,
                           np.mgrid[0:RES, 0:RES][1] - cx) / R * NR).astype(np.float32),
                 cv2.INTER_LINEAR, borderValue=0.0)
e = cv2.erode(fg.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
res = np.where(e, back, 0.0)
cv2.imwrite(f"{OUT}/chords_{ARM}.png",
            (np.clip(-res / max(-res.min(), 1e-6), 0, 1) * 255).astype(np.uint8))

dark = np.clip(-res, 0, None)
thr = dark[e].mean() + 3 * dark[e].std()
mask = ((dark > thr) & e).astype(np.uint8) * 255
lines = cv2.HoughLines(mask, 1, np.pi / 360, threshold=int(os.environ.get("HTHR", "55")))
print(f"arm {ARM}: dark linear residue after removing the radial spokes")
print(f"  threshold {thr:.4f}, {int(mask.sum()/255):,} pixels above it")
if lines is None:
    print("  no lines found at this threshold")
    sys.exit(0)


def px_to_world(px, py):
    """A pixel on the cut plane, back to 3-D: the ray through it, met with the plane."""
    ndc = np.array([px / RES * 2 - 1, 1 - py / RES * 2])
    Minv = np.linalg.inv(mvp_np)
    a = np.array([ndc[0], ndc[1], -1.0, 1.0]) @ Minv
    b = np.array([ndc[0], ndc[1], 1.0, 1.0]) @ Minv
    a, b = a[:3] / a[3], b[:3] / b[3]
    u = b - a
    t = -(n1 @ a + d1) / (n1 @ u)
    return a + t * u


print(f"  {'rho':>7} {'theta':>7}  {'offset from centre':>18}  {'3-D direction (lattice frame)':>32}")
seen = []
for L in lines[:12]:
    rho, tt = L[0]
    if any(abs(tt - s) < np.radians(4) and abs(rho - r) < 8 for r, s in seen):
        continue
    seen.append((rho, tt))
    a0, b0 = np.cos(tt), np.sin(tt)
    p0 = np.array([a0 * rho, b0 * rho])
    dirv = np.array([-b0, a0])
    q1, q2 = p0 - 250 * dirv, p0 + 250 * dirv
    w1, w2 = px_to_world(*q1), px_to_world(*q2)
    dw = w2 - w1; dw /= np.linalg.norm(dw)
    off = abs(a0 * cx + b0 * cy - rho)
    print(f"  {rho:>7.1f} {np.degrees(tt):>6.1f}d  {off:>15.1f}px  "
          f"[{dw[0]:+.3f} {dw[1]:+.3f} {dw[2]:+.3f}]")
print("STREAKS4_OK")
