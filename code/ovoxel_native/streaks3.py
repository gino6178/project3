"""Are the dark chords the longitudinal training planes printing into a transverse cut?

Two planes meet in a line. The ten longitudinal training planes are vertical; the evaluation cut is
transverse; so each longitudinal plane leaves exactly one straight line on the evaluation cut. If
that is what these are, the lines are computable in closed form and can be drawn on top of the
render without fitting anything.

The angular test that found no 18-degree peak binned about the centre. `generate_plane_center` puts
the longitudinal plane at centers[int(0.5*23)] = centers[11] of 24, which is half a step off the
middle, so every one of these lines is a CHORD and not a diameter -- and a chord contributes to
every angular bin, which is exactly the case that test is blind to.
"""
import os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON, nvdiffrast.torch as dr, anchor

W = "/workspace/ovoxel_native"; OUT = W + "/out/streaks"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"; RES = 512
ROUTE = os.environ.get("ROUTE", "1"); ARM = os.environ.get("ARM", "r1_pin"); PL = 0

st = torch.load(f"{W}/state_r{ROUTE}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel(); glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_mv.npz" if ROUTE == "1" else f"{W}/cams_mv_r2.npz")
mvp_t = torch.as_tensor(C["eh_mvp"][PL], dtype=torch.float32, device=dev)
mvp = C["eh_mvp"][PL].astype(np.float64)
n1 = C["eh_planes"][PL, :3].astype(np.float64); d1 = float(C["eh_planes"][PL, 3])

P = torch.load(f"{W}/{ARM}/params.pt", map_location=dev)
di = anchor.ColourDecoder(len(st["interior"])).to(dev); di.load_state_dict(P["dec_i"])
ds = anchor.ColourDecoder(len(st["surf_rgb"])).to(dev); ds.load_state_dict(P["dec_s"])
init_i = st["interior"].clone()
with torch.no_grad():
    st["interior"], st["surf_rgb"] = di(), ds()
st["dual_v"], st["split_w"] = P["dual_v"].to(dev), P["split_w"].to(dev)
img, al, _, _ = ON.render_section(st, glctx, mvp_t,
                                  torch.as_tensor(n1, dtype=torch.float32, device=dev), d1, RES)
A = img.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
with torch.no_grad():
    st["interior"] = init_i
im0, _, _, _ = ON.render_section(st, glctx, mvp_t,
                                 torch.as_tensor(n1, dtype=torch.float32, device=dev), d1, RES)
B = im0.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
diff = np.abs(A - B).mean(2)


def to_px(p):
    v = np.append(p, 1.0) @ mvp
    q = v[:2] / v[3]
    return np.array([(q[0] * 0.5 + 0.5) * RES, (0.5 - q[1] * 0.5) * RES])


over = (A[:, :, ::-1] * 255).astype(np.uint8).copy()
print(f"arm {ARM}: intersections of the {len(C['v_planes'])} longitudinal training planes "
      f"with held-out transverse cut {PL}\n")
hits = []
for i in range(len(C["v_planes"])):
    n2 = C["v_planes"][i, :3].astype(np.float64); d2 = float(C["v_planes"][i, 3])
    dvec = np.cross(n1, n2)
    if np.linalg.norm(dvec) < 1e-9:
        continue
    dvec /= np.linalg.norm(dvec)
    # a point on both planes: least squares on the 2x3 system
    Amat = np.stack([n1, n2]); b = -np.array([d1, d2])
    p0 = np.linalg.lstsq(Amat, b, rcond=None)[0]
    pts = np.stack([to_px(p0 + t * dvec) for t in np.linspace(-0.5, 0.5, 400)])
    ok = (pts[:, 0] > 1) & (pts[:, 0] < RES - 2) & (pts[:, 1] > 1) & (pts[:, 1] < RES - 2)
    if ok.sum() < 10:
        continue
    q = pts[ok].astype(int)
    val = diff[q[:, 1], q[:, 0]]
    # distance from the disc centre, in pixels, for the chord/diameter question
    fgm = (np.abs(A - 1).max(2) > 0.03)
    ys, xs = np.nonzero(fgm); cy, cx = ys.mean(), xs.mean()
    dist = np.abs(np.cross(pts[ok][-1] - pts[ok][0], np.array([cx, cy]) - pts[ok][0])) \
        / (np.linalg.norm(pts[ok][-1] - pts[ok][0]) + 1e-9)
    hits.append((i, float(val.mean()), float(dist)))
    cv2.polylines(over, [q.reshape(-1, 1, 2)], False, (0, 0, 255), 1)
    ang = np.degrees(np.arctan2(-(pts[ok][-1][1] - pts[ok][0][1]),
                                pts[ok][0][0] - pts[ok][-1][0])) % 180
    print(f"  v{i:<2} az {C['v_az'][i]:>5.1f}deg -> image line at {ang:6.1f}deg, "
          f"{dist:6.1f}px from the centre, |trained-init| on it {val.mean():.4f}")

base = float(diff[(np.abs(A - 1).max(2) > 0.03)].mean())
print(f"\n  |trained - init| over the whole cut face: {base:.4f}")
print(f"  mean on the ten predicted lines:           {np.mean([h[1] for h in hits]):.4f}  "
      f"({np.mean([h[1] for h in hits])/base:.2f}x)")
cv2.imwrite(f"{OUT}/overlay_{ARM}.png", over)
cv2.imwrite(f"{OUT}/diff_{ARM}.png", (diff / max(diff.max(), 1e-6) * 255).astype(np.uint8))
print(f"  -> {OUT}/overlay_{ARM}.png")
print("STREAKS3_OK")
