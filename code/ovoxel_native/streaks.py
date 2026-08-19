"""What the dark diagonal lines in a cut face actually are.

The supervision was ruled out already: binning angular energy at multiples of 18 degrees found no
peak, and the same lines appear in the free-per-cell-RGB arm, which predates the decoder. So this
takes the renderer apart instead, one piece at a time, on ONE held-out transverse plane:

    full            what the evaluation renders
    no-aa           dr.antialias off
    no-exterior     the cut face alone, nothing of the surface behind the plane
    flat-field      `interior` replaced by a single constant, so the cut face carries no signal
                    at all and anything left is not the trained state
    flat + no-aa    both

and at init as well as trained, so a property of the representation can be told from a property of
what was learned.

Detection is a Radon transform, not angular binning about the centre: a line that does not pass
through the middle contributes to every angular bin equally and hides. The residual is the
foreground minus its own radial mean, so the radial falloff of the fruit does not count as
structure.
"""
import os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON
import nvdiffrast.torch as dr
import anchor
from scipy import ndimage

W = "/workspace/ovoxel_native"
OUT = W + "/out/streaks"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
RES = int(os.environ.get("RES", "512"))
ROUTE = os.environ.get("ROUTE", "1")
ARM = os.environ.get("ARM", "r1_pin")
PLANE = int(os.environ.get("PLANE", "0"))

st = torch.load(f"{W}/state_r{ROUTE}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_mv.npz" if ROUTE == "1" else f"{W}/cams_mv_r2.npz")
mvp = torch.as_tensor(C["eh_mvp"][PLANE], dtype=torch.float32, device=dev)
n = torch.as_tensor(C["eh_planes"][PLANE, :3], dtype=torch.float32, device=dev)
d = float(C["eh_planes"][PLANE, 3])

init_i = st["interior"].clone()
init_s = st["surf_rgb"].clone()

# the trained state, decoded back out of the parameters that were saved
P = torch.load(f"{W}/{ARM}/params.pt", map_location=dev)
dec_i = anchor.ColourDecoder(len(init_i)).to(dev)
dec_s = anchor.ColourDecoder(len(init_s)).to(dev)
dec_i.load_state_dict(P["dec_i"]); dec_s.load_state_dict(P["dec_s"])
with torch.no_grad():
    tr_i, tr_s = dec_i(), dec_s()
tr_dv, tr_sw = P["dual_v"].to(dev), P["split_w"].to(dev)


def radon_peaks(img, mask, nb=180):
    """Orientation spectrum of the residual, by Radon transform. Returns (angles, energy)."""
    g = img.mean(2).astype(np.float32)
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # remove the radial mean, so the fruit's own falloff is not structure
    rb = np.clip((r / max(r[mask].max(), 1) * 64).astype(int), 0, 64)
    prof = np.zeros(65)
    for b in range(65):
        m = mask & (rb == b)
        prof[b] = g[m].mean() if m.sum() > 20 else np.nan
    prof = np.interp(np.arange(65), np.flatnonzero(~np.isnan(prof)), prof[~np.isnan(prof)])
    res = np.where(mask, g - prof[rb], 0.0)
    # erode so the rim is not the signal
    e = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    res = np.where(e, res, 0.0)
    en = []
    for a in range(nb):
        rot = ndimage.rotate(res, a, reshape=False, order=1, mode="constant")
        col = rot.sum(0)                       # integrate along the line direction
        en.append(float(np.abs(col - col.mean()).mean()))
    en = np.array(en)
    return en, res, e


def show(tag, img, alpha):
    a = img.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    m = (np.abs(a - 1).max(2) > 0.03)
    en, res, e = radon_peaks(a, m)
    z = (en - en.mean()) / (en.std() + 1e-9)
    top = np.argsort(-z)[:4]
    cv2.imwrite(f"{OUT}/{tag}.png", (a[:, :, ::-1] * 255).astype(np.uint8))
    v = np.abs(res); v = v / max(v.max(), 1e-6)
    cv2.imwrite(f"{OUT}/{tag}_res.png", (v * 255).astype(np.uint8))
    print(f"  {tag:<28} residual rms {res[e].std():.5f}   "
          f"top orientations {[f'{int(t)}deg z={z[t]:+.1f}' for t in top]}")
    return en, float(res[e].std())


def render(exterior=True, aa=True, flat=False, trained=True, res=RES):
    st["interior"] = (torch.full_like(init_i, 0.6) if flat else (tr_i if trained else init_i))
    st["surf_rgb"] = tr_s if trained else init_s
    st["dual_v"] = tr_dv if trained else st["dual_v"]
    st["split_w"] = tr_sw if trained else st["split_w"]
    return ON.render_section(st, glctx, mvp, n, d, res, exterior=exterior, aa=aa)


print(f"arm {ARM}, route {ROUTE}, held-out transverse plane {PLANE}, {RES}x{RES}\n")
print("trained state:")
E = {}
for tag, kw in [("full", {}), ("no-aa", dict(aa=False)),
                ("no-exterior", dict(exterior=False)),
                ("flat-field", dict(flat=True)),
                ("flat + no-aa", dict(flat=True, aa=False)),
                ("flat + no-ext + no-aa", dict(flat=True, exterior=False, aa=False))]:
    img, al, _, _ = render(**kw)
    E[tag] = show(tag.replace(" ", "").replace("+", "_"), img, al)

print("\ninitial state (nothing trained):")
for tag, kw in [("init full", {}), ("init flat-field", dict(flat=True))]:
    img, al, _, _ = render(trained=False, **kw)
    E[tag] = show(tag.replace(" ", "_"), img, al)

print("\nresolution, full trained render:")
for r in (256, 512, 1024):
    img, al, _, _ = render(res=r)
    E[f"res{r}"] = show(f"res{r}", img, al)

np.savez(f"{OUT}/spectra.npz", **{k: v[0] for k, v in E.items()})
print("\nresidual rms, ranked:")
for k, v in sorted(E.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k:<28} {v[1]:.5f}")
print("STREAKS_OK")
