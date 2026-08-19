"""Step 4: train the O-Voxel-native representation against the orange's six transverse
photographs, and render the pipeline's own held-out cuts before and after."""
import os, sys, time, json
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON
import nvdiffrast.torch as dr
import section_match as sm

LAT = "/workspace/rebuild/worktree/build_orange/lattice"
REFDIR = "/workspace/rebuild/worktree/secref_orraw_hsep"
CAMS = "/workspace/ovoxel_native/cams_orange.npz"
OUT = os.environ.get("OUT", "/workspace/ovoxel_native/run1")
ITERS = int(os.environ.get("ITERS", "300"))
RES = int(os.environ.get("RES", "512"))
LR_INT = float(os.environ.get("LR_INT", "0.02"))
LR_SURF = float(os.environ.get("LR_SURF", "0.02"))
LR_GEO = float(os.environ.get("LR_GEO", "1e-3"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT + "/eval_init", exist_ok=True)
os.makedirs(OUT + "/eval_final", exist_ok=True)
dev = "cuda"
torch.manual_seed(0); np.random.seed(0)

st = torch.load("/workspace/ovoxel_native/state_orange.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
C = np.load(CAMS)

def plane_in_pos_frame(i):
    a = C[f"c{i}_affine"]; A, t = a[:3], a[3]
    nt = C[f"c{i}_plane_t"][:3]; dt = float(C[f"c{i}_plane_t"][3])
    npos = np.linalg.solve(A, nt); dpos = dt - float(t @ npos)
    s = np.linalg.norm(npos); return npos / s, dpos / s

EV = [plane_in_pos_frame(i) for i in range(6)]
for i in range(1, 6):
    assert np.allclose(C["c0_fp"], C[f"c{i}_fp"]), "the six horizontal cuts share one camera"
MVP = torch.as_tensor(C["c0_fp"], dtype=torch.float32, device=dev)
NRM = torch.as_tensor(EV[0][0], dtype=torch.float32, device=dev)
d_ev = np.array([d for _, d in EV])
print(f"held-out cut depths d = {np.round(d_ev,4)}")

# training planes: the same normal, on a grid over the same band, staying clear of the six
band = (d_ev.min() - 0.04, d_ev.max() + 0.04)
cand = np.linspace(band[0], band[1], 21)
train_d = [float(x) for x in cand if np.abs(x - d_ev).min() > 0.004]
print(f"training depths: {len(train_d)} planes in [{band[0]:.3f}, {band[1]:.3f}], "
      f"nearest to a held-out one {min(np.abs(np.array(train_d)[:,None]-d_ev[None]).min(1)):.4f}")

refs = [cv2.imread(os.path.join(REFDIR, p))[:, :, ::-1].astype(np.float32) / 255.0
        for p in sorted(os.listdir(REFDIR)) if p.endswith(".png")]
print(f"{len(refs)} reference photographs, {refs[0].shape}")

glctx = dr.RasterizeCudaContext(device=dev)

def render(d):
    return ON.render_section(st, glctx, MVP, NRM, float(d), RES)

def dump(tag, folder):
    with torch.no_grad():
        for i, (n, d) in enumerate(EV):
            img, _, _, _ = render(d)
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            cv2.imwrite(f"{folder}/rh{i}_init_0.png", (a[:, :, ::-1] * 255).astype(np.uint8))
    print(f"  {tag} -> {folder}")

dump("init", OUT + "/eval_init")

for k in ("dual_v", "split_w", "surf_rgb", "interior"):
    st[k] = st[k].detach().clone().requires_grad_(True)
opt = torch.optim.Adam([
    dict(params=[st["interior"]], lr=LR_INT),
    dict(params=[st["surf_rgb"]], lr=LR_SURF),
    dict(params=[st["dual_v"], st["split_w"]], lr=LR_GEO),
])
dv0 = st["dual_v"].detach().clone()

# a fixed probe, so the curve is one number measured the same way rather than the training
# loss on whichever plane and photograph came up
def probe():
    with torch.no_grad():
        tot = 0.0
        for i, (n, d) in enumerate(EV):
            im, al, _, _ = render(d)
            tg = sm.section_target(im, refs[i], alpha=al)
            tot += float((im - tg).abs().mean())
    return tot / len(EV)

hist, probes = [], [(0, probe())]
print(f"  probe at 0: {probes[0][1]:.5f}")
t0 = time.time()
for it in range(ITERS):
    d = train_d[it % len(train_d)]
    ref = refs[(it // len(train_d)) % len(refs)] if it >= len(train_d) else refs[it % len(refs)]
    img, alpha, K, nf = render(d)
    with torch.no_grad():
        tgt = sm.section_target(img, ref, alpha=alpha)
    loss = (img - tgt).abs().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        st["interior"].clamp_(0, 1); st["surf_rgb"].clamp_(0, 1)
        st["split_w"].clamp_(1e-3, 1 - 1e-3)
    hist.append(float(loss))
    if (it + 1) % max(1, ITERS // 20) == 0:
        probes.append((it + 1, probe()))
        print(f"    probe at {it+1}: {probes[-1][1]:.5f}", flush=True)
    if it % 50 == 0 or it == ITERS - 1:
        print(f"  it {it:4d}  d {d:+.4f}  L1 {float(loss):.5f}  "
              f"cells {K:,}  tris {nf:,}  {time.time()-t0:.1f}s", flush=True)

print(f"\ntrained {ITERS} iterations in {time.time()-t0:.1f}s")
print(f"  L1 first 10 mean {np.mean(hist[:10]):.5f} -> last 10 mean {np.mean(hist[-10:]):.5f}")
print(f"  dual vertices moved: mean {float((st['dual_v']-dv0).norm(dim=1).mean()):.3e} "
      f"of a voxel, max {float((st['dual_v']-dv0).norm(dim=1).max()):.3e}")
dump("final", OUT + "/eval_final")
np.save(OUT + "/loss.npy", np.array(hist))
torch.save({k: st[k].detach().cpu() for k in ("dual_v", "split_w", "surf_rgb", "interior")},
           OUT + "/params.pt")
json.dump(dict(iters=ITERS, loss=hist, probe=probes), open(OUT + "/hist.json", "w"))
print("probe curve: " + "  ".join(f"{i}:{v:.5f}" for i, v in probes))
print("TRAIN_OK")
