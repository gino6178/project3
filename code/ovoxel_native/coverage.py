"""What fraction of each parameter tensor ever receives a gradient.

Two regimes, measured the same way: run one backward pass per training view, OR the per-row
gradient supports together, and report the fraction.

  before   one fixed transverse camera, 17 planes on a linspace band -- what run1 and run2 trained
           under, and the measurement that said the supervision was the bottleneck.
  after    both section families and the six exterior views: 16 transverse depths at the trainer's
           own camera, 10 longitudinal azimuths at 18-degree spacing, 6 exterior directions.

The exterior views are reported separately as well, because they are the only supervision that
sees the skin away from a cut rim, and the whole question is how much that is worth.
"""
import os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native")
sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON
import nvdiffrast.torch as dr
import section_match as sm
import refsel

W = "/workspace/ovoxel_native"
FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
REF_H = os.path.join(FN, "secref_orraw_hsep")
REF_V = os.path.join(FN, "secref_orraw_vsep")
EXT = os.path.join(FN, "cube_or6_prep")
RES = int(os.environ.get("RES", "512"))
dev = "cuda"
KEYS = ("dual_v", "split_w", "surf_rgb", "interior")

st = torch.load(W + "/state_orange.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
for k in KEYS:
    st[k] = st[k].detach().clone().requires_grad_(True)


def fresh():
    return {k: torch.zeros(len(st[k]), dtype=torch.bool, device=dev) for k in KEYS}


def accumulate(touch, fn):
    img, al, _, _ = fn()
    return img, al


def step(touch, img, al, ref):
    tgt = sm.section_target(img, ref, alpha=al)
    (img - tgt).abs().mean().backward()
    for k in KEYS:
        if st[k].grad is not None:
            touch[k] |= (st[k].grad.abs().sum(-1) > 0)
            st[k].grad = None


def report(name, touch):
    print(f"{name}")
    for k in KEYS:
        print(f"  {k:<10} {int(touch[k].sum()):>9,} / {len(touch[k]):>9,}  "
              f"({100*float(touch[k].float().mean()):5.1f}%)")
    return {k: float(touch[k].float().mean()) for k in KEYS}


# ---------------------------------------------------------------- before
C0 = np.load(W + "/cams_orange.npz")


def plane_in_pos_frame(i):
    a = C0[f"c{i}_affine"]; A, t = a[:3], a[3]
    nt = C0[f"c{i}_plane_t"][:3]; dt = float(C0[f"c{i}_plane_t"][3])
    n = np.linalg.solve(A, nt); d = dt - float(t @ n); s = np.linalg.norm(n)
    return n / s, d / s


EV = [plane_in_pos_frame(i) for i in range(6)]
d_ev = np.array([d for _, d in EV])
band = (d_ev.min() - 0.04, d_ev.max() + 0.04)
train_d = [x for x in np.linspace(band[0], band[1], 21) if np.abs(x - d_ev).min() > 0.004]
MVP0 = torch.as_tensor(C0["c0_fp"], dtype=torch.float32, device=dev)
NRM0 = torch.as_tensor(EV[0][0], dtype=torch.float32, device=dev)
ref0 = cv2.imread(os.path.join(REF_H, "or_trans_00.png"))[:, :, ::-1].astype(np.float32) / 255.

t_before = fresh()
for d in train_d:
    img, al, _, _ = ON.render_section(st, glctx, MVP0, NRM0, float(d), RES)
    step(t_before, img, al, ref0)
b = report(f"before: one transverse camera, {len(train_d)} planes", t_before)

# ---------------------------------------------------------------- after
C = np.load(W + "/cams_mv.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]

t_h, t_v, t_e = fresh(), fresh(), fresh()
for j, i in enumerate(range(H_LO, H_HI)):
    img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, float(hd[i]), RES)
    step(t_h, img, al, refsel.as_array(refsel.solved_photo(REF_H, j, H_HI - H_LO), RES))
report(f"  transverse alone ({H_HI-H_LO} depths, the trainer's own band)", t_h)

NV = len(C["v_planes"])
for i in range(NV):
    n = torch.as_tensor(C["v_planes"][i, :3], dtype=torch.float32, device=dev)
    mvp = torch.as_tensor(C["v_mvp"][i], dtype=torch.float32, device=dev)
    img, al, _, _ = ON.render_section(st, glctx, mvp, n, float(C["v_planes"][i, 3]), RES)
    step(t_v, img, al, refsel.as_array(refsel.photo(REF_V, i, NV), RES))
report(f"  longitudinal alone ({NV} azimuths)", t_v)

names = [str(x) for x in C["e_names"]]
for i, nm in enumerate(names):
    mvp = torch.as_tensor(C["e_mvp"][i], dtype=torch.float32, device=dev)
    img, al, _, _ = ON.render_exterior(st, glctx, mvp, RES)
    ref = cv2.imread(os.path.join(EXT, f"{nm}_ref.png"))[:, :, ::-1].astype(np.float32) / 255.
    step(t_e, img, al, ref)
report(f"  exterior alone ({len(names)} views)", t_e)

t_all = {k: (t_h[k] | t_v[k] | t_e[k]) for k in KEYS}
a = report(f"after: both families + the six exterior views "
           f"({H_HI-H_LO}+{NV}+{len(names)} = {H_HI-H_LO+NV+len(names)} views)", t_all)
t_sec = {k: (t_h[k] | t_v[k]) for k in KEYS}
report("  (sections only, no exterior views)", t_sec)

print("\nchange")
for k in KEYS:
    print(f"  {k:<10} {100*b[k]:5.1f}%  ->  {100*a[k]:5.1f}%   x{a[k]/max(b[k],1e-9):.2f}")

np.savez(W + "/out/coverage.npz",
         **{f"before_{k}": t_before[k].cpu().numpy() for k in KEYS},
         **{f"after_{k}": t_all[k].cpu().numpy() for k in KEYS},
         **{f"sec_{k}": t_sec[k].cpu().numpy() for k in KEYS},
         **{f"ext_{k}": t_e[k].cpu().numpy() for k in KEYS})
print("COVERAGE_OK")
