"""Step 2 and 3: build the representation from the orange's lattice, then render a cut."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
import ovnative as ON
import nvdiffrast.torch as dr

LAT = "/workspace/rebuild/worktree/build_orange/lattice"
CAMS = "/workspace/ovoxel_native/cams_orange.npz"
OUT = "/workspace/ovoxel_native/out"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
CACHE = "/workspace/ovoxel_native/state_orange.pt"

t0 = time.time()
if os.path.exists(CACHE) and os.environ.get("REBUILD", "0") != "1":
    st = torch.load(CACHE, map_location=dev, weights_only=False)
    ON.FDG = ON._load_ovoxel()
    print(f"state loaded from cache in {time.time()-t0:.1f}s")
else:
    st = ON.build(LAT, device=dev)
    torch.save(st, CACHE)
    print(f"built in {time.time()-t0:.1f}s -> {CACHE}")

print("\n--- the representation ---")
for k in ("coords", "dual_v", "inter", "split_w", "surf_rgb", "interior", "solid", "idx3"):
    v = st[k]
    print(f"  {k:<10} {tuple(v.shape)!s:<18} {str(v.dtype):<14} "
          f"{'LEARNED' if k in ('dual_v','split_w','surf_rgb','interior') else 'fixed'}")
n_par = sum(st[k].numel() for k in ("dual_v", "split_w", "surf_rgb", "interior"))
print(f"  parameters: {n_par:,} floats ({n_par*4/2**20:.1f} MiB)")

# --- planes and cameras, from the pipeline's own held-out sequence
C = np.load(CAMS)
def plane_in_pos_frame(i):
    a = C[f"c{i}_affine"]            # (4,3): pos = [tpos,1] @ a
    A, t = a[:3], a[3]
    nt = C[f"c{i}_plane_t"][:3]; dt = C[f"c{i}_plane_t"][3]
    npos = np.linalg.solve(A, nt)
    dpos = dt - float(t @ npos)
    s = np.linalg.norm(npos)
    return npos / s, dpos / s

glctx = dr.RasterizeCudaContext(device=dev)
res = 512
import cv2

for i in range(6):
    n, d = plane_in_pos_frame(i)
    nt_ = torch.as_tensor(n, dtype=torch.float32, device=dev)
    mvp = torch.as_tensor(C[f"c{i}_fp"], dtype=torch.float32, device=dev)
    t1 = time.time()
    img, alpha, K, nf = ON.render_section(st, glctx, mvp, nt_, float(d), res)
    torch.cuda.synchronize()
    a = img.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()
    cov = float(alpha.mean())
    ref = cv2.imread(f"/workspace/rebuild/worktree/eval_orange/rh{i}_init_0.png")[:, :, ::-1] / 255.0
    ref = cv2.resize(ref, (res, res))
    fgr = (np.abs(ref - 1).max(2) > 0.03)
    iou = float(((alpha[0].cpu().numpy() > 0.5) & fgr).sum() /
                max(((alpha[0].cpu().numpy() > 0.5) | fgr).sum(), 1))
    print(f"  cut {i}: plane n={np.round(n,4)} d={d:+.5f}  {K:,} crossed cells, {nf:,} tris, "
          f"{time.time()-t1:.2f}s, coverage {cov:.3f} (pipeline {fgr.mean():.3f}), silhouette IoU {iou:.3f}")
    cv2.imwrite(f"{OUT}/native_rh{i}.png", (a[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(f"{OUT}/pipe_rh{i}.png", (ref[:, :, ::-1] * 255).astype(np.uint8))

# gradient check through the whole thing
print("\n--- gradients through render_section ---")
st["dual_v"].requires_grad_(True); st["split_w"].requires_grad_(True)
st["surf_rgb"].requires_grad_(True); st["interior"].requires_grad_(True)
n, d = plane_in_pos_frame(0)
img, alpha, K, nf = ON.render_section(st, glctx,
    torch.as_tensor(C["c0_fp"], dtype=torch.float32, device=dev),
    torch.as_tensor(n, dtype=torch.float32, device=dev), float(d), res)
loss = ((img - 0.5) ** 2).mean()
loss.backward()
for k in ("dual_v", "split_w", "surf_rgb", "interior"):
    g = st[k].grad
    print(f"  {k:<10} nonzero rows {int((g.abs().sum(-1)>0).sum()):>8,}/{len(g):<9,} "
          f"|g|max {g.abs().max().item():.4g}")
print("STEP23_OK")
