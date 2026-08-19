"""What the training actually reached, and what one step costs."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON, nvdiffrast.torch as dr, section_match as sm, cv2
dev = "cuda"
st = torch.load("/workspace/ovoxel_native/state_orange.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
C = np.load("/workspace/ovoxel_native/cams_orange.npz")
def pl(i):
    a = C[f"c{i}_affine"]; A, t = a[:3], a[3]
    nt = C[f"c{i}_plane_t"][:3]; dt = float(C[f"c{i}_plane_t"][3])
    n = np.linalg.solve(A, nt); d = dt - float(t @ n); s = np.linalg.norm(n)
    return n / s, d / s
EV = [pl(i) for i in range(6)]
MVP = torch.as_tensor(C["c0_fp"], dtype=torch.float32, device=dev)
NRM = torch.as_tensor(EV[0][0], dtype=torch.float32, device=dev)
glctx = dr.RasterizeCudaContext(device=dev)

# how much of the state a training run can even reach
d_ev = np.array([d for _, d in EV]); band = (d_ev.min() - 0.04, d_ev.max() + 0.04)
train_d = [x for x in np.linspace(band[0], band[1], 21) if np.abs(x - d_ev).min() > 0.004]
for k in ("dual_v", "split_w", "surf_rgb", "interior"):
    st[k] = st[k].detach().clone().requires_grad_(True)
touch = {k: torch.zeros(len(st[k]), dtype=torch.bool, device=dev)
         for k in ("dual_v", "split_w", "surf_rgb", "interior")}
ref = cv2.imread("/workspace/rebuild/worktree/secref_orraw_hsep/or_trans_00.png")[:, :, ::-1] / 255.0
ref = ref.astype(np.float32)
for d in train_d:
    img, al, K, nf = ON.render_section(st, glctx, MVP, NRM, float(d), 512)
    tgt = sm.section_target(img, ref, alpha=al)
    (img - tgt).abs().mean().backward()
    for k in touch:
        touch[k] |= (st[k].grad.abs().sum(-1) > 0)
        st[k].grad = None
print(f"over the {len(train_d)} training planes, parameters that ever get a gradient:")
for k in touch:
    print(f"  {k:<10} {int(touch[k].sum()):>9,} / {len(touch[k]):>9,}  "
          f"({100*float(touch[k].float().mean()):.1f}%)")

# cost of one step
torch.cuda.reset_peak_memory_stats()
ts = []
for r in range(6):
    torch.cuda.synchronize(); t0 = time.time()
    img, al, K, nf = ON.render_section(st, glctx, MVP, NRM, float(train_d[r]), 512)
    torch.cuda.synchronize(); t1 = time.time()
    tgt = sm.section_target(img, ref, alpha=al)
    t2 = time.time()
    (img - tgt).abs().mean().backward()
    for k in touch: st[k].grad = None
    torch.cuda.synchronize(); t3 = time.time()
    ts.append((t1-t0, t2-t1, t3-t2))
ts = np.array(ts[1:]).mean(0)
print(f"one step at 512x512: forward {ts[0]*1000:.0f} ms, section_target (CPU) {ts[1]*1000:.0f} ms, "
      f"backward {ts[2]*1000:.0f} ms; peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

# the closed-form cut on its own
torch.cuda.synchronize(); t0 = time.time()
for r in range(10):
    P, T, K = ON.cut_polygons(st, NRM, float(train_d[0]), device=dev)
torch.cuda.synchronize()
print(f"cut_polygons over {len(st['solid']):,} solid cells: {(time.time()-t0)/10*1000:.1f} ms, "
      f"{K:,} crossed cells -> {len(T):,} triangles")
