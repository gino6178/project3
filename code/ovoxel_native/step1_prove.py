"""Step 1: one interpreter holding both halves.
flexible_dual_grid_to_mesh (o_voxel, CUDA, differentiable) -> nvdiffrast -> scalar loss,
and a nonzero gradient at the dual vertices."""
import os, sys, time
sys.path.insert(0, "/workspace/rebuild/ovox_study")
import numpy as np, torch
import ovoxload
import nvdiffrast.torch as dr

dev = "cuda"
print("torch", torch.__version__, torch.version.cuda, "| device", torch.cuda.get_device_name(0))
fdg = ovoxload.load("convert.flexible_dual_grid")
print("o_voxel loaded, _C =", sys.modules["o_voxel._C"].__file__)

# --- a small closed solid: a cube surface, voxelised
def cube(n=10):
    V, F = [], []
    g = np.linspace(-1, 1, n)
    for a in range(3):
        for s in (-1, 1):
            base = len(V)
            for i in range(n):
                for j in range(n):
                    p = [0, 0, 0]; p[a] = s; p[(a+1) % 3] = g[i]; p[(a+2) % 3] = g[j]
                    V.append(p)
            for i in range(n-1):
                for j in range(n-1):
                    q = base + i*n + j
                    F += [[q, q+1, q+n+1], [q, q+n+1, q+n]]
    return np.array(V, np.float32), np.array(F, np.int32)

V, F = cube(10)
vs = 0.1
lo = torch.tensor(V.min(0) - 0.3); hi = torch.tensor(V.max(0) + 0.3)
aabb = torch.stack([lo, hi])
coords, dual, inter = fdg.mesh_to_flexible_dual_grid(torch.tensor(V), torch.tensor(F),
                                                     voxel_size=vs, aabb=aabb)
coords, dual, inter, aabb = coords.to(dev), dual.to(dev), inter.to(dev), aabb.to(dev)
N = len(coords)
print(f"dual grid: {N} voxels")

dv = (dual / vs - coords.float()).clone().requires_grad_(True)      # (N,3) fractional
sw = torch.full((N, 1), 0.5, device=dev).requires_grad_(True)       # (N,1) split weight
mv, mf = fdg.flexible_dual_grid_to_mesh(coords.int(), dv, inter.bool(), sw, aabb,
                                        voxel_size=vs, train=True)
print(f"mesh: {tuple(mv.shape)} verts (N={N} dual + {len(mv)-N} quad-midpoints), "
      f"{tuple(mf.shape)} tris, grad_fn={type(mv.grad_fn).__name__}")

# --- nvdiffrast, CUDA backend (no display needed)
glctx = dr.RasterizeCudaContext(device=dev)
print("nvdiffrast", dr.__spec__.origin.split('/')[-2], "RasterizeCudaContext ok")

def look_at(eye, at, up):
    eye = np.asarray(eye, np.float32); at = np.asarray(at, np.float32); up = np.asarray(up, np.float32)
    f = at - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3], M[1, :3], M[2, :3] = r, u, f
    M[:3, 3] = -M[:3, :3] @ eye
    return M

def persp(fovy, aspect, n, f):
    t = 1.0 / np.tan(fovy / 2)
    return np.array([[t/aspect,0,0,0],[0,t,0,0],[0,0,(f+n)/(f-n),-2*f*n/(f-n)],[0,0,1,0]], np.float32)

mvp = torch.tensor(persp(np.deg2rad(45), 1.0, 0.1, 100.0) @ look_at([3.0, 2.0, 3.0], [0, 0, 0], [0, 1, 0]),
                   device=dev)
res = 256
pos_h = torch.cat([mv, torch.ones_like(mv[:, :1])], 1) @ mvp.T
pos_h = pos_h[None]                                          # (1, V, 4)
tri = mf.int().contiguous()

# a per-vertex colour that is itself a parameter, so both halves of the state get a gradient
col = torch.rand(len(mv), 3, device=dev, requires_grad=True)
rast, _ = dr.rasterize(glctx, pos_h, tri, resolution=[res, res])
img, _ = dr.interpolate(col[None], rast, tri)
img = dr.antialias(img, rast, pos_h, tri)                    # this is what puts geometry in the gradient
cov = float((rast[..., 3] > 0).float().mean())
print(f"rasterised: coverage {cov:.3f} of {res}x{res}")

target = torch.zeros_like(img)
loss = ((img - target) ** 2).mean()
loss.backward()
for nme, t in [("dual_vertices dv", dv), ("split_weight sw", sw), ("vertex colour", col)]:
    g = t.grad
    print(f"  {nme:<18} grad {'None' if g is None else ''}"
          f"nonzero rows {int((g.abs().sum(-1) > 0).sum())}/{len(g)}  "
          f"|g|max {g.abs().max().item():.4g}  |g|mean {g.abs().mean().item():.4g}")

# finite difference on the dual vertex, through the renderer
i = int(dv.grad.abs().sum(1).argmax()); j = int(dv.grad[i].abs().argmax())
def L(x):
    v, f_ = fdg.flexible_dual_grid_to_mesh(coords.int(), x, inter.bool(), sw.detach(), aabb,
                                           voxel_size=vs, train=True)
    ph = (torch.cat([v, torch.ones_like(v[:, :1])], 1) @ mvp.T)[None]
    r_, _ = dr.rasterize(glctx, ph, f_.int().contiguous(), resolution=[res, res])
    im, _ = dr.interpolate(col.detach()[None], r_, f_.int().contiguous())
    im = dr.antialias(im, r_, ph, f_.int().contiguous())
    return ((im - target) ** 2).mean()
eps = 1e-3
with torch.no_grad():
    a = dv.detach().clone(); a[i, j] += eps
    b = dv.detach().clone(); b[i, j] -= eps
    num = (L(a) - L(b)) / (2 * eps)
print(f"  finite difference at dual vertex {i} axis {j}: analytic {dv.grad[i,j].item():+.6e}"
      f"   numeric {num.item():+.6e}")
print("STEP1_OK")
