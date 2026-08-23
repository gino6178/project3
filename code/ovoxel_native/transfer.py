"""What survives the trip from a photograph to a cell, along the axis and across it?

Everything else has been ruled out: coverage, observation counts, decoder capacity, the tug between
the families, their spatial balance, the prior's direction, and the photographs themselves, which
carry three times as much difference between depths as they do within a depth. The field still
varies only 0.65 to 0.82 as much along the axis as across it.

That leaves the renderer. This puts a KNOWN field in and reads it back through the same path the
training uses: a sinusoid along the polar axis, and the same sinusoid across it, at several
wavelengths. Rendering a plane and scattering the rendered pixels back to the cells they came from
gives, per wavelength and per direction, the fraction of the amplitude that survives. If the axial
direction loses more, the loss is in the sampling, and the number says how much.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "watermelon_sp")
RES = int(os.environ.get("RES", "512"))
dev = "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]
hc = st["hc"]
cen = (st["solid"].float() + 0.5) * hc + torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
axis = np.asarray(C["h_planes"][0, :3], float); axis /= np.linalg.norm(axis)
ax_t = torch.as_tensor(axis, dtype=torch.float32, device=dev)
a0 = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
per = np.cross(axis, a0); per /= np.linalg.norm(per)
per_t = torch.as_tensor(per, dtype=torch.float32, device=dev)
ctr = cen.mean(0)

s_ax = (cen - ctr[None]) @ ax_t          # position along the axis, per cell
s_pe = (cen - ctr[None]) @ per_t         # and across it


def fit_amplitude(vals, phase):
    """Least squares amplitude of a known sinusoid in `vals`, given its phase per cell."""
    b = torch.stack([torch.cos(phase), torch.sin(phase), torch.ones_like(phase)], 1)
    sol = torch.linalg.lstsq(b, vals[:, None]).solution
    return float((sol[0] ** 2 + sol[1] ** 2).sqrt())


print(f"{OBJ}: cell {hc:.5f}; a wave written into the field, rendered, and read back from the "
      f"pixels\n")
print(f"  {'wavelength':>12}{'direction':>14}{'amplitude in':>14}{'amplitude out':>15}{'kept':>8}")
for lam_cells in (4, 8, 16, 32):
    lam = lam_cells * hc
    for name, coord in (("along axis", s_ax), ("across axis", s_pe)):
        phase = 2 * np.pi * coord / lam
        col = 0.5 + 0.25 * torch.cos(phase)
        st["interior"] = torch.stack([col, col, col], 1)
        got = torch.zeros(len(col), device=dev)
        cnt = torch.zeros(len(col), device=dev)
        with torch.no_grad():
            for i in range(H_LO, H_HI):
                P, T, _ = ON.cut_polygons(st, hn, float(hd[i]), device=dev)
                if len(T) == 0:
                    continue
                # where each polygon vertex sits, and what the field says there
                v = ON.sample_interior(st, P)[:, 0]
                idx = torch.round((P - torch.as_tensor(st["org"], dtype=torch.float32,
                                                       device=dev)) / hc - 0.5
                                  - torch.as_tensor(st["idx_lo"], dtype=torch.float32,
                                                    device=dev)).long()
                G = torch.tensor(st["idx3"].shape, device=dev)
                ok = ((idx >= 0) & (idx < G)).all(1)
                idx = idx.clamp(min=torch.zeros(3, dtype=torch.long, device=dev), max=G - 1)
                row = st["idx3"][idx[:, 0], idx[:, 1], idx[:, 2]].long()
                m = ok & (row >= 0)
                got.index_add_(0, row[m], v[m])
                cnt.index_add_(0, row[m], torch.ones_like(v[m]))
        seen = cnt > 0
        out = got[seen] / cnt[seen]
        a_in = fit_amplitude(col[seen], phase[seen])
        a_out = fit_amplitude(out, phase[seen])
        print(f"  {lam_cells:>9} cells{name:>14}{a_in:>14.4f}{a_out:>15.4f}"
              f"{100 * a_out / max(a_in, 1e-9):>7.0f}%")
