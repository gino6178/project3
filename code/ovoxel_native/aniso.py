"""Is the defect on an unphotographed plane the lattice showing through?

Nothing measured so far sees what the eye sees. DreamSim scores the unphotographed planes BETTER
than the photographed ones, the difference between neighbouring cells does not vary with how often
they were supervised, and neither a bigger decoder nor a robust loss changes anything. So before
another fix is tried, the defect itself has to become a number.

The prediction: the field is one colour per cell, so it carries a structure aligned with the
lattice. A photograph laid on a plane covers that structure with its own; a plane with no
photograph has nothing to cover it, and what remains is the grid -- blocks and stripes along the
lattice axes. If that is what is happening, the image's gradient will be stronger along the two
lattice directions than along the diagonals, and more so where there is no photograph.

Real texture has no such preference, so the photographs are the control: their ratio should be 1.
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovnative as ON
import anchor
import refsel
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RUN = os.environ.get("AN_RUN", f"s_rs_{os.environ.get('OBJ', 'orange_sp')}")
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
RES = int(os.environ.get("RES", "512"))
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(f"{W}/cams_{OBJ}_bal.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
NH = H_HI - H_LO
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
step_h = float(hd[1] - hd[0])

p = torch.load(f"{W}/{RUN}/params.pt", map_location=dev)
st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
if "dec_i" in p:
    w = p["dec_i"]["stage1.0.weight"].shape[0]
    n = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
    anchor.W_HID, anchor.N_HID = w, n
    di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
    di.load_state_dict(p["dec_i"])
    ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
    ds.load_state_dict(p["dec_s"])
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()


def aniso(img, mask, bins=18):
    """How concentrated the gradient's direction is, and in how many directions.

    The first version compared the image's rows and columns against its diagonals, which only tests
    for a grid if the lattice happens to project onto the image axes -- and it does not, the camera
    is not aligned with the lattice. It read 1.05 on every kind of plane and 1.07 on the
    photographs, which says nothing.
    
    This assumes no direction. The gradient's orientation is binned over 180 degrees and weighted by
    its magnitude; an isotropic texture fills the bins evenly and a grid puts its energy into two
    bins about 90 degrees apart. Reported as the peak bin over the mean bin, which is 1 for
    isotropic and grows with any preferred direction, and as the gap between the two strongest
    directions, which is near 90 for a grid.
    """
    gx = img[:, :, 1:] - img[:, :, :-1]
    gy = img[:, 1:, :] - img[:, :-1, :]
    n = min(gx.shape[1], gy.shape[1]), min(gx.shape[2], gy.shape[2])
    gx, gy = gx[:, :n[0], :n[1]].mean(0), gy[:, :n[0], :n[1]].mean(0)
    mm = mask[:n[0], :n[1]]
    mag = (gx ** 2 + gy ** 2).sqrt()
    ang = torch.atan2(gy, gx) % np.pi                       # direction, modulo a half turn
    sel = mm & (mag > mag[mm].median())                     # the edges, not the flat interior
    if int(sel.sum()) < 64:
        return float("nan"), float("nan"), float("nan")
    idx = (ang[sel] / np.pi * bins).long().clamp(0, bins - 1)
    h = torch.zeros(bins, device=img.device).index_add_(0, idx, mag[sel])
    h = h / h.mean()
    top = int(h.argmax())
    second = int((h - torch.nn.functional.one_hot(torch.tensor(top), bins).to(h) * 1e9).argmax())
    gap = abs(top - second) * 180.0 / bins
    return float(h.max()), min(gap, 180 - gap), float(h.std())

def _unused(img, mask):
    a = img
    m = mask
    def e(d0, d1, scale):
        s = a[:, max(d0, 0):a.shape[1] + min(d0, 0), max(d1, 0):a.shape[2] + min(d1, 0)]
        t = a[:, max(-d0, 0):a.shape[1] - max(d0, 0), max(-d1, 0):a.shape[2] - max(d1, 0)]
        mm = m[max(d0, 0):m.shape[0] + min(d0, 0), max(d1, 0):m.shape[1] + min(d1, 0)] & \
             m[max(-d0, 0):m.shape[0] - max(d0, 0), max(-d1, 0):m.shape[1] - max(d1, 0)]
        if int(mm.sum()) < 32:
            return float("nan")
        return float(((s - t).abs().mean(0)[mm]).mean()) / scale
    ax = 0.5 * (e(1, 0, 1.0) + e(0, 1, 1.0))
    di_ = 0.5 * (e(1, 1, 2 ** 0.5) + e(1, -1, 2 ** 0.5))
    return ax / max(di_, 1e-9), ax, di_


@torch.no_grad()
def shot(d):
    img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, float(d), RES, exterior=False)
    return img, (al[0] > 0.5)


# The held-out planes are the ones the page shows beside the supervised ones: the same two
# families, drawn at their own depths and azimuths, and never given a target photograph while
# training. A depth halfway between two photographed ones is NOT one of them -- the pipeline
# jitters each plane by half a slot, so the transverse family sweeps the whole band and every depth
# in it gets rendered at some point. These are the planes the defect is reported on.
ehm = torch.as_tensor(C["eh_mvp"], dtype=torch.float32, device=dev)
ehp = C["eh_planes"]
evm = torch.as_tensor(C["ev_mvp"], dtype=torch.float32, device=dev)
evp = C["ev_planes"]

rows = {"photographed": [], "between them": [], "held out, transverse": [],
        "held out, longitudinal": []}
print(f"{OBJ} from {RUN}: how concentrated the gradient's direction is "
      f"(1.0 = no preferred direction)")
for i in range(NH):
    im, m = shot(hd[H_LO + i])
    r, a, b = aniso(im, m)
    rows["photographed"].append(r)
for f in np.linspace(0.5, NH - 1.5, NH - 1):
    im, m = shot(float(hd[H_LO]) + step_h * f)
    r, a, b = aniso(im, m)
    rows["between them"].append(r)
with torch.no_grad():
    for k, (mv, pl, name) in enumerate([(ehm, ehp, "held out, transverse"),
                                        (evm, evp, "held out, longitudinal")]):
        for i in range(len(pl)):
            img, al, _, _ = ON.render_section(
                st, glctx, mv[i], torch.as_tensor(pl[i, :3], dtype=torch.float32, device=dev),
                float(pl[i, 3]), RES, exterior=False)
            r, a, b = aniso(img, al[0] > 0.5)
            if r == r:
                rows[name].append(r)
for k, v in rows.items():
    v = [x for x in v if x == x]
    print(f"  {k:<16} peak/mean {np.mean(v):.3f}   (min {min(v):.3f}, max {max(v):.3f}, "
          f"{len(v)} planes)")

spec = [l.split("=", 1)[1].strip() for l in open(f"{OBJDIR}/{OBJ}.conf").read().splitlines()
        if l.startswith("REF_H=")][0]
ph = []
for q in sorted(refsel.photos_in(f"{FN}/{spec}")):
    a = np.asarray(Image.open(q).convert("RGB").resize((RES, RES), Image.LANCZOS), np.float32) / 255.
    t = torch.as_tensor(a, device=dev).permute(2, 0, 1)
    r, _, _ = aniso(t, (t.min(0).values < 0.98))
    ph.append(r)
print(f"  {'the photographs':<16} {np.mean(ph):.3f}   (min {min(ph):.3f}, max {max(ph):.3f}, "
      f"{len(ph)} photographs)")

# and a picture, so the numbers can be checked against what they are supposed to describe:
# a supervised plane of each family beside the held-out planes of the same family.
def _shot(mv, n, d):
    with torch.no_grad():
        img, al, _, _ = ON.render_section(
            st, glctx, mv, torch.as_tensor(n, dtype=torch.float32, device=dev), float(d), RES,
            exterior=False)
    return img.clamp(0, 1)


vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]
top = torch.cat([shot(hd[H_LO + NH // 2])[0]] +
                [_shot(ehm[i], ehp[i, :3], ehp[i, 3]) for i in range(3)], -1)
bot = torch.cat([_shot(vmvp[len(vp) // 2], vp[len(vp) // 2, :3], vp[len(vp) // 2, 3])] +
                [_shot(evm[i], evp[i, :3], evp[i, 3]) for i in range(3)], -1)
sheet = torch.cat([top, bot], -2).permute(1, 2, 0)
a = (sheet.cpu().numpy() * 255).astype(np.uint8)
# label the panels, because a figure whose columns have to be counted is a figure that gets
# misread -- and this one exists to be checked against a description
try:
    from PIL import ImageDraw
    img = Image.fromarray(a)
    dr_ = ImageDraw.Draw(img)
    for r, fam in enumerate(("transverse", "longitudinal")):
        for c in range(4):
            tag = ("SUPERVISED" if c == 0 else "HELD OUT") + f"  {fam}"
            col = (20, 110, 20) if c == 0 else (170, 30, 30)
            x, y = c * RES + 10, r * RES + 8
            dr_.rectangle([x - 4, y - 2, x + 9 * len(tag), y + 18], fill=(255, 255, 255))
            dr_.text((x, y), tag, fill=col)
    a = np.asarray(img)
except Exception as e:
    print("  (labels skipped:", e, ")")
Image.fromarray(a).save(f"{W}/aniso_{OBJ}.jpg", quality=95)
print(f"SHEET aniso_{OBJ}.jpg  (each row: one supervised plane, then three held-out ones; "
      f"top transverse, bottom longitudinal)")
