"""Every arm scored on both halves: the planes it was shown, and the planes it was not.

`scoreruns.py` reports the held-out half only, which says how well an arm generalises and nothing
about what it did with its own supervision. An arm can win the held-out column by fitting its
photographs loosely, or lose it by fitting them tightly and interpolating badly, and the two cases
call for opposite conclusions. So this renders each run twice: at the depths and azimuths its
photographs were taken at, scored against those photographs, and at the held-out cuts, scored
against the held-out photographs.
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism
import ovnative as ON
import anchor
import refsel
import nvdiffrast.torch as dr
from PIL import Image

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
OBJDIR = "/workspace/rebuild/project3/code/objects"
W = os.path.dirname(os.path.abspath(__file__))
OBJ = os.environ.get("OBJ", "orange_sp")
RUNS = [r for r in os.environ.get("SF_RUNS", "").split(",") if r]
RES = int(os.environ.get("RES", "512"))
dev = "cuda"
OUT = "/tmp/sf"

conf = open(f"{OBJDIR}/{OBJ}.conf").read()


def spec(key, default=None):
    v = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(key)]
    return v[0] if v else default


st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
# the camera set the run was trained against: a run on the corrected cameras scored
# against the old ones is a different object at every plane
C = np.load(f"{W}/cams_{OBJ}{os.environ.get('SF_CAMS', '_v2')}.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hd = C["h_planes"][:, 3]
vmvp = torch.as_tensor(C["v_mvp"], dtype=torch.float32, device=dev)
vp = C["v_planes"]
ehm = torch.as_tensor(C["eh_mvp"], dtype=torch.float32, device=dev)
ehp = C["eh_planes"]
evm = torch.as_tensor(C["ev_mvp"], dtype=torch.float32, device=dev)
evp = C["ev_planes"]

sup_h = realism._paths(os.path.join(FN, spec("REF_H=")))
sup_v = realism._paths(os.path.join(FN, spec("REF_V=")))
hld_h = realism._paths(os.path.join(FN, spec("EVAL_REF="))) if spec("EVAL_REF=") else []
hld_v = realism._paths(os.path.join(FN, spec("EVAL_REF_V=", spec("EVAL_REF=")))) if hld_h else []
print(f"{OBJ}: {len(sup_h)}/{len(sup_v)} photographs it was shown, "
      f"{len(hld_h)}/{len(hld_v)} it was not")


def load(run):
    p = torch.load(f"{W}/{run}/params.pt", map_location=dev)
    st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
    if "dec_i" in p:
        # the decoder's shape comes from the checkpoint, not from this process's environment: a run
        # trained with a wider trunk cannot be loaded into the default one, and silently scoring
        # the wrong architecture would be worse than the crash it caused
        w = p["dec_i"]["stage1.0.weight"].shape[0]
        n = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
        if (w, n) != (anchor.W_HID, anchor.N_HID):
            print(f"    {run}: trunk {w}x{n}")
        anchor.W_HID, anchor.N_HID = w, n
        di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
        di.load_state_dict(p["dec_i"])
        ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
        ds.load_state_dict(p["dec_s"])
        with torch.no_grad():
            st["interior"], st["surf_rgb"] = di(), ds()
    else:
        st["interior"] = p["interior"].to(dev); st["surf_rgb"] = p["surf_rgb"].to(dev)


@torch.no_grad()
def render(tag, planes):
    d = f"{OUT}/{tag}"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*.png"):
        os.remove(f)
    for i, (mvp, n, dd) in enumerate(planes):
        img, _, _, _ = ON.render_section(st, glctx, mvp, n, float(dd), RES)
        a = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(a).save(f"{d}/{i:03d}.png")
    return sorted(glob.glob(f"{d}/*.png"))


PL_SUP_H = [(hmvp, hn, hd[H_LO + i]) for i in range(H_HI - H_LO)]
PL_SUP_V = [(vmvp[j], torch.as_tensor(vp[j, :3], dtype=torch.float32, device=dev), vp[j, 3])
            for j in range(len(vp))]
PL_HLD_H = [(ehm[i], torch.as_tensor(ehp[i, :3], dtype=torch.float32, device=dev), ehp[i, 3])
            for i in range(len(ehp))]
PL_HLD_V = [(evm[i], torch.as_tensor(evp[i, :3], dtype=torch.float32, device=dev), evp[i, 3])
            for i in range(len(evp))]

print(f"\n  {'run':<22}{'shown h':>9}{'shown v':>9}{'unseen h':>10}{'unseen v':>10}{'mean':>8}")
for r in RUNS:
    if not os.path.exists(f"{W}/{r}/params.pt"):
        print(f"  {r:<22}{'(no params)':>9}")
        continue
    load(r)
    a = realism._dreamsim(sup_h, render(f"{r}_sh", PL_SUP_H), dev)
    b = realism._dreamsim(sup_v, render(f"{r}_sv", PL_SUP_V), dev)
    c = realism._dreamsim(hld_h, render(f"{r}_hh", PL_HLD_H), dev) if hld_h else float("nan")
    e = realism._dreamsim(hld_v, render(f"{r}_hv", PL_HLD_V), dev) if hld_v else float("nan")
    m = float(np.nanmean([a, b, c, e]))
    print(f"  {r:<22}{a:>9.4f}{b:>9.4f}{c:>10.4f}{e:>10.4f}{m:>8.4f}", flush=True)
