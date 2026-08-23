"""The two target constructions on every object, with the distortion each applies.

The per-ray map gives every angle its own scale factor, so how much it bends a photograph is the
spread of those factors: if the rendered silhouette were a perfect circle and the photograph too,
every angle would share one factor and nothing would bend. The number reported is the coefficient
of variation of the per-angle scale, which is that spread, and it is what the affine map replaces
with a single linear transform.

One transverse and one longitudinal plane per object, so a change that helps the watermelon can be
seen failing elsewhere before it is adopted.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import ovnative as ON
import anchor
import refsel
import section_match as sm
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw



W = os.path.dirname(os.path.abspath(__file__))
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
OBJS = [o for o in os.environ.get("MA_OBJS",
        "watermelon_sp,orange_sp,apple1_sp,bread_sp,cake2_sp,pomegranate2_sp,doughnut").split(",")]
RUN = os.environ.get("MA_RUN", "s_rs")
RES = int(os.environ.get("RES", "256"))
# The camera set with both corrections in it -- the polar axis each conf names, and a distance far
# enough back that the object is inside the frame.  It is byte-identical to the trained set for the
# four objects neither correction touches; the other three are drawn from their own retrained run.
CS = os.environ.get("MA_CAMS", "_v2")
dev = "cuda"
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
rows, notes, AXNOTE = [], [], {}

for OBJ in OBJS:
    try:
        st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
        conf = open(f"{OBJDIR}/{OBJ}.conf").read()
        C = np.load(f"{W}/cams_{OBJ}{CS}.npz")
        # the retrained run if it has finished a save, not merely if its directory
        # exists -- a run creates that on its first second
        run = "s_v2" if os.path.exists(f"{W}/s_v2_{OBJ}/params.pt") else RUN
        p = torch.load(f"{W}/{run}_{OBJ}/params.pt", map_location=dev)
    except Exception as e:
        notes.append(f"{OBJ}: skipped ({type(e).__name__}: {e})"); continue
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

    def spec(k):
        return [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(k)][0]

    H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0])
    NH, NV = H_HI - H_LO, len(C["v_planes"])
    # Which family is the transverse one is decided by the object, not by the label it was trained
    # under: the apple's stored polar axis is far enough from its stem that the family called
    # transverse is the one cutting along it.  This relabels the figure's columns and turns its
    # cameras; nothing trained changes.
    panels, cv, refs, ext_mvp = [], {}, [], {}
    for col in ("h", "v"):
        fam = col
        # the plane of this family that cuts the most, not the middle of the list: the doughnut's
        # middle longitudinal plane grazes the ring and its mask is a thin bar, which says nothing
        # about what the family sees
        cand = []
        n_pl = NH if fam == "h" else NV
        for idx in range(n_pl):
            if fam == "h":
                mv_ = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
                nn_ = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
                dd_ = float(C["h_planes"][H_LO + idx, 3])
            else:
                mv_ = torch.as_tensor(C["v_mvp"][idx], dtype=torch.float32, device=dev)
                nn_ = torch.as_tensor(C["v_planes"][idx, :3], dtype=torch.float32, device=dev)
                dd_ = float(C["v_planes"][idx, 3])
            with torch.no_grad():
                _, a_, _, _ = ON.render_section(st, glctx, mv_, nn_, dd_, RES, exterior=False)
            cand.append((float((a_ > 0.5).float().sum()), idx, mv_, nn_, dd_))
        _, k, mv, n_, d_ = max(cand)
        ext_mvp[col] = mv
        n_col = NH if col == "h" else NV
        kk = int(round(k * (n_col - 1) / max(n_pl - 1, 1)))
        ref = (refsel.as_array(refsel.solved_photo(f"{FN}/{spec('REF_H=')}", kk, NH), RES)
               if col == "h" else
               refsel.as_array(refsel.photo(f"{FN}/{spec('REF_V=')}", kk, NV), RES))
        with torch.no_grad():
            img, al, _, _ = ON.render_section(st, glctx, mv, n_, d_, RES,
                                              exterior=False)
            os.environ["SEC_MAP"] = "affine"
            t_aff = sm.section_target(img, ref, alpha=al)
            os.environ["SEC_MAP"] = "ray"
        comp = (al[0] > 0.5).cpu().numpy()
        m_ref = (np.asarray(ref).min(2) < 0.98)
        try:
            (_, _), _, dro = sm._ray_coords(comp)
            (_, _), _, sro = sm._ray_coords(m_ref)
            g = dro / np.maximum(sro, 1e-6)
            g = g[np.isfinite(g) & (g > 0)]
            cv[col] = float(np.std(g) / max(np.mean(g), 1e-9))
        except Exception:
            cv[col] = float("nan")
        refs.append(torch.as_tensor(ref, device=dev).permute(2, 0, 1))
        panels += [al[0][None].expand(3, -1, -1), t_aff]

    # Every object drawn at the same size. The renders arrive at whatever fraction of the frame
    # each object's own camera gives it -- the pomegranate fills its frame and the doughnut's
    # longitudinal cut is a thin bar -- and a figure meant for comparing shapes should not also be
    # comparing camera distances.
    def _fit(t, bg, frac=0.72):
        m = (t.mean(0) > 0.015) if bg == 0.0 else (t.mean(0) < 0.985)
        ys, xs = m.nonzero(as_tuple=True)
        if ys.numel() < 16:
            return t
        cy, cx = float(ys.float().mean()), float(xs.float().mean())
        h = max(float(ys.max() - ys.min()), float(xs.max() - xs.min())) / 2 + 1
        want = frac * t.shape[-1] / 2
        k = want / h
        g = torch.nn.functional.affine_grid(
            torch.tensor([[[1 / k, 0, (2 * cx / (t.shape[-1] - 1) - 1)],
                           [0, 1 / k, (2 * cy / (t.shape[-2] - 1) - 1)]]], device=t.device),
            (1, 3, t.shape[-2], t.shape[-1]), align_corners=True)
        return torch.nn.functional.grid_sample(t[None] - bg, g, align_corners=True,
                                               padding_mode="zeros")[0] + bg

    # the outside from each family's own camera, so it stands next to the cuts it belongs to
    with torch.no_grad():
        ext = {c: ON.render_exterior(st, glctx, m, RES)[0] for c, m in ext_mvp.items()}
    panels = [panels[0], panels[1], ext["h"], ext["v"], panels[2], panels[3]]
    # a couple of objects fill their own camera far more than the rest; the figure compares shapes,
    # so each one is drawn at the fraction of the frame that lets it be compared
    fr = {"pomegranate2_sp": 0.56}.get(OBJ, 0.72)
    panels = [_fit(q, bg, fr) for q, bg in zip(panels, (0.0, 1.0, 1.0, 1.0, 0.0, 1.0))]

    # the two references share the last column, one above the other
    half = torch.nn.functional.interpolate(torch.stack(refs), size=(RES // 2, RES // 2),
                                           mode="area")
    pad = torch.ones(3, RES, RES, device=dev)
    pad[:, : RES // 2, RES // 4: RES // 4 + RES // 2] = half[0]
    pad[:, RES // 2:, RES // 4: RES // 4 + RES // 2] = half[1]
    panels.append(pad)
    rows.append((OBJ, torch.cat([q.clamp(0, 1) for q in panels], -1)))
    ax_off = ""
    try:
        pts = st["solid"].float().cpu().numpy() * float(st["hc"])
        pts = pts - pts.mean(0)
        wv, V = np.linalg.eigh(pts.T @ pts / len(pts))
        a_ = np.asarray(C["h_planes"][0, :3], float); a_ /= np.linalg.norm(a_)
        off = min(np.degrees(np.arccos(min(1.0, abs(float(a_ @ V[:, k]))))) for k in range(3))
        ax_off = f", polar axis {off:.0f} deg off the nearest principal axis" if off > 20 else ""
    except Exception:
        pass
    notes.append(f"  {OBJ:<18} per-ray scale spread: transverse {cv.get('h', float('nan')):.3f}, "
                 f"longitudinal {cv.get('v', float('nan')):.3f}{ax_off}")
    AXNOTE[OBJ] = ax_off

print("how unevenly the current map stretches each photograph (0 = one scale for every angle)")
print("\n".join(notes))
if rows:
    sheet = torch.cat([r for _, r in rows], -2).permute(1, 2, 0)
    a = (sheet.cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(a)
    d_ = ImageDraw.Draw(im)
    for i, (name, _) in enumerate(rows):
        for j, t in enumerate(("the cut's own MASK, transverse", "the TARGET on it",
                               "the OUTSIDE, transverse camera",
                               "the OUTSIDE, longitudinal camera",
                               "MASK, longitudinal", "the TARGET on it",
                               "the PHOTOGRAPHS used")):
            x, y = j * RES + 6, i * RES + 4
            lab = (f"{name} - {t}{AXNOTE.get(name, '')}") if j == 0 else t
            d_.rectangle([x - 3, y - 2, x + 6 * len(lab), y + 14], fill=(255, 255, 255))
            d_.text((x, y), lab, fill=(150, 30, 30) if "affine" in t else (20, 20, 120))
    im.save(f"{W}/mapall.jpg", quality=94)
    print(f"SHEET mapall.jpg  ({len(rows)} objects)")
