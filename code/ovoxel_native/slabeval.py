"""The 2x2: {O-Voxel} x {plane, slab}, on the twelve held-out cuts.

The arms were TRAINED at zero thickness; this changes only what is drawn at evaluation, which is
the question -- how much of the difference between the two pictures is the reconstruction filter
rather than the state.
"""
import glob, json, os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/ovoxel_native/vendor")
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import ovnative as ON, nvdiffrast.torch as dr, section_match as sm, refsel, anchor, realism

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
dev = "cuda"; RES = 512
SL = np.load(f"{W}/slab.npz")
SD_H, SD_V = float(SL["surf_dis_h"][0]), float(SL["surf_dis_v"][0])
NSUB = int(os.environ.get("NSUB", "7"))
refs_h = realism._paths(f"{FN}/secref_orraw_hsep")
refs_v = realism._paths(f"{FN}/secref_orraw_vsep")
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
print(f"slab half-width: transverse {SD_H:.5f} ({SD_H/float(SL['hc'][0]):.2f} cells), "
      f"longitudinal {SD_V:.5f}; {NSUB} sub-planes, uniform weight")

STATES, CAMS = {}, {}
for r in ("1", "2"):
    STATES[r] = torch.load(f"{W}/state_r{r}.pt", map_location=dev, weights_only=False)
    CAMS[r] = np.load(f"{W}/cams_mv.npz" if r == "1" else f"{W}/cams_mv_r2.npz")

NH = 16; NV = 10
P_H = [refsel.as_array(refsel.solved_photo(f"{FN}/secref_orraw_hsep", j, NH), RES)
       for j in range(NH)]
P_V = [refsel.as_array(refsel.photo(f"{FN}/secref_orraw_vsep", i, NV), RES) for i in range(NV)]


def load(arm, r):
    st = {k: v for k, v in STATES[r].items()}
    P = torch.load(f"{W}/{arm}/params.pt", map_location=dev)
    di = anchor.ColourDecoder(len(STATES[r]["interior"])).to(dev); di.load_state_dict(P["dec_i"])
    ds = anchor.ColourDecoder(len(STATES[r]["surf_rgb"])).to(dev); ds.load_state_dict(P["dec_s"])
    if bool(json.load(open(f"{W}/{arm}/hist.json"))["shell_pin"]):
        ds.pin_colour(torch.ones(len(STATES[r]["surf_rgb"]), dtype=torch.bool, device=dev),
                      STATES[r]["surf_rgb"].detach())
    with torch.no_grad():
        st["interior"], st["surf_rgb"] = di(), ds()
    st["dual_v"], st["split_w"] = P["dual_v"].to(dev), P["split_w"].to(dev)
    return st


def run(arm, r, th_h, th_v, folder):
    st = load(arm, r); C = CAMS[r]
    os.makedirs(folder, exist_ok=True)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(len(C["eh_planes"])):
            n = torch.as_tensor(C["eh_planes"][i, :3], dtype=torch.float32, device=dev)
            mvp = torch.as_tensor(C["eh_mvp"][i], dtype=torch.float32, device=dev)
            im, al, _, _ = ON.render_section(st, glctx, mvp, n, float(C["eh_planes"][i, 3]), RES,
                                             thickness=th_h, n_sub=NSUB)
            cv2.imwrite(f"{folder}/rh{i}_init_0.png",
                        (im.permute(1, 2, 0).clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
            tot += float((im - sm.section_target(im, P_H[i % NH], alpha=al)).abs().mean()); cnt += 1
        for i in range(len(C["ev_planes"])):
            n = torch.as_tensor(C["ev_planes"][i, :3], dtype=torch.float32, device=dev)
            mvp = torch.as_tensor(C["ev_mvp"][i], dtype=torch.float32, device=dev)
            im, al, _, _ = ON.render_section(st, glctx, mvp, n, float(C["ev_planes"][i, 3]), RES,
                                             thickness=th_v, n_sub=NSUB)
            cv2.imwrite(f"{folder}/rv{i}_init_0.png",
                        (im.permute(1, 2, 0).clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
            tot += float((im - sm.section_target(im, P_V[i % NV], alpha=al)).abs().mean()); cnt += 1
    a = realism._dreamsim(refs_h, sorted(glob.glob(folder + "/rh*_init_0.png")), dev)
    b = realism._dreamsim(refs_v, sorted(glob.glob(folder + "/rv*_init_0.png")), dev)
    return a, b, tot / cnt


ARMS = [("r1_pin", "1"), ("r1_pin_full", "1"), ("r1flat_pin", "1"), ("r2_pin_full", "2")]
print(f"\n  {'arm':<16} {'filter':<8} {'DS rh':>7} {'DS rv':>7} {'probe L1':>9}")
out = {}
for arm, r in ARMS:
    for tag, th, tv in (("plane", 0.0, 0.0), ("slab", SD_H, SD_V)):
        a, b, l1 = run(arm, r, th, tv, f"{W}/slabeval/{arm}_{tag}")
        out[(arm, tag)] = (a, b, l1)
        print(f"  {arm:<16} {tag:<8} {a:>7.4f} {b:>7.4f} {l1:>9.5f}", flush=True)
    p, s = out[(arm, "plane")], out[(arm, "slab")]
    print(f"  {'':<16} {'delta':<8} {s[0]-p[0]:>+7.4f} {s[1]-p[1]:>+7.4f} {s[2]-p[2]:>+9.5f}")
json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, open(f"{W}/out/slabeval.json", "w"))
print("SLABEVAL_OK")
