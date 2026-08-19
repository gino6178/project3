"""The exterior of each arm, from the six cameras that supervised it.

`ovnative.render_exterior` has existed since the multi-view step and nothing had ever called it, so
no arm in this study had been drawn from outside. The six cameras are `e_mvp` in cams_mv.npz -- the
ones the exterior views used during training -- not new viewpoints.

Two things this is here to check, beyond showing the peel:

  SHELL_PIN   a pinned arm must reproduce its input exactly. `col_pin` overwrites the decoded
              colour after the head, and the dual geometry is out of the optimiser, so the rendered
              exterior of r1_pin, r1flat_pin and r1_pin_full should be bit-identical to each other
              and to the seed they all share. If it is not, that is a bug and this prints it.
  the routes  route 1 takes the exterior from the released model, route 2 projects it from six
              photographs. The difference has never been looked at.

State comes from the cached `state_r*.pt` (03:51) plus each arm's own `params.pt`, NOT from
`build_orange/lattice`: the other tenant's rerun.sh rebuilt that lattice at 05:15 with a flat
interior, so it is no longer the lattice these arms were built on.
"""
import json, os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON
import nvdiffrast.torch as dr
import anchor

W = "/workspace/ovoxel_native"
OUT = W + "/out/ext"
dev = "cuda"
RES = int(os.environ.get("RES", "512"))
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)

ARMS = [("r1_pin", "1"), ("r1flat_pin", "1"), ("r1_pin_full", "1"), ("r2_pin_full", "2"),
        ("r1_free", "1"), ("r2_free", "2")]
STATES, CAMS = {}, {}
for r in ("1", "2"):
    STATES[r] = torch.load(f"{W}/state_r{r}.pt", map_location=dev, weights_only=False)
    CAMS[r] = np.load(f"{W}/cams_mv.npz" if r == "1" else f"{W}/cams_mv_r2.npz")
NAMES = [str(x) for x in CAMS["1"]["e_names"]]
print(f"six exterior cameras: {NAMES}")
print(f"route 1: {len(STATES['1']['dual_v']):,} dual vertices; "
      f"route 2: {len(STATES['2']['dual_v']):,}")


def render(st, cams, folder):
    os.makedirs(folder, exist_ok=True)
    outs = {}
    with torch.no_grad():
        for i, nm in enumerate(NAMES):
            mvp = torch.as_tensor(cams["e_mvp"][i], dtype=torch.float32, device=dev)
            img, _, _, _ = ON.render_exterior(st, glctx, mvp, RES)
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            cv2.imwrite(f"{folder}/{nm}.png", (a[:, :, ::-1] * 255).astype(np.uint8))
            outs[nm] = a
    return outs


# the seed exterior of each route: what a pinned arm must reproduce
seeds = {}
for r in ("1", "2"):
    st = STATES[r]
    st["surf_rgb"] = st["surf_rgb"].detach()
    seeds[r] = render(st, CAMS[r], f"{OUT}/seed_r{r}")
    print(f"  seed r{r} -> {OUT}/seed_r{r}")

got = {}
for arm, r in ARMS:
    p = f"{W}/{arm}/params.pt"
    if not os.path.exists(p):
        print(f"  {arm}: no params.pt, skipped")
        continue
    st = {k: v for k, v in STATES[r].items()}
    P = torch.load(p, map_location=dev)
    ds = anchor.ColourDecoder(len(STATES[r]["surf_rgb"])).to(dev)
    ds.load_state_dict(P["dec_s"])
    # `pin` is a plain attribute, not part of the state_dict, so loading the weights does not
    # restore it -- and without it this draws the decoder's own output where the run drew the
    # pinned seed. Which arms were pinned is recorded in each run's hist.json.
    pinned = bool(json.load(open(f"{W}/{arm}/hist.json"))["shell_pin"])
    if pinned:
        ds.pin_colour(torch.ones(len(STATES[r]["surf_rgb"]), dtype=torch.bool, device=dev),
                      STATES[r]["surf_rgb"].detach())
    with torch.no_grad():
        st["surf_rgb"] = ds()
    st["dual_v"] = P["dual_v"].to(dev)
    st["split_w"] = P["split_w"].to(dev)
    got[arm] = render(st, CAMS[r], f"{OUT}/{arm}")
    d = np.mean([np.abs(got[arm][n] - seeds[r][n]).mean() for n in NAMES]) * 255
    mx = np.max([np.abs(got[arm][n] - seeds[r][n]).max() for n in NAMES]) * 255
    print(f"  {arm:<14} (route {r})  SHELL_PIN={int(pinned)}  "
          f"mean |exterior - seed| = {d:8.5f}/255  max = {mx:8.4f}/255")

print("\nSHELL_PIN check: the pinned route-1 arms share one exterior seed")
base = "r1_pin"
for arm in ("r1flat_pin", "r1_pin_full"):
    if arm in got and base in got:
        d = np.mean([np.abs(got[arm][n] - got[base][n]).max() for n in NAMES]) * 255
        print(f"  max |{arm} - {base}| over the six views = {d:.6f}/255")
print("EXT_OK")
