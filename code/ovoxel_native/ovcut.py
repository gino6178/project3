"""A trained object being cut, drawn entirely from the O-Voxel representation.

    python ovcut.py OBJ OUTDIR [TAG]

Every pixel here comes from the three tensors and the occupancy: the exterior is the dual grid's
own surface through nvdiffrast, the cut face is the closed-form polygon of the plane against the
occupancy coloured by the interior field, and the two go into one rasteriser pass so the depth test
composes them. No Gaussian is rendered and nothing is blended between two renderers -- which is the
reason the exterior was converted in the first place.

Two sweeps per object, because the two families are not the same question. The transverse plane is
perpendicular to the polar axis and walks down it; the longitudinal plane contains the axis and
walks across. Each runs there and back so the loop has no jump.

The plane's range is read from the object rather than set: `render_section` keeps the part of the
surface with (x . n + d) < 0, so d from -max(x . n) to -min(x . n) takes it from nothing removed to
everything removed, and the sweep is that interval with a margin at each end.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor                                                        # noqa: E402
import ovnative as ON                                                # noqa: E402

W = "/workspace/ovoxel_native"
RES = int(os.environ.get("CUT_RES", "512"))
NFRAME = int(os.environ.get("CUT_FRAMES", "48"))
FPS = int(os.environ.get("CUT_FPS", "24"))
dev = "cuda"


def load(obj, tag="ov"):
    """The trained object: its state, with the decoders' output written where the renderer reads."""
    st = torch.load(f"{W}/state_{obj}.pt", map_location=dev, weights_only=False)
    p = torch.load(f"{W}/{tag}_{obj}/params.pt", map_location=dev, weights_only=False)
    st["dual_v"] = p["dual_v"].to(dev)
    st["split_w"] = p["split_w"].to(dev)
    if "dec_i" in p:
        for key, sd in (("interior", p["dec_i"]), ("surf_rgb", p["dec_s"])):
            d = anchor.ColourDecoder(len(st[key]), init_rgb=st[key]).to(dev)
            d.load_state_dict(sd)
            with torch.no_grad():
                st[key] = d()
    else:
        st["interior"] = p["interior"].to(dev)
        st["surf_rgb"] = p["surf_rgb"].to(dev)
    return st


def depth_range(st, n, inset=0.03):
    """The interval of d that walks the plane through the object, kept just inside both ends.

    `render_section` keeps (x . n + d) < 0, so d = -max(x . n) keeps everything and d = -min keeps
    nothing -- and nothing is not a frame, it is an empty rasteriser call. The interval is inset at
    both ends so every frame has a surface on one side and a cut face on the other.
    """
    c = (st["solid"].float().to(dev) + 0.5) * float(st["hc"]) \
        + torch.as_tensor(st["org"], dtype=torch.float32, device=dev)
    pr = (c @ n).cpu().numpy()
    lo, hi = float(pr.min()), float(pr.max())
    m = inset * (hi - lo)
    return -(hi - m), -(lo + m)


def sweep(st, glctx, mvp, n):
    """The frames of one plane walking through the object, there and back."""
    d0, d1 = depth_range(st, n)
    ds = np.linspace(d0, d1, NFRAME)
    ds = np.concatenate([ds, ds[::-1][1:-1]])          # there and back, no repeated end frame
    frames = []
    with torch.no_grad():
        for d in ds:
            try:
                img, _, _, _ = ON.render_section(st, glctx, mvp, n, float(d), RES)
            except RuntimeError as e:
                # a depth at which the plane happens to enclose no geometry at all: skip it rather
                # than write a frame of background, which would read as a flicker in the loop
                print(f"    d {d:+.4f} produced nothing ({str(e)[:40]}), skipped")
                continue
            a = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            frames.append((a * 255).astype(np.uint8))
    return frames


def main(obj, outdir, tag="ov"):
    import nvdiffrast.torch as dr
    os.makedirs(outdir, exist_ok=True)
    ON.FDG = ON._load_ovoxel()
    st = load(obj, tag)
    # an arm trained at a different plane count carries its own cameras, and drawing it
    # through the default set would show a cut the model was never given
    C = np.load(os.environ.get("CAMS", f"{W}/cams_{obj}.npz"))
    glctx = dr.RasterizeCudaContext(device=dev)
    print(f"  {obj}: {len(st['interior']):,} interior cells, {len(st['dual_v']):,} dual vertices")

    hm = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
    hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
    a = sweep(st, glctx, hm, hn)

    k = len(C["v_mvp"]) // 2
    vm = torch.as_tensor(C["v_mvp"][k], dtype=torch.float32, device=dev)
    vn = torch.as_tensor(C["v_planes"][k, :3], dtype=torch.float32, device=dev)
    b = sweep(st, glctx, vm, vn)

    # One file per object with the two families side by side, rather than two files: they are the
    # same object under the two cuts and the comparison is the point, and fourteen autoplaying
    # videos on one page is a page nobody waits for.
    import imageio.v2 as imageio
    n = min(len(a), len(b))
    pair = [np.concatenate([a[i], b[i]], 1) for i in range(n)]
    out = os.path.join(outdir, f"ovcut_{obj}.mp4")
    imageio.mimwrite(out, pair, fps=FPS, codec="libx264", quality=7,
                     macro_block_size=1, output_params=["-pix_fmt", "yuv420p"])
    print(f"    -> {out}  {n} frames, {pair[0].shape[1]}x{pair[0].shape[0]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ov")
