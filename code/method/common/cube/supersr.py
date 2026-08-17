"""Cut-time super-resolution, baked into the band rather than applied to the frame.

Section 1.2's fifth omission, and the other half of section 12.2's first limitation: local
subdivision improves geometry and topology, but a child inherits its parent's feature, so a finer
cut boundary is finer geometry and the same appearance. Generating detail at cut time is the
obvious answer and the obvious way to do it is wrong -- a pass over the rendered frame is a
screen-space filter, so it changes when the camera moves and it does not travel with the piece
when the piece does.

So the generated face is written back into the cells. Render the exposed section down its own
normal, generate from it once, and give every cut leaf the colour of the pixel it projects to.
After that the detail is in the volume: it is the same from any angle, it moves with the piece,
and nothing regenerates per frame.

What that buys is bounded, and the bound is the point of measuring it. A leaf can hold one
colour, so detail finer than a leaf cannot survive the write-back no matter how good the
generator is -- super-resolution and subdivision are one budget. The retention curve against
pixels per leaf is what says where that budget runs out, and it is measured here rather than
argued.

    python method/common/cube/supersr.py LATTICE OUT_DIR
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys

import numpy as np
import torch

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

from method.common.cube import cutmesh as cm                        # noqa: E402
from method.common.cube import subdivide as sd                      # noqa: E402

DEV = "cuda:0"
STRENGTH = float(_os.environ.get("SR_STRENGTH", "0.45"))   # see the sweep below
GUID = float(_os.environ.get("SR_GUIDANCE", "9"))
NEG = "watermark, text, letters, logo, signature, border, frame"


def face_image(r, h, n, d, colour, size, pad=1.03):
    """The exposed section, drawn down its own normal by leaf lookup.

    Orthographic and axis-free: the plane's own basis is the image's, so a pixel is a point on
    the plane and the leaf containing it is the colour. No camera and no rasteriser, which keeps
    the write-back the exact inverse of the render.
    """
    nn, u, v = cm.plane_basis(n)
    hl = h / (2.0 ** r["level"].astype(np.float64))
    ctr = ((r["leaf"] + 0.5) * hl[:, None]).mean(0)
    ctr = ctr - (ctr @ nn + d) * nn
    rad = np.linalg.norm(((r["leaf"] + 0.5) * hl[:, None]) - ctr, axis=1).max() * pad

    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    gx, gy = np.meshgrid(t, t, indexing="xy")
    pts = ctr[None] + gx.reshape(-1, 1) * (u * rad) + gy.reshape(-1, 1) * (v * rad)
    lf = _leaf_at(pts, r, h)
    img = np.ones((size * size, 3), np.float32)
    hit = lf >= 0
    img[hit] = colour[lf[hit]]
    return img.reshape(size, size, 3), hit.reshape(size, size), (ctr, u, v, rad, lf)


def _leaf_at(pts, r, h):
    out = np.full(len(pts), -1, np.int64)
    for L in sorted({int(x) for x in r["level"]}, reverse=True):
        m = r["level"] == L
        if not m.any():
            continue
        c = np.floor(pts / (h / (2.0 ** L))).astype(np.int64)
        mn = int(min(r["leaf"][m].min(), c.min())) - 2
        span = int(max(r["leaf"][m].max(), c.max()) + (-mn) + 3)
        k = sd._pack(r["leaf"][m], -mn, span)
        o = np.argsort(k)
        ks, idx = k[o], np.nonzero(m)[0][o]
        q = np.clip(c, mn, mn + span - 1)
        kk = sd._pack(q, -mn, span)
        pos = np.clip(np.searchsorted(ks, kk), 0, len(ks) - 1)
        got = (ks[pos] == kk) & (out < 0)
        out[got] = idx[pos[got]]
    return out


def bake(gen, lf, colour, size):
    """Every leaf takes the mean of the pixels that landed on it.

    The mean and not a sample: a leaf covering several pixels is being asked to hold one colour,
    and averaging is what "one colour" means. It is also exactly where the detail finer than a
    leaf goes, which is the budget this file is here to measure.
    """
    out = colour.copy()
    g = gen.reshape(-1, 3)
    hit = lf >= 0
    acc = np.zeros_like(colour)
    cnt = np.zeros(len(colour))
    np.add.at(acc, lf[hit], g[hit])
    np.add.at(cnt, lf[hit], 1.0)
    m = cnt > 0
    out[m] = acc[m] / cnt[m][:, None]
    return out, int(m.sum())


def detail(img, mask):
    """High-frequency energy inside the section."""
    import cv2
    g = img.mean(2)
    lo = cv2.GaussianBlur(g, (0, 0), 2.0)
    return float(np.abs(g - lo)[mask].mean())


def retained(base, gen, baked, mask):
    """How much closer to the generated face the baked one is than the base was.

    The obvious measure -- added detail recovered, (d_baked - d_base) / (d_gen - d_base) -- is
    unusable here and it took a nonsense number to notice: at strength 0.45 the generator
    *removes* high-frequency energy from this face (0.01153 against 0.01238), because a cube face
    at three pixels per leaf is full of cell edges and the sampler smooths them. The denominator
    then goes through zero and the ratio reported eight-digit percentages.
    
    Distance to the generated image has no such singularity: 1 means the baked face is the
    generated one, 0 means the write-back changed nothing.
    """
    a = np.linalg.norm((base - gen)[mask])
    b = np.linalg.norm((baked - gen)[mask])
    return float(1.0 - b / max(a, 1e-9))


def main(lattice_dir, out_dir, size=768, colour_from=None):
    import cv2
    from plyfile import PlyData
    from diffusers import StableDiffusionDepth2ImgPipeline
    from PIL import Image
    from method.common.cube.occupancy import close_and_fill, to_grid
    from method.common.cube import multicut as mc

    _os.makedirs(out_dir, exist_ok=True)
    lat = torch.load(_os.path.join(lattice_dir, "lattice.pt"))
    hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(lattice_dir, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    C0 = 0.28209479177387814
    rgb = np.clip(np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float32)
                  * C0 + 0.5, 0, 1)
    # This one is about appearance, so the lattice's own colour is the wrong source and silently
    # so: a lattice built by make_shape carries flat grey in its interior, and super-resolution
    # measured on a face with no texture reports how much of nothing it preserved. Every panel of
    # the generated route came out uniform grey and the retention numbers -- 100.0% at the
    # coarsest setting, falling to 57.5% -- were describing quantisation noise on a constant.
    if colour_from and _os.path.isfile(colour_from):
        e2 = PlyData.read(colour_from).elements[0]
        if len(e2["x"]) == len(xyz):
            rgb = np.clip(np.stack([e2["f_dc_0"], e2["f_dc_1"], e2["f_dc_2"]], 1)
                          .astype(np.float32) * C0 + 0.5, 0, 1)
            print(f"  colour from {colour_from}")
        else:
            print(f"  {colour_from} has {len(e2['x']):,} rows against {len(xyz):,}; "
                  f"keeping the lattice's colour")
    lvl = torch.load(_os.path.join(lattice_dir, "cell_level.pt")).reshape(-1)[:len(xyz)].numpy()
    # from a corner, not from a centre: floor of a centre sitting exactly on a cell
    # boundary lets floating point choose the side, and on a lattice whose cells are at
    # (i + 1/2)h that discards 49% of them. Offset by half the finest spacing used here.
    org = xyz[lvl == 0].min(0) - 0.5 * hf
    coords, first = np.unique(np.floor((xyz[lvl == 0] - org) / hc).astype(np.int64),
                              axis=0, return_index=True)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1
    c = ((solid + 0.5) * hc).mean(0)
    n = np.array([0.0, 1.0, 0.0])
    d = float(-c @ n) + 0.37 * hc

    r = mc.cut(solid, hc, [(n, d)], hf)
    # (19): a child's colour is its parent's, which is the state this file starts from
    from scipy.spatial import cKDTree
    tree = cKDTree((coords + 0.5) * hc)
    hl = hc / (2.0 ** r["level"].astype(np.float64))
    _, par = tree.query((r["leaf"] + 0.5) * hl[:, None], k=1)
    colour = np.clip(rgb[lvl == 0][first][par], 0, 1)
    print(f"  {len(solid):,} solid cells, {len(r['leaf']):,} leaves, {r['K']} pieces")

    base, mask, (ctr, u, v, rad, lf) = face_image(r, hc, n, d, colour, size)
    cv2.imwrite(_os.path.join(out_dir, "0_before.png"),
                (base[:, :, ::-1] * 255).astype(np.uint8))

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-depth", torch_dtype=torch.float16).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    dep = torch.from_numpy(mask.astype(np.float32))[None, None].to(DEV)
    dep = torch.nn.functional.interpolate(dep, size=(512, 512), mode="nearest")[0].half()
    gen = pipe(prompt=_os.environ.get("SR_PROMPT",
                                      "the cross-section of a navel orange, juice vesicles, "
                                      "white radial segment membranes, macro photograph"),
               image=Image.fromarray((base * 255).astype(np.uint8)).resize((512, 512)),
               depth_map=dep, negative_prompt=NEG, strength=STRENGTH, guidance_scale=GUID,
               num_inference_steps=50,
               generator=torch.Generator(DEV).manual_seed(1234), return_dict=False)
    gimg = gen[0][0] if isinstance(gen, tuple) else gen.images[0]
    gimg = np.asarray(gimg.resize((size, size)), np.float32) / 255.0
    cv2.imwrite(_os.path.join(out_dir, "1_generated.png"),
                (gimg[:, :, ::-1] * 255).astype(np.uint8))

    colour2, nset = bake(gimg, lf, colour, size)
    baked, _, _ = face_image(r, hc, n, d, colour2, size)
    cv2.imwrite(_os.path.join(out_dir, "2_baked.png"),
                (baked[:, :, ::-1] * 255).astype(np.uint8))

    d0, dg, db = detail(base, mask), detail(gimg, mask), detail(baked, mask)
    px_per_leaf = size / (2 * rad / hf)
    print(f"  {nset:,} leaves took a colour; the face is {2 * rad:.4f} across, "
          f"{px_per_leaf:.2f} pixels per leaf at {size}")
    print(f"  detail: before {d0:.5f}, generated {dg:.5f}, baked {db:.5f}")
    print(f"  the baked face is {100 * retained(base, gimg, baked, mask):.1f}% of the way from "
          f"the original to the generated one")

    # the budget, swept: rendering the face finer than a leaf cannot store more
    print(f"  {'render size':>12}{'px per leaf':>13}{'toward gen':>12}{'detail kept':>13}")
    for s in (256, 384, 512, 768, 1024, 1536):
        b2, m2, (_, _, _, r2, lf2) = face_image(r, hc, n, d, colour, s)
        g2 = np.asarray(Image.fromarray((gimg * 255).astype(np.uint8)).resize((s, s)),
                        np.float32) / 255.0
        c2, _ = bake(g2, lf2, colour, s)
        bk, _, _ = face_image(r, hc, n, d, c2, s)
        ddg, ddb = detail(g2, m2), detail(bk, m2)
        # the second column passes 100% at high resolution and that is not detail gained: a
        # baked face is drawn from cells, so its own boundaries add high-frequency energy the
        # generated image does not have
        print(f"  {s:>12}{s / (2 * r2 / hf):>13.2f}"
              f"{100 * retained(b2, g2, bk, m2):>11.1f}%{100 * ddb / max(ddg, 1e-9):>12.1f}%")

    # and it is in the volume, not on the frame: the same cells seen from a tilted plane
    m = np.array([0.10, 0.98, -0.17]); m /= np.linalg.norm(m)
    r2 = mc.cut(solid, hc, [(m, float(-c @ m) + 0.37 * hc)], hf)
    _, par2 = tree.query((r2["leaf"] + 0.5)
                         * (hc / (2.0 ** r2["level"].astype(np.float64)))[:, None], k=1)
    # carry the baked colours across by position, which is what "in the volume" means
    t2 = cKDTree((r["leaf"] + 0.5) * hl[:, None])
    _, j = t2.query((r2["leaf"] + 0.5)
                    * (hc / (2.0 ** r2["level"].astype(np.float64)))[:, None], k=1)
    tilt, mk2, _ = face_image(r2, hc, m, float(-c @ m) + 0.37 * hc, colour2[j], size)
    cv2.imwrite(_os.path.join(out_dir, "3_tilted.png"),
                (tilt[:, :, ::-1] * 255).astype(np.uint8))
    # And this is where it stops. The write-back reached the cells the face passed through and
    # no others, so a plane six degrees away meets mostly cells that never received it and shows
    # the volume as it was. The scalar does not say so -- detail comes back at 79% of the baked
    # face's, because a cube face's own cell edges dominate that measure -- and the picture does:
    # `3_tilted.png` has two streaks where `2_baked.png` has thirty membranes. The enhancement is
    # per cut, which is what section 12.2 says it is.
    print(f"  a plane 6 degrees away, drawn from the same cells: detail {detail(tilt, mk2):.5f} "
          f"against the baked face's {db:.5f} -- but look at 3_tilted.png, the membranes are "
          f"not there; the detail scalar is measuring cell edges")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sr_out",
         colour_from=sys.argv[3] if len(sys.argv) > 3 else None)
