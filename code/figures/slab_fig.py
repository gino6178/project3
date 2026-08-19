"""Figure 9: one slab through four representations, every primitive at its own footprint.

A Gaussian is drawn at its own sigma and a cell at its own h, so the picture answers what each
representation actually puts in the volume rather than how each one's renderer resolves it.
Nothing in this repository drew it before.

    python code/figures/slab_fig.py OUT.png  name=MODEL.ply[:LATTICE] ...
"""
import os as _os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

def load(spec):
    """A ply, with a per-primitive footprint: exp(scale) for a Gaussian, h for a lattice cell."""
    path, _, lat = spec.partition(":")
    el = PlyData.read(path)["vertex"]; names = el.data.dtype.names
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    f = [n for n in names if n.startswith("f_dc_")][:3]
    rgb = np.clip(np.stack([el[n] for n in f], 1) * 0.28209479 + 0.5, 0, 1) if f else np.full((len(xyz),3),0.6)
    sc = [n for n in names if n.startswith("scale_")][:3]
    if lat:
        import torch
        h = float(torch.load(_os.path.join(lat, "lattice.pt"), map_location="cpu")["coarse_dx"])
        r = np.full(len(xyz), h * 0.5)
    else:
        r = np.exp(np.stack([el[n] for n in sc], 1)).mean(1) if sc else np.full(len(xyz), 1e-3)
    return xyz, rgb, r

def slab(spec, S=560, frac=0.035):
    xyz, rgb, r = load(spec)
    c = np.median(xyz, 0); q = xyz - c
    ext = np.percentile(np.abs(q), 98)
    keep = np.abs(q[:, 1]) < ext * frac
    q, rgb, r = q[keep], rgb[keep], r[keep]
    im = Image.new("RGB", (S, S), (255, 255, 255)); d = ImageDraw.Draw(im, "RGBA")
    sc = S * 0.5 / (ext * 1.06)
    o = np.argsort(-r)                       # big first, so small primitives stay visible
    for i in o:
        x = S/2 + q[i,0]*sc; y = S/2 - q[i,2]*sc; rr = max(r[i]*sc, 0.45)
        col = tuple(int(v*255) for v in rgb[i])
        d.ellipse([x-rr, y-rr, x+rr, y+rr], fill=col + (150,))
    return im, int(keep.sum()), float(np.median(r))

def main(out, *specs):
    FT="/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans%s.ttf"
    try: FB, FR = ImageFont.truetype(FT%"-Bold",16), ImageFont.truetype(FT%"",14)
    except Exception: FB=FR=ImageFont.load_default()
    S=560; GAP=22; HDR=26; CAP=44
    tiles=[]
    for s in specs:
        name, _, rest = s.partition("=")
        im, n, med = slab(rest, S)
        tiles.append((name, im, n, med)); print(f"  {name:<26} {n:,} primitives in the slab, median footprint {med:.3e}")
    G=Image.new("RGB",(len(tiles)*S+(len(tiles)-1)*GAP, HDR+S+CAP),(255,255,255)); d=ImageDraw.Draw(G)
    for k,(name,im,n,med) in enumerate(tiles):
        x=k*(S+GAP); d.text((x,4),name,font=FB,fill=(17,17,17)); G.paste(im,(x,HDR))
        d.text((x,HDR+S+10), f"{n:,} primitives", font=FR, fill=(95,95,95))
        d.text((x,HDR+S+28), f"median footprint {med:.2e}", font=FR, fill=(95,95,95))
    G.save(out); print("->",out,G.size)

if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
