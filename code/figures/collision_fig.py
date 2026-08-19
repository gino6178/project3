"""M7's contact test, drawn: the same slab at four separations, with its own counts.

physics.py prints these numbers and nothing in this repository draws them, so the figure the
page carries was made by a script that was never published. This is that script, and it drives
physics.py's own index, bodies and contact test on the same lattice by the same construction,
so the picture and the numbers cannot drift apart.

    python code/figures/collision_fig.py LATTICE OUT.png
"""
import os as _os, sys
import numpy as np, torch
from plyfile import PlyData
from PIL import Image, ImageDraw, ImageFont

_H = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path[:0] = [_H + "/evaluate", _H + "/src"]
import physics as ph, subdivide as sd
from occupancy import to_grid, close_and_fill

def main(ld, out):
    lat = torch.load(_os.path.join(ld, "lattice.pt")); hc, hf = float(lat["coarse_dx"]), float(lat["fine_dx"])
    el = PlyData.read(_os.path.join(ld, "gs_fill.ply")).elements[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    lvl = torch.load(_os.path.join(ld, "cell_level.pt")).reshape(-1)
    keep = (lvl[:len(xyz)] == 0).numpy()
    org = xyz[keep].min(0) - 0.5 * hc
    coords = np.unique(np.floor((xyz[keep] - org) / hc).astype(np.int64), axis=0)
    occ, _, _ = to_grid(torch.from_numpy(coords).float(), 1.0)
    solid = close_and_fill(occ, 1).nonzero().numpy() + coords.min(0) - 1
    n = np.array([0.13, 0.97, -0.21]); n /= np.linalg.norm(n)
    d = float(-((solid + 0.5) * hc).mean(0) @ n) + 0.37 * hc
    r = sd.cut(solid, hc, n, d, hf)
    ix = ph.CollisionIndex(r, hc, org=org, plane=(n, d))
    A, B = ph.Body(ix, 0), ph.Body(ix, 1)
    truth = np.where(np.sign((xyz - org) @ n + d) == A.side, A.piece, B.piece)
    pa = xyz[truth == A.piece]

    S=520; PADX=30; CAP=56; HDR=28
    FT="/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans%s.ttf"
    try: FB,FR = ImageFont.truetype(FT%"-Bold",16), ImageFont.truetype(FT%"",15)
    except Exception: FB=FR=ImageFont.load_default()
    e1 = np.cross(n, [0,0,1.0]); e1/=np.linalg.norm(e1)
    q = np.stack([(pa-org)@e1, (pa-org)@n], 1)
    tiles=[]
    for push in (0.0, 1.0, 2.0, 4.0):
        Am = A.move(t=n * -A.side * push * hc) if push else A
        hit,_ = ph.contact(Am, B, pa)
        raw,_ = ix.occupied(pa, B.piece) if push==0 else (hit, None)
        im=Image.new("RGB",(S,S),(255,255,255)); px=im.load()
        rr=np.abs(q).max()*1.04
        X=((q[:,0]/rr*0.5+0.5)*S).astype(int).clip(0,S-1)
        Y=((-q[:,1]/rr*0.5+0.5)*S).astype(int).clip(0,S-1)
        for x,y,hh in zip(X,Y,hit):
            px[x,y]=(198,38,38) if hh else (176,176,176)
        a=np.asarray(im).min(2); ys,xs=np.where(a<250)
        if len(ys):
            pd=int(0.04*max(np.ptp(ys),np.ptp(xs)))
            im=im.crop((max(xs.min()-pd,0),max(ys.min()-pd,0),
                        min(xs.max()+pd,S),min(ys.max()+pd,S)))
            k=S/max(im.size); im=im.resize((int(im.width*k),int(im.height*k)),Image.LANCZOS)
            sq=Image.new("RGB",(S,S),(255,255,255)); sq.paste(im,((S-im.width)//2,(S-im.height)//2)); im=sq
        tiles.append((im,int(raw.sum()),int(hit.sum()),push))
    Wt=len(tiles)*S+(len(tiles)-1)*PADX
    G=Image.new("RGB",(Wt,HDR+S+CAP),(255,255,255)); dr=ImageDraw.Draw(G)
    for k,(im,rawn,hitn,push) in enumerate(tiles):
        x=k*(S+PADX)
        dr.text((x,4), f"pushed in {push:.0f} coarse cells", font=FB, fill=(17,17,17))
        G.paste(im,(x,HDR))
        if push==0:
            dr.text((x,HDR+S+12), f"occupancy alone {rawn:,}", font=FR, fill=(120,120,120))
        dr.text((x,HDR+S+(32 if push==0 else 12)), f"with the plane test {hitn:,}", font=FR, fill=(17,17,17))
    G.save(out); print("->",out,G.size)
    for im,rawn,hitn,push in tiles: print(f"  push {push:.0f}: plane {hitn:,}" + (f"   occupancy {rawn:,}" if push==0 else ""))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
