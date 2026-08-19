"""Figure 13, from phaseopt's own samplers and the phases it actually solved.

    python code/figures/chords_fig.py H_DIR V_DIR OUT.png
"""
import os as _os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
_H=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0,_H+"/src")
import phaseopt as po

def strip(im, S):
    if isinstance(im, np.ndarray):
        im = Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8) if im.max()<=1.5 else im.astype(np.uint8))
    a=np.asarray(im.convert("RGB")); v=a.min(2); ys,xs=np.where(v<245)
    pd=int(0.03*max(np.ptp(ys),np.ptp(xs)))
    c=im.crop((max(xs.min()-pd,0),max(ys.min()-pd,0),min(xs.max()+pd,im.width),min(ys.max()+pd,im.height)))
    k=S/max(c.size); c=c.resize((int(c.width*k),int(c.height*k)),Image.LANCZOS)
    sq=Image.new("RGB",(S,S),(255,255,255)); sq.paste(c,((S-c.width)//2,(S-c.height)//2)); return sq

def plot(a,b,S,lo,hi,label):
    im=Image.new("RGB",(S,S),(255,255,255)); d=ImageDraw.Draw(im)
    m=int(S*0.10)
    d.rectangle([m,m,S-m,S-m], outline=(205,205,205))
    def poly(y,col):
        pts=[(m+(S-2*m)*i/(len(y)-1), S-m-(S-2*m)*float(np.clip((y[i]-lo)/(hi-lo),0,1))) for i in range(len(y))]
        d.line(pts, fill=col, width=3)
    poly(a,(40,90,170)); poly(b,(190,50,40))
    return im

def main(hd, vd, out):
    H,hp=po.load(hd); V,vp=po.load(vd)
    zs=np.linspace(-0.55,0.55,len(H)); azs=np.pi*np.arange(len(V))/len(V)
    # greedy (11): cross-correlate each against the first of its family
    def ang_profile(im, nb=180):
        c,r=po.disc(im); t=np.linspace(0.15,0.95,40)[None,:]
        a=(2*np.pi*np.arange(nb)/nb)[:,None]
        pts=np.stack([(c[0]+r*t*np.cos(a)).ravel(),(c[1]+r*t*np.sin(a)).ravel()],1)
        v=po.bilinear(im, pts).mean(1).reshape(nb,-1).mean(1)
        return v-v.mean()
    r0=ang_profile(H[0]); greedy=np.zeros(len(H))
    for i in range(1,len(H)):
        cc=np.fft.irfft(np.fft.rfft(r0)*np.conj(np.fft.rfft(ang_profile(H[i]))),n=len(r0))
        greedy[i]=2*np.pi*int(np.argmax(cc))/len(r0)
    z=np.load(_os.path.join(hd,"phase_opt.npz")) if _os.path.exists(_os.path.join(hd,"phase_opt.npz")) else None
    joint=z["phases"] if z is not None else greedy
    cg=po.cost(H,V,greedy,zs,azs); cj=po.cost(H,V,joint,zs,azs)
    i,j=0,0
    hg=po.h_line(H[i], azs[j]+greedy[i]); hj=po.h_line(H[i], azs[j]+joint[i]); vl=po.v_line(V[j], zs[i])
    lo=min(hg.min(),hj.min(),vl.min()); hi=max(hg.max(),hj.max(),vl.max())
    S=380; GAP=22; CAP=34
    FT="/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans%s.ttf"
    FB=ImageFont.truetype(FT%"-Bold",14); FR=ImageFont.truetype(FT%"",13)
    tiles=[("a transverse reference", strip(H[i],S)),
           ("a longitudinal reference", strip(V[j],S)),
           (f"under the greedy phase of (11)   mean |difference| {cg:.4f}", plot(hg.mean(1),vl.mean(1),S,lo,hi,"")),
           (f"under the phases (27) solves   mean |difference| {cj:.4f}", plot(hj.mean(1),vl.mean(1),S,lo,hi,""))]
    G=Image.new("RGB",(len(tiles)*S+(len(tiles)-1)*GAP, S+CAP),(255,255,255)); d=ImageDraw.Draw(G)
    for k,(t,im) in enumerate(tiles):
        x=k*(S+GAP); G.paste(im,(x,0)); d.text((x,S+10),t,font=FR,fill=(60,60,60))
    G.save(out); print("->",out,G.size); print(f"  greedy {cg:.5f}   joint {cj:.5f}   {100*(cg-cj)/cg:.1f}% less")

if __name__=="__main__": main(sys.argv[1],sys.argv[2],sys.argv[3])
