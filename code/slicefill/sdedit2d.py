# Does the 2-D prior restore a cut face?  Take a plane from the O-Voxel, noise it to t, and run
# the full reverse chain on that single image.  No volume, no write-back -- this tests the prior
# alone, which has to work before the 3-D chain can mean anything.
import os,sys,math,torch
SP=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SP); os.chdir(SP)
exec(open(SP+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image,ImageDraw
dv="cuda:1"; T=1000
ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
vol=Vol(os.environ.get("GRID",SP+"/grid128.pt"),dv)   # the grid must match the prior it is testing
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(","))
def load(p):
    m=UNet2D(64,MULT).to(dv); m.load_state_dict(torch.load(p,map_location=dv)); m.eval(); return m
MV=load(os.environ["CKV"]); MH=load(os.environ["CKH"])
LEVELS=[float(x) for x in os.environ.get("LEVELS","0.3,0.5,0.7").split(",")]
NSTEP=int(os.environ.get("NSTEP","200"))
def restore(img,model,t0f,mask=None):
    """mask=1 where the interior is.  Everything else -- shell and background -- is observed, so
    it is never noised and is restored after every step: the prior sees it as fixed context and
    only has to invent what is actually unknown."""
    torch.manual_seed(0)
    iT=max(int(t0f*(T-1)),1)
    M=torch.ones_like(img[None]) if mask is None else mask[None]
    x=img[None]*(1-M)+(ab[iT].sqrt()*img[None]+(1-ab[iT]).sqrt()*torch.randn_like(img[None]))*M
    noisy=x[0].clone()
    ts=[int(round(v)) for v in torch.linspace(iT,0,NSTEP+1).tolist()]
    for tc,tn in zip(ts[:-1],ts[1:]):
        with torch.no_grad(): e=model(x,torch.full((1,),tc,device=dv,dtype=torch.long))
        x0=((x-(1-ab[tc]).sqrt()*e)/ab[tc].sqrt()).clamp(-1,1)
        x0=img[None]*(1-M)+x0*M
        if tn<=0: x=x0; break
        x=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*e
        x=img[None]*(1-M)+x*M
    return noisy,x[0]
def im(t):
    return Image.fromarray(((t.clamp(-1,1)*0.5+0.5)*255).byte().permute(1,2,0).cpu().numpy())
NOMASK=int(os.environ.get("NOMASK","0"))   # test an unmasked-trained prior the way it was trained
C3=vol.CORE.expand(3,-1,-1,-1).contiguous()
rows=[("longitudinal",vol.long_slice(vol.V0,0.7),MV,(vol.long_slice(C3,0.7)>0.5).float()),
      ("transverse",  vol.trans_slice(vol.V0,vol.N//2),MH,(vol.trans_slice(C3,vol.N//2)>0.5).float())]
W=200; sheet=Image.new("RGB",(W*(1+2*len(LEVELS))+20,2*W+46),(255,255,255))
d=ImageDraw.Draw(sheet)
for r,(name,img,model,MK) in enumerate(rows):
    y=26+r*(W+10)
    sheet.paste(im(img).resize((W,W)),(6,y))
    if r==0: d.text((8,8),"O-Voxel plane",fill=(30,30,30))
    for c,lv in enumerate(LEVELS):
        n,o=restore(img,model,lv,None if NOMASK else MK)
        sheet.paste(im(n).resize((W,W)),(6+W*(1+2*c),y))
        sheet.paste(im(o).resize((W,W)),(6+W*(2+2*c),y))
        if r==0:
            d.text((8+W*(1+2*c),8),f"noised t0={lv}",fill=(150,60,40))
            d.text((8+W*(2+2*c),8),"restored",fill=(30,110,60))
    err=[]
    for lv in LEVELS:
        _,o=restore(img,model,lv,None if NOMASK else MK); err.append(float(((o-img).abs()*MK).sum()/MK.sum()))
    print(f"  {name:14s} restore err at {LEVELS}: {[round(e,4) for e in err]}",flush=True)
sheet.save(os.environ["OUT"]); print("  wrote",os.environ["OUT"],flush=True)
