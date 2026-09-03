"""Dense targets along each family's sweep normal, so every swept cell has a nearby target.

The 6 held-out planes span 5-28 cells but the ovcut sweep covers the whole object (91/129 cells),
so cells between the sparse target planes were never updated and the updated ones showed as bright
bands. This samples offsets densely along the exact normal ovcut sweeps -- match+SDEdit at every
offset -- plus the supervised planes frozen to their render, so coverage is continuous and the
photo planes are held.
"""
import os, sys, time
os.environ["CUT_DEFERRED"]="1"
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,"/workspace/ovoxel_native"); sys.path.insert(0,"/workspace/sindiff")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import ovnative as ON, anchor
import nvdiffrast.torch as dr
W="/workspace/ovoxel_native"; OBJ="orange_sp"; dev="cuda"; S=256
STR=float(os.environ.get("STR","0.3")); SPACING=float(os.environ.get("SPACING","2.5"))
ON.FDG=ON._load_ovoxel(); glctx=dr.RasterizeCudaContext(device=dev)
st=torch.load(f"{W}/state_{OBJ}.pt",map_location=dev,weights_only=False)
C=np.load(f"{W}/cams_{OBJ}_v2.npz"); p=torch.load(f"{W}/s_v2_{OBJ}/params.pt",map_location=dev)
st["dual_v"]=p["dual_v"].to(dev); st["split_w"]=p["split_w"].to(dev)
w=p["dec_i"]["stage1.0.weight"].shape[0]; nl=sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight"))-1
anchor.W_HID,anchor.N_HID=w,nl
di=anchor.ColourDecoder(len(st["interior"]),init_rgb=st["interior"]).to(dev); di.load_state_dict(p["dec_i"])
dsr=anchor.ColourDecoder(len(st["surf_rgb"]),init_rgb=st["surf_rgb"]).to(dev); dsr.load_state_dict(p["dec_s"])
with torch.no_grad(): st["interior"],st["surf_rgb"]=di(),dsr()
hc=float(st["hc"]); org=torch.as_tensor(st["org"],dtype=torch.float32,device=dev)
solid=st["solid"].long(); cen=(solid.float()+0.5)*hc+org
def load(ckpt):
    d=model_and_diffusion_defaults()
    d.update(image_size=256,num_channels=64,num_head_channels=16,channel_mult="1,2,4",
             attention_resolutions="2",num_res_blocks=1,resblock_updown=False,use_fp16=True,
             use_scale_shift_norm=True,use_checkpoint=True,diffusion_steps=1000,
             noise_schedule="linear",learn_sigma=False,class_cond=False)
    m,D=create_model_and_diffusion(**d); m.load_state_dict(torch.load(ckpt,map_location="cpu"))
    m.cuda().eval(); m.convert_to_fp16(); return m,D
mL,diff=load("/workspace/sindiff/OUTPUT/sd-long00/model012000.pt")
mT,_=load("/workspace/sindiff/OUTPUT/sd-orange_h/model006000.pt")
def loadphoto(pth):
    a=np.asarray(Image.open(pth).convert("RGB").resize((S,S))).astype(np.float32)/255
    return torch.from_numpy(a).permute(2,0,1).to(dev)
photoL=loadphoto("/workspace/rebuild/worktree/spl_orange_v/or_long_00.png")
photoT=loadphoto("/workspace/rebuild/worktree/spl_orange_h/or_trans_00.png")
psilL=(photoL.min(0).values<0.92); psilT=(photoT.min(0).values<0.92)
def bbox(m):
    ys,xs=torch.where(m); return xs.min().item(),ys.min().item(),xs.max().item(),ys.max().item()
def warp(photo,psil,rmask):
    px0,py0,px1,py1=bbox(psil); rx0,ry0,rx1,ry1=bbox(rmask[0,0]>0.5)
    sx=(rx1-rx0)/max(px1-px0,1); sy=(ry1-ry0)/max(py1-py0,1)
    ys,xs=torch.meshgrid(torch.arange(S,device=dev),torch.arange(S,device=dev),indexing="ij")
    src_x=(xs-(rx0+rx1)/2)/sx+(px0+px1)/2; src_y=(ys-(ry0+ry1)/2)/sy+(py0+py1)/2
    grid=torch.stack([src_x/(S-1)*2-1,src_y/(S-1)*2-1],-1)[None]
    return F.grid_sample(photo[None],grid,align_corners=True,padding_mode="border")[0]
model_cur=None
@torch.no_grad()
def sdedit(x0,mask,strength):
    ts=int(strength*diff.num_timesteps)-1
    x=diff.q_sample(x0,torch.full((1,),ts,device=dev,dtype=torch.long))
    for i in reversed(range(ts+1)):
        t=torch.full((1,),i,device=dev,dtype=torch.long)
        x=mask*x+(1-mask)*diff.q_sample(x0,t)
        x=diff.p_sample(model_cur,x,t,clip_denoised=True,model_kwargs={})["sample"]
    return mask*x+(1-mask)*x0
white=torch.ones(1,3,S,S,device=dev)

def rgb2ycbcr(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    y=0.299*r+0.587*g+0.114*b; cb=-0.168736*r-0.331264*g+0.5*b+0.5; cr=0.5*r-0.418688*g-0.081312*b+0.5
    return torch.stack([y,cb,cr],1)
def ycbcr2rgb(x):
    y,cb,cr=x[:,0],x[:,1]-0.5,x[:,2]-0.5
    return torch.stack([y+1.402*cr, y-0.344136*cb-0.714136*cr, y+1.772*cb],1).clamp(0,1)
def lc_swap(gen01,render01):
    g=rgb2ycbcr(gen01); r=rgb2ycbcr(render01)
    return ycbcr2rgb(torch.stack([g[:,0],r[:,1],r[:,2]],1))

k=len(C["v_mvp"])//2
fams=[("long",  torch.as_tensor(C["v_mvp"][k],dtype=torch.float32,device=dev).contiguous(),
       C["v_planes"][k,:3], mL, photoL, psilL),
      ("trans", torch.as_tensor(C["h_mvp"],dtype=torch.float32,device=dev).contiguous(),
       C["h_planes"][0,:3], mT, photoT, psilT)]
targets=[]; t0=time.time()
for name,mvp,nvec,model,photo,psil in fams:
    n=torch.as_tensor((nvec/np.linalg.norm(nvec)).astype(np.float32),device=dev)
    pr=(cen@n); d0=float(-pr.max())+3*hc; d1=float(-pr.min())-3*hc     # object extent, margin like ovcut
    ND=int(abs(d1-d0)/(SPACING*hc))
    model_cur=model
    for dd in np.linspace(d0,d1,ND):
        with torch.no_grad():
            img,af,_,_=ON.render_section(st,glctx,mvp,n,float(dd),S,exterior=True)
        mask=(af[:1]>0).float()[None]
        if float(mask.sum())<100: continue
        warped=warp(photo,psil,mask)
        matched=mask*warped[None]+(1-mask)*white
        out=sdedit(matched*2-1,mask,STR)
        gen01=(out*0.5+0.5).clamp(0,1); render01=img[None].clamp(0,1)
        out=(mask*lc_swap(gen01,render01)+(1-mask)*white)*2-1
        targets.append(dict(name=name,n=n.cpu(),d=float(dd),mvp=mvp.cpu(),tgt=out.cpu().half(),mask=mask.cpu().bool()))
    print(f"  {name}: {ND} dense offsets, {time.time()-t0:.0f}s",flush=True)
# supervised planes frozen to render
HL,HH=int(C["h_lo"][0]),int(C["h_hi"][0])
supg=[("long_sup",C["v_planes"],C["v_mvp"]),("trans_sup",C["h_planes"][HL:HH],np.broadcast_to(C["h_mvp"][None],(HH-HL,4,4)))]
for name,P,M in supg:
    for j in range(len(P)):
        nj=torch.as_tensor((P[j,:3]/np.linalg.norm(P[j,:3])).astype(np.float32),device=dev)
        mvpj=torch.as_tensor(M[j].copy(),dtype=torch.float32,device=dev).contiguous()
        with torch.no_grad():
            img,_,_,_=ON.render_section(st,glctx,mvpj,nj,float(P[j,3]),S,exterior=True)
            _,af,_,_=ON.render_section(st,glctx,mvpj,nj,float(P[j,3]),S,exterior=True)
        mask=(af[:1]>0).float()[None]
        targets.append(dict(name=name,n=nj.cpu(),d=float(P[j,3]),mvp=mvpj.cpu(),tgt=(img[None]*2-1).cpu().half(),mask=mask.cpu().bool(),sup=True))
torch.save(targets,f"{W}/targets_denselc_{OBJ}.pt")
nd=sum(1 for t in targets if not t.get("sup")); ns=len(targets)-nd
print(f"saved {len(targets)} targets: {nd} dense match+SDEdit, {ns} supervised frozen, {time.time()-t0:.0f}s")
