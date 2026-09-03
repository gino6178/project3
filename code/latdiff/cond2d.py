"""Conditional SinDiffusion: output depends on the input render, so different planes differ.

The unconditional model produces the same generic orange for any input, which forces every plane to
the same thing under joint lift. This conditions the denoiser on a render-like image: the target is
a real photo patch, the condition is that patch degraded (blurred + desaturated) to look like an
O-Voxel render. The model learns to restore realism WHILE respecting the structure in the
condition, so feeding two different plane renders gives two different, structure-appropriate outputs.
Trained from the single photo via random crops/flips, in the SinDiffusion spirit.
"""
import os, sys, time, argparse
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.sinddpm import UNetModel
from guided_diffusion.gaussian_diffusion import (GaussianDiffusion, ModelMeanType,
                                                 ModelVarType, LossType, get_named_beta_schedule)
dev="cuda"; S=256
ap=argparse.ArgumentParser()
ap.add_argument("--photo", default="/workspace/rebuild/worktree/spl_orange_v/or_long_00.png")
ap.add_argument("--tag", default="cond_long")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--crop", type=int, default=192)   # random crop then resize to S
a=ap.parse_args()

photo=torch.from_numpy(np.asarray(Image.open(a.photo).convert("RGB").resize((S,S))).astype(np.float32)/255).permute(2,0,1).to(dev)

def gauss_blur(x, k=9, sig=4.0):
    ax=torch.arange(k,device=dev)-k//2; g=torch.exp(-(ax**2)/(2*sig**2)); g=(g/g.sum())
    ker=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
    return F.conv2d(F.pad(x,(k//2,)*4,mode="reflect"),ker,groups=3)

def degrade(x):
    """Mimic the O-Voxel render: blur + desaturate + slight value shift."""
    b=gauss_blur(x)
    gray=b.mean(1,keepdim=True)
    return (0.5*b+0.5*gray).clamp(0,1)

def sample_pair(bs):
    tg=[]; cd=[]
    for _ in range(bs):
        # random crop + flip of the photo -> target; degrade -> condition
        c=a.crop; y0=np.random.randint(0,S-c+1); x0=np.random.randint(0,S-c+1)
        patch=photo[:,y0:y0+c,x0:x0+c][None]
        if np.random.rand()<0.5: patch=torch.flip(patch,[3])
        patch=F.interpolate(patch,S,mode="bilinear",align_corners=False)[0]
        tg.append(patch); cd.append(degrade(patch[None])[0])
    return torch.stack(tg)*2-1, torch.stack(cd)*2-1

model=UNetModel(image_size=S,in_channels=6,model_channels=64,out_channels=3,num_res_blocks=1,
                attention_resolutions=(S//2,),channel_mult=(1,2,4),num_head_channels=16,
                use_scale_shift_norm=True,use_checkpoint=True,use_fp16=False).cuda()
diff=GaussianDiffusion(betas=get_named_beta_schedule("linear",1000),
                       model_mean_type=ModelMeanType.EPSILON,model_var_type=ModelVarType.FIXED_LARGE,
                       loss_type=LossType.MSE)
opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
print(f"conditional UNet in=6 out=3, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params",flush=True)

def call(x_t, t, cond):
    return model(torch.cat([x_t,cond],1), t)

t0=time.time()
for step in range(1,a.steps+1):
    tg,cd=sample_pair(a.batch)
    t=torch.randint(0,diff.num_timesteps,(a.batch,),device=dev)
    loss=diff.training_losses(lambda z,tt: call(z,tt,cd), tg, t)["loss"].mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step%500==0 or step==1:
        print(f"  step {step:5d}  loss {float(loss):.4f}  {time.time()-t0:.0f}s",flush=True)
    if step%2000==0 or step==a.steps:
        torch.save({"model":model.state_dict(),"step":step}, f"/workspace/{a.tag}.pt")
print(f"saved /workspace/{a.tag}.pt")
