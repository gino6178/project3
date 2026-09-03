"""Measure the effective receptive field of the current SinDiffusion UNet, and draw it on the orange.

The receptive field is measured, not derived: a gradient of the output's centre pixel with respect
to the input is nonzero exactly over the input region that can influence that pixel. Its extent is
the patch the model actually sees. Overlaid on the 256px orange, it shows how much of the fruit one
patch covers -- the columella spans the whole height, so if the patch is much smaller than the
orange the model can never see the columella as a single centred structure.
"""
import os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/sindiff")
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
dev = "cuda"; S = 256
d = model_and_diffusion_defaults()
d.update(image_size=256, num_channels=64, num_head_channels=16, channel_mult="1,2,4",
         attention_resolutions="2", num_res_blocks=1, resblock_updown=False, use_fp16=False,
         use_scale_shift_norm=True, use_checkpoint=False, diffusion_steps=1000,
         noise_schedule="linear", learn_sigma=False, class_cond=False)
model, _ = create_model_and_diffusion(**d)
model.load_state_dict(torch.load("/workspace/sindiff/OUTPUT/sd-long00/model008000.pt", map_location="cpu"))
model.cuda().eval()

x = torch.randn(1, 3, S, S, device=dev, requires_grad=True)
t = torch.full((1,), 500, device=dev, dtype=torch.long)
out = model(x, t)
c = S // 2
# sum the centre pixel over output channels, backprop
model.zero_grad()
out[0, :, c, c].sum().backward()
g = x.grad.detach().abs().sum(1)[0].cpu().numpy()      # (S,S) sensitivity map
g = g / g.max()
# extent where sensitivity > 1% of peak
ys, xs = np.where(g > 0.01)
h_ext = ys.max() - ys.min() + 1; w_ext = xs.max() - xs.min() + 1
# effective (90% energy) box
flat = np.sort(g.ravel())[::-1]
csum = np.cumsum(flat); thr = flat[np.searchsorted(csum, 0.9 * csum[-1])]
ys2, xs2 = np.where(g >= thr)
h90 = ys2.max() - ys2.min() + 1; w90 = xs2.max() - xs2.min() + 1
print(f"receptive field (>1% of peak): {w_ext} x {h_ext} px of {S}")
print(f"effective (90% energy):        {w90} x {h90} px of {S}")

# schematic: orange photo with the RF box centred
orange = Image.open("/workspace/sindiff/data/long00.png").convert("RGB").resize((S, S))
im = orange.copy(); dr = ImageDraw.Draw(im)
def box(w, h, col, wd):
    dr.rectangle([c - w//2, c - h//2, c + w//2, c + h//2], outline=col, width=wd)
box(w_ext, h_ext, (255, 0, 0), 3)      # full RF, red
box(w90, h90, (0, 90, 255), 3)         # effective RF, blue
im.save("/workspace/rf.png")
# also save the sensitivity heatmap
Image.fromarray((np.stack([g, g*0, 1-g], -1) * 255).astype(np.uint8)).resize((S,S)).save("/workspace/rf_heat.png")
print("rf.png: red=full RF, blue=90% RF, on the orange")

# clearer: log-scale sensitivity heat overlaid on the orange
import numpy as _np
gl = _np.log10(g + 1e-6); gl = (gl - gl.min()) / (gl.max() - gl.min())
heat = _np.stack([gl, gl*0.3, 1-gl], -1)
base = _np.asarray(orange).astype(float) / 255
blend = (0.45*base + 0.55*heat)
Image.fromarray((blend*255).astype(_np.uint8)).save("/workspace/rf_log.png")
# horizontal profile through the centre row, and vertical through centre col
prof_h = g[c]; prof_v = g[:, c]
print("frac of input pixels with >1% peak sensitivity: %.1f%%" % (100*(g>0.01).mean()))
print("frac with >0.1%% peak: %.1f%%" % (100*(g>0.001).mean()))
