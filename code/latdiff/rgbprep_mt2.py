"""The fitted interior as an RGB volume, for a 3-D SinDiffusion that works in colour space.

The latent version generated colours the decoder had never seen -- an 8-D latent sampled near the
training distribution still decodes to blue-violet where no training latent lived. Training the
diffusion on the decoded RGB directly removes the decoder from the loop: what the model samples is
the colour, so there is no out-of-distribution decode to go wrong.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
import anchor
W = "/workspace/ovoxel_native"; OBJ = "orange_sp"; dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
p = torch.load(f"{W}/s_v2_mt2_{OBJ}/params.pt", map_location=dev)
w = p["dec_i"]["stage1.0.weight"].shape[0]
nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
anchor.W_HID, anchor.N_HID = w, nl
di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
di.load_state_dict(p["dec_i"])
with torch.no_grad():
    rgb = di().clamp(0, 1)                              # (N, 3), the fitted interior colour

solid = st["solid"].long()
hc = float(st["hc"])
dims = [int(solid[:, i].max()) + 1 for i in range(3)]
vol = torch.zeros(*dims, 3, device=dev)
occ = torch.zeros(*dims, dtype=torch.bool, device=dev)
vol[solid[:, 0], solid[:, 1], solid[:, 2]] = rgb
occ[solid[:, 0], solid[:, 1], solid[:, 2]] = True
m, s = rgb.mean(0), rgb.std(0)
print(f"{OBJ}: RGB volume {tuple(dims)}, {len(solid):,} cells; mean {[round(float(x),3) for x in m]} "
      f"std {[round(float(x),3) for x in s]}")
np.savez(f"{W}/lat_rgb_mt2_{OBJ}.npz", vol=vol.cpu().numpy().astype(np.float32),
         occ=occ.cpu().numpy(), solid=solid.cpu().numpy().astype(np.int32),
         mean=m.cpu().numpy(), std=s.cpu().numpy(), hc=hc,
         org=np.asarray(st["org"], dtype=np.float32))
print(f"lat_rgb_mt2_{OBJ}.npz written")
