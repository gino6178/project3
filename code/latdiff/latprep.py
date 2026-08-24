"""Stage 1's output as a dense volume: the per-cell latent the fit produced.

The lattice already is a latent space -- one 8-D vector per coarse cell, read by a shared decoder.
Nothing here is trained; this only lays those vectors out on the integer grid they belong to, with
the occupancy beside them, so a 3-D model can be run over the volume rather than over a list.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
import anchor

W = "/workspace/ovoxel_native"
OBJ = os.environ.get("OBJ", "orange_sp")
dev = "cuda"

st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
solid = st["solid"].long()
hc = float(st["hc"])

assert "dec_i" in p, "this run has no interior decoder; the latent is the decoded colour only"
w = p["dec_i"]["stage1.0.weight"].shape[0]
nl = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
anchor.W_HID, anchor.N_HID = w, nl
di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
di.load_state_dict(p["dec_i"])
feat = di.feat.detach()                      # (N, 8)
with torch.no_grad():
    rgb = di().clamp(0, 1)                   # what the decoder makes of it

dims = [int(solid[:, i].max()) + 1 for i in range(3)]
vol = torch.zeros(*dims, feat.shape[1], device=dev)
occ = torch.zeros(*dims, dtype=torch.bool, device=dev)
vol[solid[:, 0], solid[:, 1], solid[:, 2]] = feat
occ[solid[:, 0], solid[:, 1], solid[:, 2]] = True

m, s = feat.mean(0), feat.std(0)
print(f"{OBJ}: lattice {tuple(dims)}, {len(solid):,} solid cells "
      f"({occ.float().mean()*100:.1f}% of the box), latent {feat.shape[1]}-D")
print(f"  per-channel mean {[round(float(x), 3) for x in m]}")
print(f"  per-channel std  {[round(float(x), 3) for x in s]}")
print(f"  decoded rgb: mean {[round(float(x), 3) for x in rgb.mean(0)]}, "
      f"range [{float(rgb.min()):.3f}, {float(rgb.max()):.3f}]")

np.savez(f"{W}/lat_{OBJ}.npz", vol=vol.cpu().numpy().astype(np.float32),
         occ=occ.cpu().numpy(), solid=solid.cpu().numpy().astype(np.int32),
         mean=m.cpu().numpy(), std=s.cpu().numpy(), hc=hc,
         org=np.asarray(st["org"], dtype=np.float32))
print(f"lat_{OBJ}.npz written")
