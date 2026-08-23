"""Does a pretrained VQGAN carry the high-frequency detail our held-out cuts lack?

The test, before any training: take a held-out cut face this representation renders, pass it
through the pretrained VQGAN's encoder and decoder, and ask two questions of the result.

  1. Does it gain high-frequency energy? Measured as the power above a quarter of Nyquist as a
     fraction of total power, against the render and against the photograph the cut is scored on.
  2. Does it move towards the photograph or away from it? Measured by the same DreamSim distance
     the paper reports.

A prior that contains the missing detail should raise (1) towards the photograph's level without
worsening (2). One that merely hallucinates texture will raise (1) and worsen (2).
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/workspace/taming-transformers")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import ovnative as ON, anchor, refsel
import nvdiffrast.torch as dr
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
dev = "cuda"
RES = 256                                    # f=16 wants a multiple of 16
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)

# the encoder, decoder and quantiser directly: taming's VQModel is a LightningModule and the
# training half of it is not needed to ask what its latent space contains
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

ddconfig = dict(double_z=False, z_channels=256, resolution=256, in_channels=3, out_ch=3, ch=128,
                ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)


class VQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # the checkpoint's codebook is 1024 entries, whatever the repository is named
        self.quantize = VectorQuantizer(1024, 256, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], 256, 1)
        self.post_quant_conv = torch.nn.Conv2d(256, ddconfig["z_channels"], 1)

    def forward(self, x):
        q, _, _ = self.quantize(self.quant_conv(self.encoder(x)))
        return self.decoder(self.post_quant_conv(q)), None


vq = VQ().to(dev).eval()
sd = torch.load("/workspace/vq/pytorch_model.bin", map_location="cpu")
sd = sd.get("state_dict", sd)
missing, unexpected = vq.load_state_dict(sd, strict=False)
print(f"VQGAN: {len(missing)} missing, {len(unexpected)} unexpected")
if missing:
    print("  missing:", missing[:6])


def hf_fraction(x, mask=None):
    """Power above a quarter of Nyquist, as a fraction of total power.

    Restricted to the object when a mask is given: the background is uniform in a photograph and
    not necessarily in a render, so comparing the two over the whole frame compares backgrounds.
    The foreground is isolated by mean-filling outside it, which removes the silhouette edge from
    the spectrum as well.
    """
    g = x.mean(-1) if x.ndim == 3 else x
    if mask is not None:
        g = np.where(mask, g, g[mask].mean() if mask.any() else 0.0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    n, m = F.shape
    yy, xx = np.mgrid[:n, :m]
    r = np.hypot(yy - n / 2, xx - m / 2) / (n / 2)
    return float(F[r > 0.25].sum() / max(F.sum(), 1e-9))



sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism

print(f"\n{'object':14s} {'family':6s} {'HF render':>10s} {'HF vqgan':>10s} {'HF photo':>10s} "
      f"{'DS render':>10s} {'DS vqgan':>10s}")
for OBJ in ("orange_sp", "watermelon_sp"):
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
    conf = open(f"{OBJDIR}/{OBJ}.conf").read()
    C = np.load(f"{W}/cams_{OBJ}_v2.npz")
    p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
    st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
    if "dec_i" in p:
        w = p["dec_i"]["stage1.0.weight"].shape[0]
        n = sum(1 for k in p["dec_i"] if k.startswith("stage1.") and k.endswith(".weight")) - 1
        anchor.W_HID, anchor.N_HID = w, n
        di = anchor.ColourDecoder(len(st["interior"]), init_rgb=st["interior"]).to(dev)
        di.load_state_dict(p["dec_i"])
        ds = anchor.ColourDecoder(len(st["surf_rgb"]), init_rgb=st["surf_rgb"]).to(dev)
        ds.load_state_dict(p["dec_s"])
        with torch.no_grad():
            st["interior"], st["surf_rgb"] = di(), ds()

    def spec(k):
        v = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith(k)]
        return v[0] if v else None

    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    for fam, key in (("h", "EVAL_REF="), ("v", "EVAL_REF_V=")):
        mvp = C["eh_mvp"] if fam == "h" else C["ev_mvp"]
        pl = C["eh_planes"] if fam == "h" else C["ev_planes"]
        sp = spec(key)
        if len(pl) == 0 or not sp:
            continue
        k = len(pl) // 2
        with torch.no_grad():
            img, _, _, _ = ON.render_section(st, glctx, T(mvp[k]), T(pl[k, :3]), float(pl[k, 3]),
                                             RES)
            x = (img.clamp(0, 1)[None] * 2 - 1)
            rec, _ = vq(x)
            rec = ((rec[0] + 1) / 2).clamp(0, 1)
        ref = refsel.as_array(refsel.photo(f"{FN}/{sp}", k, max(len(pl), 1)), RES)
        a = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        m_a = a.mean(-1) < 0.985
        b = rec.permute(1, 2, 0).cpu().numpy()
        c = np.asarray(ref, np.float32)
        m_c = c.mean(-1) < 0.985
        Image.fromarray((np.concatenate([a, b, c], 1) * 255).astype(np.uint8)).save(
            f"{W}/vq_{OBJ}_{fam}.jpg", quality=94)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pa, pb, pc = f"{td}/r.png", f"{td}/v.png", f"{td}/p.png"
            for arr, path in ((a, pa), (b, pb), (c, pc)):
                Image.fromarray((arr * 255).astype(np.uint8)).save(path)
            d_r = realism._dreamsim([pc], [pa], dev)
            d_v = realism._dreamsim([pc], [pb], dev)
        # scale-matched: an object drawn smaller puts the same structure at a higher image
        # frequency, so the two are resampled to equal foreground area before the spectra are
        # compared
        def crop_to(arr, m, side=192):
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sub = Image.fromarray((arr[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(
                (side, side), Image.BILINEAR)
            sm = Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(
                (side, side), Image.NEAREST)
            return np.asarray(sub, np.float32) / 255, np.asarray(sm) > 127

        ca, ma = crop_to(a, m_a)
        cb, _ = crop_to(b, m_a)
        cc, mc = crop_to(c, m_c)
        print(f"{OBJ:14s} {fam:6s} {hf_fraction(ca, ma):10.5f} {hf_fraction(cb, ma):10.5f} "
              f"{hf_fraction(cc, mc):10.5f} {d_r:10.4f} {d_v:10.4f}   "
              f"raw fg {100*m_a.mean():.0f}%/{100*m_c.mean():.0f}%")
