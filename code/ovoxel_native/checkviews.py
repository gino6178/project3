"""One picture per training-view family, beside the reference it is matched to, so that a plane
in the wrong frame or a camera on the wrong side is visible rather than inferred."""
import os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/ovoxel_native"); sys.path.insert(0, "/workspace/ovoxel_native/vendor")
import ovnative as ON, nvdiffrast.torch as dr, section_match as sm, refsel

W = "/workspace/ovoxel_native"; FN = "/workspace/rebuild/worktree"
REF_H, REF_V = f"{FN}/secref_orraw_hsep", f"{FN}/secref_orraw_vsep"
EXT = f"{FN}/cube_or6_prep"
OUT = W + "/out/views"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"; RES = 512
st = torch.load(W + "/state_orange.pt", map_location=dev, weights_only=False)
ON.FDG = ON._load_ovoxel(); glctx = dr.RasterizeCudaContext(device=dev)
C = np.load(W + "/cams_mv.npz")
H_LO, H_HI = int(C["h_lo"][0]), int(C["h_hi"][0]); NH = H_HI - H_LO
NV = len(C["v_planes"])


def save(tag, img, al, ref):
    a = img.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    t = sm.section_target(img, ref, alpha=al)
    b = t.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    r = cv2.resize(ref, (RES, RES))
    sheet = np.concatenate([a, b, r], 1)
    cv2.imwrite(f"{OUT}/{tag}.png", (sheet[:, :, ::-1] * 255).astype(np.uint8))
    print(f"  {tag:<16} coverage {float(al.mean()):.3f}  L1 {float((img-t).abs().mean()):.4f}  "
          f"render mean {a.reshape(-1,3).mean(0).round(3)}  target mean {b.reshape(-1,3).mean(0).round(3)}")


hn = torch.as_tensor(C["h_planes"][0, :3], dtype=torch.float32, device=dev)
hmvp = torch.as_tensor(C["h_mvp"], dtype=torch.float32, device=dev)
for j in (0, NH // 2, NH - 1):
    d = float(C["h_planes"][H_LO + j, 3])
    img, al, _, _ = ON.render_section(st, glctx, hmvp, hn, d, RES)
    save(f"train_h{j:02d}", img, al, refsel.as_array(refsel.solved_photo(REF_H, j, NH), RES))
for i in (0, NV // 2, NV - 1):
    n = torch.as_tensor(C["v_planes"][i, :3], dtype=torch.float32, device=dev)
    mvp = torch.as_tensor(C["v_mvp"][i], dtype=torch.float32, device=dev)
    img, al, _, _ = ON.render_section(st, glctx, mvp, n, float(C["v_planes"][i, 3]), RES)
    save(f"train_v{i:02d}", img, al, refsel.as_array(refsel.photo(REF_V, i, NV), RES))
for i, nm in enumerate([str(x) for x in C["e_names"]]):
    mvp = torch.as_tensor(C["e_mvp"][i], dtype=torch.float32, device=dev)
    img, al, _, _ = ON.render_exterior(st, glctx, mvp, RES)
    ref = cv2.imread(f"{EXT}/{nm}_ref.png")[:, :, ::-1].astype(np.float32) / 255.
    save(f"train_e_{nm}", img, al, ref)
print("CHECK_OK")
