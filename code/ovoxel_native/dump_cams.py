"""Run the pipeline's own held-out-cut sequence, but instead of drawing anything, record the
camera and the plane. Same file, same seed, same HELDOUT_BAND -> the same twelve cuts."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/workspace/rebuild/project3/code/src")
import random_cuts as rc

OUT = "/workspace/ovoxel_native/cams_orange.npz"
rec = {}
n = [0]

def hook(cam, plane, mask, mask_suf, size, tpos, pos):
    i = n[0]; n[0] += 1
    if i == 0:
        print("cam attrs:", [a for a in dir(cam) if not a.startswith("_")])
    T = tpos.detach().cpu().double().numpy()
    P = pos.detach().cpu().double().numpy()
    # tpos -> pos is a similarity; solve the 4x3 affine exactly, on a subsample
    k = np.random.RandomState(0).choice(len(T), min(20000, len(T)), replace=False)
    A = np.concatenate([T[k], np.ones((len(k), 1))], 1)
    M, *_ = np.linalg.lstsq(A, P[k], rcond=None)          # (4,3)
    res = np.abs(A @ M - P[k]).max()
    rec[f"c{i}_wv"] = cam.world_view_transform.detach().cpu().numpy()
    rec[f"c{i}_fp"] = cam.full_proj_transform.detach().cpu().numpy()
    rec[f"c{i}_center"] = cam.camera_center.detach().cpu().numpy()
    rec[f"c{i}_fov"] = np.array([cam.FoVx, cam.FoVy], np.float64)
    rec[f"c{i}_plane_t"] = np.asarray(plane, np.float64)
    rec[f"c{i}_affine"] = M
    rec[f"c{i}_affine_res"] = np.array([res])
    rec[f"c{i}_size"] = np.array([size])
    print(f"  cut {i}: plane(tpos) {np.round(plane,5)}  affine residual {res:.3e}  "
          f"fov {cam.FoVx:.4f},{cam.FoVy:.4f}  kept {int(mask.sum())}/{len(mask)}")
    return np.ones((size, size, 3), np.float32)

rc.RENDER_HOOK[0] = hook
os.environ["HELDOUT_BAND"] = "0.30,0.70"
os.environ["FULL_SH"] = "1"
rc.main("orange/orange_demo_epoch_199.ply", "config/orange_physics.json",
        "config/sphere_demo", "/tmp/dump_cuts", 12, 512)
np.savez(OUT, **rec)
print("wrote", OUT, len(rec))
