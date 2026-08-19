"""What is the gap made of?  The obvious suspect is the cell size: the interior field carries
one colour per solid COARSE cell, so the cut face cannot hold detail finer than h_c.  Blur the
existing pipeline's own renders down to that and score them the same way."""
import os, sys, glob, numpy as np, cv2, torch
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
REF = "/workspace/rebuild/worktree/secref_orraw_hsep"
PIPE = "/workspace/rebuild/worktree/eval_orange"
TMP = "/workspace/ovoxel_native/diag"
import realism

dev = "cuda"
refs = realism._paths(REF)

def score(paths):
    return realism._dreamsim(refs, paths, dev)

print(f"references {len(refs)}")
src = sorted(glob.glob(PIPE + "/rh*_init_0.png"))
print("pipeline as rendered:", round(score(src), 4))
for n in (256, 197, 160, 128, 96, 64):
    d = f"{TMP}/down{n}"; os.makedirs(d, exist_ok=True)
    for p in src:
        a = cv2.imread(p)
        b = cv2.resize(cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA),
                       (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(d, os.path.basename(p)), b)
    print(f"  pipeline resampled through {n}x{n}: {score(sorted(glob.glob(d+'/rh*_init_0.png'))):.4f}")

# and the other direction: how much of our number is the surface (peel) vs the cut face
for tag, d in [("ovoxel-native init", "/workspace/ovoxel_native/run1/eval_init"),
               ("ovoxel-native 400it", "/workspace/ovoxel_native/run1/eval_final")]:
    ps = sorted(glob.glob(d + "/rh*_init_0.png"))
    if ps:
        print(f"  {tag}: {score(ps):.4f}")
