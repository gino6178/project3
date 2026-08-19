import glob, os, sys
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism, cv2, numpy as np
W="/workspace/ovoxel_native"; FN="/workspace/rebuild/worktree"; EXT=W+"/out/ext"
NAMES=["up","down","front","right","back","left"]
refs=[f"{FN}/cube_or6_prep/{n}_ref.png" for n in NAMES]
SHEET={"up":(0,0),"front":(0,1),"right":(0,2),"down":(1,0),"back":(1,1),"left":(1,2)}
def split(p,tag):
    a=cv2.imread(p); d=f"{EXT}/{tag}"; os.makedirs(d,exist_ok=True)
    for n,(r,c) in SHEET.items(): cv2.imwrite(f"{d}/{n}.png", a[r*512:(r+1)*512, c*512:(c+1)*512])
    return d
split(f"{EXT}/pipe_r1_sheet.png","pipe_r1"); split(f"{EXT}/pipe_r2_sheet.png","pipe_r2")
for tag in ["seed_r1","r1_pin","r1_pin_full","r1_free","seed_r2","r2_pin_full","r2_free","pipe_r1","pipe_r2"]:
    ps=[f"{EXT}/{tag}/{n}.png" for n in NAMES]
    if not all(os.path.exists(x) for x in ps): print("RESULT",tag,"missing"); continue
    print("RESULT %-14s %.4f" % (tag, realism._dreamsim(refs, ps, "cuda")))