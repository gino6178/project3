"""Filter bad propagated targets and downsample density, then keep supervised + held-out anchors.

Sequential propagation accumulates errors: some planes go over-saturated or washed. Those bad
targets pull the shared field the wrong way. Score each propagated target by how far its colour
statistics sit from the good ones (median saturation/value) and drop the outliers; also thin the
dense set so slabs no longer heavily overlap and conflict. Supervised planes (render) and the
held-out targets keep their place.
"""
import os, torch, numpy as np
W="/workspace/ovoxel_native"; OBJ="orange_sp"
KEEP_FRAC=float(os.environ.get("KEEP_FRAC","0.6"))    # drop worst 40%
STRIDE=int(os.environ.get("STRIDE","2"))              # keep every STRIDE-th propagated (density down)
T=torch.load(f"{W}/targets_seq_{OBJ}.pt",map_location="cpu")
prop=[t for t in T if not t.get("sup")]
sup=[t for t in T if t.get("sup")]
def stats(t):
    x=(t["tgt"][0].float()*0.5+0.5).clamp(0,1)   # 3,H,W
    m=t["mask"][0,0].bool()
    if m.sum()<50: return None
    rgb=x[:,m]                                    # 3,N
    mx,_=rgb.max(0); mn,_=rgb.min(0)
    sat=((mx-mn)/mx.clamp_min(1e-3)).mean().item()
    val=mx.mean().item()
    # orange-ness: red>green>blue expected; penalise deviation
    r,g,b=rgb[0].mean().item(),rgb[1].mean().item(),rgb[2].mean().item()
    return sat,val,r,g,b
S=[stats(t) for t in prop]
ok=[i for i,s in enumerate(S) if s is not None]
sat=np.array([S[i][0] for i in ok]); val=np.array([S[i][1] for i in ok])
# median-based good range
smed,sm=np.median(sat),np.std(sat); vmed,vm=np.median(val),np.std(val)
score=np.abs(sat-smed)/(sm+1e-6)+np.abs(val-vmed)/(vm+1e-6)   # distance from typical
order=np.argsort(score)               # smallest deviation = best
nkeep=int(len(ok)*KEEP_FRAC)
keep_idx=set(ok[order[j]] for j in range(nkeep))
# also thin by stride within each family (sort by depth)
kept=[]
for fam in ("long","trans"):
    idxs=[i for i in range(len(prop)) if prop[i]["name"]==fam and i in keep_idx]
    idxs.sort(key=lambda i:prop[i]["d"])
    kept += idxs[::STRIDE]
newT=[prop[i] for i in kept]+sup
torch.save(newT,f"{W}/targets_filt_{OBJ}.pt")
print(f"prop {len(prop)} -> kept {len(kept)} (frac {KEEP_FRAC}, stride {STRIDE}); +{len(sup)} sup = {len(newT)} total")
