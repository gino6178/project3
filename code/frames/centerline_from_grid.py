# The declared centreline of a curved object, read off its own occupancy: at every height the centroid of
# the occupied cross-section, smoothed -- the low-cost structural prior section 1.5 asks the user for,
# here taken from the generated exterior rather than drawn by hand.
#   python centerline_from_grid.py <grid.pt> <out centerline.json>
import sys,json,numpy as np,torch
from scipy import ndimage as ndi
G=torch.load(sys.argv[1],map_location="cpu"); N=G["N"]; C=(N-1)/2; occ=G["OCC"][0].numpy()>0.5
vs,cx,cz=[],[],[]
for j in range(N):
    sl=occ[:,j,:]
    if sl.sum()<20: continue
    a,b=np.nonzero(sl); vs.append(j-C); cx.append(a.mean()-C); cz.append(b.mean()-C)
vs,cx,cz=map(np.array,(vs,cx,cz)); cx=ndi.uniform_filter1d(cx,9,mode="nearest"); cz=ndi.uniform_filter1d(cz,9,mode="nearest")
full=np.linspace(-C,C,N+1)                                   # the grid's whole height, whatever N
json.dump(dict(v=full.tolist(),cx=np.interp(full,vs,cx).tolist(),cz=np.interp(full,vs,cz).tolist()),open(sys.argv[2],"w"))
print("centreline over",len(vs),"heights; offset range u0",round(cx.min(),1),round(cx.max(),1),"u1",round(cz.min(),1),round(cz.max(),1))
