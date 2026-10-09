# A curved object's grid turned so its bend lies in the theta=0 lengthwise plane (u0, v) -- the plane its
# lengthwise photographs are framed against -- and its centreline written for the curvilinear lattice.
#   python orient_bend.py <grid.pt> <out grid.pt> <out centerline.json>
import sys,subprocess,json,torch,os
G=torch.load(sys.argv[1],map_location="cpu"); N=G["N"]; C=(N-1)/2; occ=G["OCC"][0]>0.5
cu0,cu1=[],[]
for j in range(N):
    sl=occ[:,j,:]
    if sl.sum()<20: continue
    a,b=sl.nonzero(as_tuple=True); cu0.append(float(a.float().mean())); cu1.append(float(b.float().mean()))
if (max(cu1)-min(cu1))>(max(cu0)-min(cu0)):          # the bend is along u1: swap u0 and u1
    for k in ("V","OCC","SHELL","CORE"): G[k]=G[k].permute(0,3,2,1).contiguous()
    print("bend turned into the theta=0 plane")
torch.save(G,sys.argv[2])
subprocess.run([sys.executable,os.path.join(os.path.dirname(os.path.abspath(__file__)),"centerline_from_grid.py"),sys.argv[2],sys.argv[3]],check=True)
