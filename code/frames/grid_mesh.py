# A smooth exterior for an object that has no generated mesh (the photographed fruit): the carrier's own
# occupancy, blurred by a voxel and contoured (marching cubes), Taubin-smoothed, and coloured from the
# carrier's skin a voxel inside the surface -- the captured exterior, nothing invented.  The mesh is written
# in the grid's voxel frame (centred), which render_asset.py reads as it is when given an .npz.
#   python grid_mesh.py <grid.pt> <out.npz>
import sys,numpy as np,torch,torch.nn.functional as F,trimesh
from scipy import ndimage as ndi
from skimage import measure
G=torch.load(sys.argv[1],map_location="cpu"); N=G["N"]; C=(N-1)/2
occ=ndi.gaussian_filter(G["OCC"][0].float().numpy(),1.0)
v,f,_,_=measure.marching_cubes(np.pad(occ,1),0.5); v-=1
m=trimesh.Trimesh(v,f[:,::-1],process=True); trimesh.smoothing.filter_taubin(m,iterations=20)
m=m.split(only_watertight=False); m=max(m,key=lambda x:len(x.faces))                 # the object, not stray specks
v=np.asarray(m.vertices)-C; n=np.asarray(m.vertex_normals)
V=G["V"].float()[None]
def samp(P):
    q=torch.tensor(P/C,dtype=torch.float32).flip(-1).reshape(1,1,1,-1,3)
    return ((F.grid_sample(V,q,mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,0].T+1)/2).clamp(0,1).numpy()
col=samp(v-1.2*n)                                                                         # the skin, just inside the surface
np.savez(sys.argv[2],v=v.astype(np.float32),f=np.asarray(m.faces,dtype=np.int32),c=col.astype(np.float32))
print(sys.argv[2],len(v),"vertices",len(m.faces),"faces")
