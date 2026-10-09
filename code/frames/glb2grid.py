# A generated exterior (TRELLIS.2 GLB) -> the dense grid the lift works on (planes.Vol format).
# The mesh states inside-or-outside and where the surface is, as mesh_to_voxel.py argues; its
# texture is the shell's colour.  The polar axis is the mesh's up (+Y, glTF convention) = grid dim 1.
# Size: the largest of radius and half-height is SIZE voxels, as prep_ai.py's shapes.
#   python glb2grid.py <mesh.glb> <out grid.pt> [N=128]
import sys,numpy as np,torch,trimesh
from scipy import ndimage as ndi
src,out=sys.argv[1],sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 128; C=(N-1)/2; SIZE=50.0*N/128
sc=trimesh.load(src,force="scene"); mesh=trimesh.util.concatenate([g for g in sc.dump()])
v=mesh.vertices.copy(); lo,hi=v.min(0),v.max(0); ctr=(lo+hi)/2
rad=np.sqrt((v[:,0]-ctr[0])**2+(v[:,2]-ctr[2])**2).max(); hh=(hi[1]-lo[1])/2
mesh.apply_translation(-ctr)
import os
if os.environ.get("ASPECT"):          # match the height/diameter of the cut photographs (a generated exterior need not)
    a=float(os.environ["ASPECT"]); k=a*(2*rad)/(2*hh); mesh.vertices[:,1]*=k; hh*=k
s=SIZE/max(rad,hh); mesh.apply_scale(s)
# occupancy: the surface voxelised at one cell, then filled -- exact enough at this pitch and fast
lin=np.arange(N)-C; A,B,Cc=np.meshgrid(lin,lin,lin,indexing="ij"); P=np.stack([A,B,Cc],-1).reshape(-1,3)
# A generated mesh need not be watertight (an open end, a seam), and a 3-D hole fill then leaves the
# solid hollow.  So: the surface alone, dilated one cell to close small gaps, filled slice by slice
# across the axis (an open end is a hole in no transverse slice) and in 3-D, then eroded back.
vx=mesh.voxelized(pitch=0.5)
ij=np.round(vx.points+C).astype(int); ij=ij[((ij>=0)&(ij<N)).all(1)]
surf=np.zeros((N,N,N),bool); surf[ij[:,0],ij[:,1],ij[:,2]]=True
surf=ndi.binary_dilation(surf,iterations=1)
occ=np.stack([ndi.binary_fill_holes(surf[:,j,:]) for j in range(N)],1)|ndi.binary_fill_holes(surf)
occ=ndi.binary_erosion(occ,iterations=1)|(ndi.binary_erosion(surf,iterations=0)&False)
occ=ndi.binary_fill_holes(occ)
dist=ndi.distance_transform_edt(occ)                                  # voxels to the outside
shell=occ&(dist<=2); core=occ&(dist>2)
# shell colour: the texture at the nearest of a dense coloured sampling of the surface
from scipy.spatial import cKDTree
sp,fi=trimesh.sample.sample_surface(mesh,2000000)
vis=mesh.visual
if hasattr(vis,"uv") and vis.uv is not None and getattr(vis.material,"baseColorTexture",None) is not None:
    bc=trimesh.triangles.points_to_barycentric(mesh.triangles[fi],sp)
    uv=(vis.uv[mesh.faces[fi]]*bc[:,:,None]).sum(1)
    tex=np.asarray(vis.material.baseColorTexture.convert("RGB")).astype(np.float32)/255; H,W,_=tex.shape
    px=np.clip((uv[:,0]%1)*(W-1),0,W-1).astype(int); py=np.clip((1-uv[:,1]%1)*(H-1),0,H-1).astype(int)
    scol=tex[py,px]
else:
    scol=vis.to_color().vertex_colors[mesh.faces[fi][:,0],:3].astype(np.float32)/255
idx=np.where(shell.reshape(-1))[0]; q=P[idx]
_,nn=cKDTree(sp).query(q,k=8); col=scol[nn].mean(1)
V=np.ones((N**3,3),np.float32); V[idx]=col
V=V.reshape(N,N,N,3).transpose(3,0,1,2)*2-1
V[:,core]=torch.tensor(col.mean(0)*2-1).numpy()[:,None]               # placeholder; the carrier fits the interior
G=dict(V=torch.from_numpy(V.copy()),OCC=torch.from_numpy(occ[None].astype(np.float32)),
       SHELL=torch.from_numpy(shell[None].astype(np.float32)),CORE=torch.from_numpy(core[None].astype(np.float32)),N=N,
       mesh_scale=float(s),mesh_ctr=torch.tensor(ctr))
torch.save(G,out); print(out,"occ",int(occ.sum()),"shell",int(shell.sum()),"core",int(core.sum()))
