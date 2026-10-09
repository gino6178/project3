# A new object into the paper's carrier pipeline by its first route (code/README.md): a coloured
# Gaussian point set, quantised onto the lattice by voxelize.py "as it is", the exterior coming in
# with it.  Points sit at the centres of a 2x-refined occupancy (one per fine cell, as the released
# models are filled); the shell carries the object's own exterior colour, the interior ONE mean
# colour -- the carrier must learn the inside from the cross-sections alone, nothing leaks in.
# Frame: the grid's polar axis (dim 1) becomes world y, the released orange's axis (its physics file
# says up = -y), and the solid is scaled and placed like the released orange.
#   python make_gsply.py <grid.pt (Vol format)> <out.ply> <out.json>
import sys,json,numpy as np,torch,torch.nn.functional as F
src,out,meta=sys.argv[1:4]; C0=0.28209479177387814
G=torch.load(src,map_location="cpu"); N=G["N"]; V=G["V"].float(); OCC=G["OCC"].float(); CORE=G["CORE"].float()
M=2*N
occ=F.interpolate(OCC[None],size=(M,M,M),mode="trilinear",align_corners=True)[0,0]>0.5
core=F.interpolate(CORE[None],size=(M,M,M),mode="trilinear",align_corners=True)[0,0]>0.5
Vu=F.interpolate(V[None],size=(M,M,M),mode="trilinear",align_corners=True)[0]
idx=occ.nonzero().float()                                          # (n,3) in fine grid units, dims (u0, v, u1)
rgb=((Vu[:,occ].T+1)/2).clamp(0,1)
mean=rgb[core[occ]].mean(0) if core.any() else rgb.mean(0)
rgb[core[occ]]=mean                                                # the interior: one colour, nothing to copy
SPAN=1.52; CTR=np.array([0.076,0.71,0.60])                          # the released orange's span and centre
pos=(idx/(M-1)-0.5)*SPAN; xyz=np.stack([pos[:,0].numpy(),pos[:,1].numpy(),pos[:,2].numpy()],1)+CTR
n=len(xyz); fdc=((rgb.numpy()-0.5)/C0).astype("<f4")
names=["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
a=np.zeros((n,len(names)),"<f4"); a[:,0:3]=xyz; a[:,6:9]=fdc; a[:,9]=10000.0; a[:,10:13]=-16.0; a[:,13]=1.0
hdr=("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"%n+"".join(f"property float {k}\n" for k in names)+"end_header\n").encode()
open(out,"wb").write(hdr+a.tobytes())
occ_xyz=xyz; span=float((occ_xyz.max(0)-occ_xyz.min(0)).max())
json.dump(dict(src=src,N=N,M=M,span=SPAN,ctr=CTR.tolist(),coarse_dx=span/125,points=n,interior_colour=mean.tolist()),open(meta,"w"),indent=1)
print(out,n,"points; coarse dx",round(span/125,5))
