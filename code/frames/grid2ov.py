# The lifted interior (object-grid frame) into the frame of voxelize_ov.py's grid, so that the
# paper's writeback.py can bake it into the trained O-Voxel PLY's interior cells -- the inverse of
# ov2grid.py, by the same affine make_gsply.py placed the object with.
#   python grid2ov.py <object grid.pt> <make_gsply json> <voxelize_ov grid> <state.pt> <out state_ov.pt>
import sys,json,torch,torch.nn.functional as F
gsrc,mj,ovp,sp,out=sys.argv[1:6]
G=torch.load(gsrc,map_location="cpu"); N=G["N"]; J=json.load(open(mj)); O=torch.load(ovp,map_location="cpu")
X=torch.load(sp,map_location="cpu")["X"].float(); M,SPAN,CTR=J["M"],J["span"],torch.tensor(J["ctr"])
No=O["N"]; lin=torch.arange(No).float(); A,B,C=torch.meshgrid(lin,lin,lin,indexing="ij")
xyz=(torch.stack([A,B,C],-1)/(No-1)-0.5)*O["ext"]+O["ctr"]               # voxelize_ov index -> world
fine=((xyz-CTR)/SPAN+0.5)*(M-1); idx=fine*(N-1)/(M-1)                    # world -> object grid index
gg=torch.stack([idx[...,2],idx[...,1],idx[...,0]],-1)/(N-1)*2-1
Xo=F.grid_sample(X[None],gg.reshape(1,No,No,No,3),mode="bilinear",padding_mode="border",align_corners=True)[0]
torch.save({"X":Xo},out); print(out)
