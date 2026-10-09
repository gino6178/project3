# The paper's fitted carrier (voxelize_ov.py's dense grid, in the PLY's own frame) back onto the
# object's grid (the frame its photographs, its ground truth and the lift share), by the exact affine
# make_gsply.py placed it with.  Only V changes; OCC/SHELL/CORE stay the object's.
#   python ov2grid.py <object grid.pt> <make_gsply json> <voxelize_ov grid> <out carrier.pt>
import sys,json,torch,torch.nn.functional as F
gsrc,mj,ovp,out=sys.argv[1:5]
G=torch.load(gsrc,map_location="cpu"); N=G["N"]; J=json.load(open(mj)); O=torch.load(ovp,map_location="cpu")
M,SPAN,CTR=J["M"],J["span"],torch.tensor(J["ctr"])
lin=torch.arange(N).float(); A,B,C=torch.meshgrid(lin,lin,lin,indexing="ij")
fine=torch.stack([A,B,C],-1)*(M-1)/(N-1)                          # object grid index -> make_gsply fine index
xyz=(fine/(M-1)-0.5)*SPAN+CTR                                      # -> world, as written into the PLY
No=O["N"]; idx=((xyz-O["ctr"])/O["ext"]+0.5)*(No-1)               # -> voxelize_ov's grid index
gg=torch.stack([idx[...,2],idx[...,1],idx[...,0]],-1)/(No-1)*2-1
V=F.grid_sample(O["V"].float()[None],gg.reshape(1,N,N,N,3),mode="bilinear",padding_mode="border",align_corners=True)[0]
G2=dict(G); G2["V"]=torch.where(G["CORE"].bool().expand(3,-1,-1,-1),V,G["V"].float())
torch.save(G2,out); print(out,"interior from the paper's carrier; shell kept")
