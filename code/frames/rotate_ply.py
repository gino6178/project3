# Stand a released Gaussian model the way the paper's carrier expects (its polar axis along world y, the
# top toward -y, as the orange's physics file says), when the capture lies another way.  Positions, normals
# and the Gaussians' own rotations are turned by one proper rotation; colours are DC only, so nothing else
# changes.  The apple's stem and the cake's cream lie along +z and -z in the released files.
#   python rotate_ply.py <in.ply> <out.ply> <+z|-z>      (+z: (x,y,z)->(x,-z,y);  -z: (x,y,z)->(x,z,-y))
import sys,numpy as np
src,dst,top=sys.argv[1:4]
raw=open(src,"rb").read(); end=raw.index(b"end_header\n")+len(b"end_header\n"); hdr=raw[:end].decode()
names=[l.split()[-1] for l in hdr.splitlines() if l.startswith("property")]
assert all(l.split()[1]=="float" for l in hdr.splitlines() if l.startswith("property")) and "f_rest_0" not in names
a=np.frombuffer(raw[end:],dtype=np.dtype([(n,"<f4") for n in names])).copy()
Q=np.array([[1,0,0],[0,0,-1],[0,1,0]],float) if top=="+z" else np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
def turn(keys):
    P=np.stack([a[k] for k in keys],1).astype(np.float64)@Q.T
    for i,k in enumerate(keys): a[k]=P[:,i]
turn(["x","y","z"]); turn(["nx","ny","nz"])
# the Gaussians' orientation: q' = qQ * q  (quaternions w,x,y,z = rot_0..rot_3)
w=np.sqrt(max(1e-12,1+np.trace(Q)))/2; qQ=np.array([w,(Q[2,1]-Q[1,2])/(4*w),(Q[0,2]-Q[2,0])/(4*w),(Q[1,0]-Q[0,1])/(4*w)])
q=np.stack([a[f"rot_{i}"] for i in range(4)],1).astype(np.float64)
w1,x1,y1,z1=qQ; w2,x2,y2,z2=q.T
r=np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],1)
for i in range(4): a[f"rot_{i}"]=r[:,i]
open(dst,"wb").write(raw[:end]+a.tobytes()); print(dst,len(a),"Gaussians, top",top,"-> -y")
