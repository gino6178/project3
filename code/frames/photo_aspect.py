# Height over width of an object's training lengthwise photographs (accepted, stood upright), the aspect
# glb2grid.py's ASPECT corrects a generated exterior to -- a generated mesh need not share the cut faces' shape.
#   python photo_aspect.py <aiobj root> <object>      -> prints the mean aspect
import sys,glob,os,numpy as np
from PIL import Image
from scipy import ndimage as ndi
root,obj=sys.argv[1],sys.argv[2]; rej=set()
for l in open(f"{root}/REJECT.txt"):
    f=l.split()
    if f and not l.startswith("#") and f[0]==obj: rej|=set(f[1:])
ps=[p for p in sorted(glob.glob(f"{root}/{obj}/raw/long_*.png"),key=lambda s:int(s.rsplit("_",1)[1][:-4])) if os.path.basename(p)[:-4] not in rej][:3]
asp=[]
for p in ps:
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255
    bg=np.median(np.concatenate([a[:16,:16].reshape(-1,3),a[-16:,-16:].reshape(-1,3)]),0); m=np.abs(a-bg).max(2)>0.10
    m=ndi.binary_opening(m,iterations=2); lab,n=ndi.label(m); m=lab==(np.argmax(ndi.sum(m,lab,range(1,n+1)))+1)
    ys,xs=np.where(m); h,w=ys.max()-ys.min(),xs.max()-xs.min(); asp.append(max(h,w)/min(h,w))   # stood upright
print(round(float(np.mean(asp)),3))
