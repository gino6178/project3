"""Render a model at the six directions the exterior branch supervises, in its order.

EXT_CUBE in train_voxel is [(0,90),(0,-90),(0,0),(90,0),(180,0),(270,0)] for
up, down, front, right, back, left -- view ttt is compared against o{ttt}_ref.png, so the
references have to be rendered at those angles and written in that order or every view is
matched to a picture of a different side.
"""
import os as _os
# The repository root, so this runs on another machine too. See method/README.md: eight
# scripts had this written three times each and a run on the remote box failed with "no
# such file" for a file that was plainly there, because the chdir had moved underneath it.
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import sys, os, argparse
sys.path.append(_FN_ROOT)
sys.path.append(_FN_ROOT + "/gaussian-splatting")
os.chdir(_FN_ROOT)
import torch, numpy as np, cv2
from scene.gaussian_model import GaussianModel
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs, initialize_resterize, convert_SH
from utils.transformation_utils import *
from utils.camera_view_utils import get_camera_view
DEV="cuda:0"
CUBE=[("up",0,90),("down",0,-90),("front",0,0),("right",90,0),("back",180,0),("left",270,0)]
class P: convert_SHs_python=False; compute_cov3D_python=True; debug=False

def main(ply, cfg, demo, out_dir, size=512, min_scale=None):
    os.makedirs(out_dir, exist_ok=True)
    (mat,bc,tp,pre,cam_p)=decode_param_json(cfg)
    g=GaussianModel(0); g.load_ply_zero_sh(ply)
    if min_scale is not None:
        # Their plys store a placeholder log-scale near -20, so every primitive is a
        # point far below one pixel and a render of them is speckle, not a surface. Their
        # own pipeline never renders them at this count -- it densifies to millions first.
        # Raise the floor so the primitives tile the surface they came from.
        with torch.no_grad():
            g._scaling.clamp_(min=float(np.log(min_scale)))
    par=load_params_from_gs(g,P())
    pos0,cov0=par["pos"],par["cov3D_precomp"]
    sp,op,shs=par["screen_points"],par["opacity"],par["shs"]
    rot_m=generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),pre["rotation_axis"])
    vc_c=torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape((1,3)).cuda()
    up=torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(3).float().cuda(); up=up/up.norm()
    tpos,so,om=transform2origin(pos0); tpos=shift2center111(tpos)
    cov0=apply_cov_rotations(cov0,rot_m); cov0=so*so*cov0
    cov=apply_inverse_cov_rotations(cov0/(so*so),rot_m)
    world=apply_inverse_rotations(undotransform2origin(undoshift2center111(tpos.to(DEV)),so,om),rot_m)
    vc,oc=get_center_view_worldspace_and_observant_coordinate(vc_c,up,rot_m,so,om)
    bg=torch.tensor([1.,1.,1.],device=DEV)
    for i,(n,az,el) in enumerate(CUBE):
        cam,_=get_camera_view(demo,default_camera_index=-1,center_view_world_space=vc,
            observant_coordinates=oc,show_hint=False,init_azimuthm=az,init_elevation=el,
            init_radius=cam_p["init_radius"],move_camera=False,current_frame=0,
            delta_a=None,delta_e=None,delta_r=None)
        rast=initialize_resterize(cam,g,P(),bg,image_height=size,image_width=size)
        col=convert_SH(shs,cam,g,world,None)
        img,_,_,_=rast(means3D=world,means2D=sp,shs=None,colors_precomp=col,
                       opacities=op,scales=None,rotations=None,cov3D_precomp=cov)
        a=img.permute(1,2,0).detach().clamp(0,1).cpu().numpy()
        cv2.imwrite(os.path.join(out_dir,f"o{i}_ref.png"),cv2.cvtColor(a,cv2.COLOR_BGR2RGB)*255)
    print(f"  {os.path.basename(ply)}: {world.shape[0]:,} prims -> {out_dir}  ({', '.join(n for n,_,_ in CUBE)})")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("ply"); ap.add_argument("cfg"); ap.add_argument("demo"); ap.add_argument("out")
    ap.add_argument("--min-scale", type=float, default=None)
    a=ap.parse_args()
    main(a.ply,a.cfg,a.demo,a.out,min_scale=a.min_scale)
