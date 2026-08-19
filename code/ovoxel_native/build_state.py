"""Build the O-Voxel state for whichever route the variable names.

    ROUTE=1  build_orange/lattice        the released ply, quantised as it is: shape and
                                         appearance both come from a pre-existing Gaussian model
    ROUTE=2  build_orange_r2/skin        shape from make_shape.py's ellipsoid SDF, exterior from
                                         skin_project.py's six-view projection -- nothing here owes
                                         anything to a released model

One program, one function, one variable.  `ovnative.build` already reads only lattice.pt,
gs_fill.ply and cell_level.pt, which both routes write, so the route is a directory and not a code
path.
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, "/workspace/ovoxel_native")
import ovnative as ON

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
ROUTE = os.environ.get("ROUTE", "1")
LATDIR = {"1": f"{FN}/build_orange/lattice", "2": f"{FN}/build_orange_r2/skin"}[ROUTE]
OUT = os.environ.get("STATE", f"/workspace/ovoxel_native/state_r{ROUTE}.pt")
dev = "cuda"

lat = torch.load(os.path.join(LATDIR, "lattice.pt"))
lvl = torch.load(os.path.join(LATDIR, "cell_level.pt")).reshape(-1)
print(f"route {ROUTE}: {LATDIR}")
print(f"  lattice.pt keys: {sorted(lat.keys())}")
print(f"  coarse_dx {float(lat['coarse_dx']):.6f}  fine_dx {float(lat['fine_dx']):.6f}")
u, c = torch.unique(lvl, return_counts=True)
print(f"  cell_level: {[(int(a), int(b)) for a, b in zip(u, c)]}")

t0 = time.time()
st = ON.build(LATDIR, device=dev)
print(f"built in {time.time()-t0:.1f}s")
for k in ("coords", "dual_v", "split_w", "surf_rgb", "interior", "solid"):
    print(f"  {k:<10} {tuple(st[k].shape)}")
print(f"  surf_rgb mean {st['surf_rgb'].mean(0).cpu().numpy().round(4)} "
      f"std {st['surf_rgb'].std(0).cpu().numpy().round(4)}")
print(f"  interior mean {st['interior'].mean(0).cpu().numpy().round(4)} "
      f"std {st['interior'].std(0).cpu().numpy().round(4)}")
torch.save(st, OUT)
print(f"-> {OUT}")
print("BUILD_OK")
