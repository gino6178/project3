"""Record each body's particles at chosen frames, so deformation can be measured rather than asserted.

DYN_DUMP=f0,f1,...  writes <outdir>/state_<frame>.npz holding every body's particle positions and
the piece each belongs to. Without it nothing changes.

The claim it exists to test is that a stiff peel and a soft interior deform by different amounts.
Rendered frames cannot settle that: a piece that translates and a piece that deforms look alike
from one camera. Removing the best rigid transform between two frames and reporting what is left
can settle it, and that needs the positions.
"""
p = "/workspace/rebuild/project3/code/cut/dynamic_cut.py"
s = open(p).read()

anchor = '            print(f"  frame {f}/{frames}  pieces {len(bodies)}  contacts {nhit_tot}")'
assert s.count(anchor) == 1, s.count(anchor)
s = s.replace(anchor, anchor + '''
        if _DUMP and f in _DUMP:
            import numpy as _np
            _xs, _pid = [], []
            for _bi, _bd in enumerate(bodies):
                _x = _bd["solver"].export_particle_x_to_torch().to(DEV)
                _xs.append(_x.detach().cpu().numpy())
                _pid.append(_np.full(len(_x), _bi, _np.int32))
            _np.savez_compressed(os.path.join(outdir, f"state_{f:04d}.npz"),
                                 x=_np.concatenate(_xs), piece=_np.concatenate(_pid),
                                 idx=_np.concatenate([bodies[i]["pidx"].detach().cpu().numpy()
                                                      for i in range(len(bodies))]))
            print(f"    dumped {sum(len(a) for a in _xs):,} particles at frame {f}", flush=True)''')

anchor2 = "    for f in range(frames):"
assert s.count(anchor2) == 1
s = s.replace(anchor2, '''    _DUMP = {int(v) for v in os.environ.get("DYN_DUMP", "").split(",") if v.strip()}
    if _DUMP:
        print(f"  dumping particle state at frames {sorted(_DUMP)}")
''' + anchor2)

open(p, "w").write(s)
print("  patched; DYN_DUMP hooks:", s.count("_DUMP"))
