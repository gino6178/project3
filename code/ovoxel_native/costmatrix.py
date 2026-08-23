"""The resource matrix rule 6 asks for: alignment, optimiser peak, per-cut latency, triangles.

Each number is timed here rather than quoted. The cut is timed warm, after one discarded call, and
reported as the median of 20; the alignment is the closed-form similarity fit of one reference to
one render; the optimiser peak is torch's own high-water mark over one full gradient step.
"""
import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import ovnative as ON, section_match as sm, refsel
import nvdiffrast.torch as dr

W = os.path.dirname(os.path.abspath(__file__))
OBJDIR = "/workspace/rebuild/project3/code/objects"
FN = "/workspace/rebuild/worktree"
dev = "cuda"
RES = int(os.environ.get("RES", "512"))
ON.FDG = ON._load_ovoxel()
glctx = dr.RasterizeCudaContext(device=dev)
rows = []
for OBJ in ("orange_sp", "watermelon_sp", "apple1_sp", "bread_sp", "cake2_sp",
            "pomegranate2_sp", "doughnut"):
    st = torch.load(f"{W}/state_{OBJ}.pt", map_location=dev, weights_only=False)
    C = np.load(f"{W}/cams_{OBJ}_v2.npz")
    p = torch.load(f"{W}/s_v2_{OBJ}/params.pt", map_location=dev)
    st["dual_v"] = p["dual_v"].to(dev); st["split_w"] = p["split_w"].to(dev)
    k = len(C["v_planes"]) // 2
    mv = torch.as_tensor(C["v_mvp"][k], dtype=torch.float32, device=dev)
    nn = torch.as_tensor(C["v_planes"][k, :3], dtype=torch.float32, device=dev)
    dd = float(C["v_planes"][k, 3])

    with torch.no_grad():                                   # warm
        ON.render_section(st, glctx, mv, nn, dd, RES)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            img, al, K, nf = ON.render_section(st, glctx, mv, nn, dd, RES)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    cut_ms = float(np.median(ts))

    P, T, Kc = ON.cut_polygons(st, nn, dd, device=dev)
    tris = len(T)

    conf = open(f"{OBJDIR}/{OBJ}.conf").read()
    spec = [l.split("=", 1)[1].strip() for l in conf.splitlines() if l.startswith("REF_V=")][0]
    ref = refsel.as_array(refsel.photo(f"{FN}/{spec}", k, len(C["v_planes"])), RES)
    with torch.no_grad():
        _ = sm.section_target(img, ref, alpha=al)           # warm
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            _ = sm.section_target(img, ref, alpha=al)
    align_ms = (time.perf_counter() - t0) / 5 * 1e3

    torch.cuda.reset_peak_memory_stats()
    st2 = dict(st); st2["interior"] = st["interior"].clone().requires_grad_(True)
    im2, _, _, _ = ON.render_section(st2, glctx, mv, nn, dd, RES)
    im2.sum().backward()
    peak = torch.cuda.max_memory_allocated() / 2**20

    rows.append((OBJ, len(st["solid"]), Kc, tris, cut_ms, align_ms, peak))
    print(f"{OBJ:16s} cells {len(st['solid']):>9,}  K {Kc:>7,}  tris {tris:>8,}  "
          f"cut {cut_ms:7.1f} ms  align {align_ms:6.1f} ms  peak {peak:7.0f} MiB")
json.dump(rows, open(f"{W}/costmatrix.json", "w"))
