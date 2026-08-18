"""Every number on the page, from the models that are on disk now.

One process per measurement, its stdout kept verbatim, and the values it printed pulled out into
one JSON file. The raw log is the record; the JSON is the convenience. A measurement that has
already run is skipped unless --force, so this resumes.

    python code/evaluate/measure.py            # every object, the core set
    python code/evaluate/measure.py orange     # one object
    python code/evaluate/measure.py --heavy    # and the sweeps, which take hours

Two interpreters, because no single one has everything: FN_PY renders and needs the CUDA
rasteriser; FN_PY_SCORE only reads images and needs DreamSim, which the render environment does
not have. If FN_PY_SCORE is unset the DreamSim rows are skipped and say so.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.environ.get("FN_ROOT") or sys.exit("set FN_ROOT")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PY = os.environ.get("FN_PY") or sys.exit("set FN_PY")
PY_SCORE = os.environ.get("FN_PY_SCORE", "")
OUT = os.path.join(ROOT, os.environ.get("EVAL_OUT", "measurements"))
OBJECTS = ["orange", "watermelon", "apple", "bread", "cake", "pomegranate", "doughnut"]

# What each object was trained from and what it produced. Both are properties of the run, so
# they are read off the disk rather than assumed.
def where(obj):
    conf = {}
    for line in open(f"{CODE}/objects/{obj}.conf"):
        m = re.match(r"^(\w+)=(\S+)", line.strip())
        if m:
            conf[m.group(1)] = m.group(2)
    lat = f"build_{obj}/skin" if os.path.isfile(f"{ROOT}/build_{obj}/skin/gs_fill.ply") \
        else f"build_{obj}/lattice"
    it = int(re.sub(r"\D", "", conf.get("ITERS", "200")) or 200)
    cands = [os.environ.get(f"MODEL_{obj.upper()}", ""), f"{obj}/orange_demo_epoch_{it-1}.ply"]
    model = next((c for c in cands if c and os.path.isfile(f"{ROOT}/{c}")), None)
    return conf, lat, model


def arms(obj, model):
    """What to score for this object. The released models are the comparison and they are not
    retrained by anything here -- they are read off disk and put through the same protocol, which
    is the only way a row of a comparison table means anything."""
    a = {"ours": model}
    f = os.path.join(HERE, "arms.json")
    if os.path.isfile(f):
        for k, v in json.load(open(f)).get(obj, {}).items():
            if os.path.isfile(os.path.join(ROOT, v)) or os.path.isfile(v):
                a[k] = v
            else:
                print(f"    (no {k} for {obj} at {v})")
    return a


def run(name, obj, argv, py=None, force=False):
    """One measurement. Returns its stdout, cached in measurements/<obj>/<name>.log."""
    d = os.path.join(OUT, obj)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, name + ".log")
    if os.path.isfile(log) and not force and os.path.getsize(log) > 0:
        return open(log).read(), True
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([HERE, f"{CODE}/src", f"{CODE}/figures", f"{CODE}/inherited",
                                  f"{CODE}/inherited/mpm_solver_warp",
                                  env.get("GS_ROOT", f"{ROOT}/gaussian-splatting")])
    t0 = time.time()
        # The band the page's numbers were measured in, and the reason they are held out: without
    # it random_cuts puts the plane at 0.04-0.15 or 0.85-0.96 of the axis, which is the two ends
    # of the object, and what comes back is very nearly the exterior. Scored against cross-section
    # photographs that reads as a distance of 0.40 where the middle of the object reads 0.07.
    # FULL_SH because a trained model keeps directional appearance in bands the plain loader drops.
    env.setdefault("HELDOUT_BAND", "0.30,0.70")
    env.setdefault("FULL_SH", "1")
    p = subprocess.run([py or PY] + argv, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    body = p.stdout + ("\n--- stderr\n" + p.stderr[-4000:] if p.returncode else "")
    body = f"# {' '.join(argv)}\n# exit {p.returncode} in {time.time()-t0:.1f}s\n" + body
    open(log, "w").write(body)
    return body, False


def grab(text, pattern, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def measure(obj, heavy=False, force=False):
    conf, lat, model = where(obj)
    if model is None:
        return {"error": "no trained model on disk"}
    cfg, demo = conf["CFG"], conf["DEMO"]
    r = {"lattice": lat, "model": model}
    def tool(n):
        """Where a measurement lives: the method in src/, the instruments beside this file."""
        for base in (HERE, f"{CODE}/src"):
            if os.path.isfile(f"{base}/{n}"):
                return f"{base}/{n}"
        raise SystemExit(f"no such tool: {n}")
    S = type("S", (), {"__truediv__": staticmethod(tool)})

    # held-out cuts: everything that scores a section reads these
    cuts = f"{OUT}/{obj}/cuts"
    out, _ = run("random_cuts", obj, [tool("random_cuts.py"), model, cfg, demo, cuts, "12"],
                 force=force)
    r["cuts_rendered"] = grab(out, r"(\d+)\s+cuts") or 12

    # the exterior, against the model's own lattice: the claim is that it is unchanged by training
    base = f"{OUT}/{obj}/ext_lattice.png"
    run("exterior_lattice", obj, [tool("exterior_views.py"), lat, cfg, demo, base, "384"],
        force=force)
    run("exterior_model", obj, [tool("exterior_views.py"), model, cfg, demo,
                                f"{OUT}/{obj}/ext_model.png", "384"], force=force)
    out, _ = run("exterior_delta", obj, [tool("extdelta.py"), base,
                                         f"{OUT}/{obj}/ext_model.png"], force=force)
    r["exterior_mean"] = grab(out, r"mean ([0-9.]+)")
    r["exterior_over015_pct"] = grab(out, r"over 0\.15 ([0-9.]+)")
    r["exterior_specks"] = grab(out, r"specks (\d+)", int)

    # flicker, which needs no photograph at all
    out, _ = run("slicing", obj, [tool("slicing_consistency.py"), model, cfg, demo,
                                  f"{OUT}/{obj}/slicing"], force=force)
    r["slicing_ssim_per_step"] = grab(out, r"1-SSIM per step\s+([0-9.]+)")
    r["slicing_jerk"] = grab(out, r"jerk \(sd of the step\)\s+([0-9.]+)")

    # what the lattice costs
    out, _ = run("memory", obj, [tool("memory.py"), lat, model], force=force)
    r["memory_raw"] = out.strip().splitlines()[-6:]
    out, _ = run("timing", obj, [tool("timing.py"), lat, model], force=force)
    r["timing_raw"] = out.strip().splitlines()[-8:]

    # appearance against the photographs, by the instruments that survive the sample size
    ref = conf.get("EVAL_REF", conf.get("REF_H", ""))
    nref = len([f for f in os.listdir(f"{ROOT}/{ref}")
                if f.lower().endswith((".png", ".jpg"))]) if ref and os.path.isdir(f"{ROOT}/{ref}") else 0
    r["references"] = nref
    if PY_SCORE and nref:
        out, _ = run("realism", obj, [tool("realism.py"), ref, f"{obj}={cuts}"],
                     py=PY_SCORE, force=force)
        r["dreamsim"] = grab(out, rf"{obj}\s+[-0-9.nan]+\s+[-0-9.nan]+\s+([0-9.]+)")
        r["precision"] = grab(out, rf"{obj}\s+([0-9.]+)")
        r["realism_raw"] = [l for l in out.splitlines() if obj in l or "photographs" in l]
    else:
        r["dreamsim"] = None
    if nref >= 10:      # below this FID is dominated by its own bias; the page says so
        out, _ = run("fid", obj, [tool("fid_eval.py"), ref] +
                     [f"{cuts}/{f}" for f in sorted(os.listdir(cuts))
                      if f.startswith("rh") and f.endswith("_init_0.png")], force=force)
        r["fid"] = grab(out, r"FID\s+([0-9.]+)")
        r["kid"] = grab(out, r"KID\s+([-0-9.e]+)")
    else:
        r["fid"] = r["kid"] = None
        r["fid_note"] = f"{nref} references: below where FID separates anything"

    # what the representation costs through one codec, both ways of counting
    out, _ = run("compress", obj, [tool("compress.py"), f"{obj}={model}", f"lattice={lat}"],
                 force=force)
    r["compress_raw"] = out.strip().splitlines()[-6:]

    # how much of a cut face has nothing behind it, at three sizes, on this object's own cuts
    out, _ = run("unpainted", obj, [tool("unpainted.py"), f"{obj}={cuts}"], force=force)
    r["unpainted_pct"] = grab(out, r"([0-9.]+)%")
    r["unpainted_raw"] = out.strip().splitlines()[-5:]

    if heavy:
        for nm, argv in (("resolution", [tool("resolution.py"), lat, model, cfg, demo,
                                         f"{OUT}/{obj}/res"]),
                         ("rho_sweep", [tool("rho_sweep.py"), lat, model, cfg, demo,
                                        f"{OUT}/{obj}/rho"])):
            if os.path.isfile(argv[0]):
                out, _ = run(nm, obj, argv, force=force)
                r[nm + "_raw"] = out.strip().splitlines()[-10:]
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="*", default=[])
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    objs = a.objects or OBJECTS
    os.makedirs(OUT, exist_ok=True)
    allr = {}
    p = os.path.join(OUT, "results.json")
    if os.path.isfile(p):
        allr = json.load(open(p))
    for o in objs:
        print(f"=== {o}", flush=True)
        allr[o] = measure(o, heavy=a.heavy, force=a.force)
        for k, v in allr[o].items():
            if not k.endswith("_raw"):
                print(f"    {k:24s} {v}")
        json.dump(allr, open(p, "w"), indent=1)
    print(f"\n-> {p}")
