"""Every arm of every object, on the same cuts, by the same instruments.

`measure.py` scores the model this work produced. The comparison tables need the released models
scored the same way, and they are not retrained by anything here: they are read off disk, put
through `random_cuts` at the same depths, and scored by the same tools. A row measured any other
way is not comparable, which is the mistake this file exists to make impossible.

    python code/evaluate/compare.py            # every object that has an arm in arms.json
    python code/evaluate/compare.py watermelon
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.environ["FN_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PY = os.environ["FN_PY"]
PY_SCORE = os.environ.get("FN_PY_SCORE", "")
OUT = os.path.join(ROOT, os.environ.get("EVAL_OUT", "measurements"))
ARMS = json.load(open(f"{HERE}/arms.json"))


def conf(obj):
    d = {}
    for line in open(f"{CODE}/objects/{obj}.conf"):
        m = re.match(r"^(\w+)=(\S+)", line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


def sh(argv, py=None, log=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([HERE, f"{CODE}/src", f"{CODE}/figures", f"{CODE}/inherited",
                                  f"{CODE}/inherited/mpm_solver_warp",
                                  env.get("GS_ROOT", f"{ROOT}/gaussian-splatting")])
        # The band the page's numbers were measured in, and the reason they are held out: without
    # it random_cuts puts the plane at 0.04-0.15 or 0.85-0.96 of the axis, which is the two ends
    # of the object, and what comes back is very nearly the exterior. Scored against cross-section
    # photographs that reads as a distance of 0.40 where the middle of the object reads 0.07.
    # FULL_SH because a trained model keeps directional appearance in bands the plain loader drops.
    env.setdefault("HELDOUT_BAND", "0.30,0.70")
    env.setdefault("FULL_SH", "1")
    p = subprocess.run([py or PY] + argv, cwd=ROOT, env=env, capture_output=True, text=True)
    if log:
        open(log, "w").write(" ".join(argv) + f"\n# exit {p.returncode}\n" + p.stdout +
                             ("\n--- stderr\n" + p.stderr[-3000:] if p.returncode else ""))
    return p.stdout


res = {}
objs = sys.argv[1:] or [k for k in ARMS if not k.startswith("_")]   # "_" keys are notes
for obj in objs:
    c = conf(obj)
    ref = c.get("EVAL_REF", c.get("REF_H"))
    for arm, ply in ARMS.get(obj, {}).items():
        path = ply if os.path.isabs(ply) else f"{ROOT}/{ply}"
        if not os.path.isfile(path):
            print(f"  {obj}/{arm}: not on disk"); continue
        d = f"{OUT}/{obj}/{arm}"
        os.makedirs(d, exist_ok=True)
        cuts = f"{d}/cuts"
        if not os.path.isdir(cuts):
            print(f"  {obj}/{arm}: rendering cuts", flush=True)
            sh([f"{CODE}/src/random_cuts.py", path, c["CFG"], c["DEMO"], cuts, "12"],
               log=f"{d}/random_cuts.log")
        r = {}
        if PY_SCORE and ref:
            out = sh([f"{HERE}/realism.py", ref, f"{arm}={cuts}"], py=PY_SCORE,
                     log=f"{d}/realism.log")
            m = re.search(rf"^\s*{arm}\s+\S+\s+\S+\s+([0-9.]+)", out, re.M)
            r["dreamsim"] = float(m.group(1)) if m else None
        out = sh([f"{HERE}/unpainted.py", f"{arm}={cuts}"], log=f"{d}/unpainted.log")
        m = re.search(r"([0-9.]+)%", out)
        r["unpainted_pct"] = float(m.group(1)) if m else None
        res.setdefault(obj, {})[arm] = r
        print(f"  {obj:12s} {arm:16s} {r}", flush=True)

p = f"{OUT}/compare.json"
old = json.load(open(p)) if os.path.isfile(p) else {}
for k, v in res.items():
    old.setdefault(k, {}).update(v)
json.dump(old, open(p, "w"), indent=1)
print(f"-> {p}")
