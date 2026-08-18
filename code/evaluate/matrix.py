"""Every object's renders against every object's reference set.

The claim the matrix supports is that the distance reads the interior and not colour or framing:
if it read those, the six reference sets would be interchangeable and the diagonal would not
stand out. It is one number per pair, so it needs the scoring interpreter rather than the
rendering one.

    python code/evaluate/matrix.py                 # after measure.py has rendered the cuts
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.environ["FN_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PY = os.environ.get("FN_PY_SCORE") or sys.exit("set FN_PY_SCORE: this needs DreamSim")
OUT = os.path.join(ROOT, os.environ.get("EVAL_OUT", "measurements"))
OBJECTS = ["orange", "watermelon", "apple", "pomegranate", "bread", "cake"]


def refdir(obj):
    for line in open(f"{CODE}/objects/{obj}.conf"):
        m = re.match(r"^(EVAL_REF|REF_H)=(\S+)", line.strip())
        if m and m.group(1) == "EVAL_REF":
            return m.group(2)
    for line in open(f"{CODE}/objects/{obj}.conf"):
        m = re.match(r"^REF_H=(\S+)", line.strip())
        if m:
            return m.group(1)


M = {}
for rows in OBJECTS:
    cuts = f"{OUT}/{rows}/cuts"
    if not os.path.isdir(cuts):
        print(f"  {rows}: no cuts rendered, run measure.py first")
        continue
    for cols in OBJECTS:
        ref = refdir(cols)
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join([HERE, f"{CODE}/src", f"{CODE}/inherited"])
        p = subprocess.run([PY, f"{HERE}/realism.py", ref, f"{rows}={cuts}"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        m = re.search(rf"^\s*{rows}\s+\S+\s+\S+\s+([0-9.]+)", p.stdout, re.M)
        M.setdefault(rows, {})[cols] = float(m.group(1)) if m else None
        print(f"  {rows:12s} vs {cols:12s} {M[rows][cols]}", flush=True)

json.dump(M, open(f"{OUT}/matrix.json", "w"), indent=1)
print(f"\n{'renders \\ refs':16s}" + "".join(f"{c[:9]:>10s}" for c in OBJECTS))
for rw in OBJECTS:
    if rw in M:
        print(f"{rw:16s}" + "".join(
            f"{(f'{M[rw][c]:.4f}' if M[rw].get(c) is not None else '-'):>10s}" for c in OBJECTS))
print(f"-> {OUT}/matrix.json")
