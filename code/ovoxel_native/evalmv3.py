"""Every arm, both families, one table. DreamSim from evaluate/realism.py, imported not copied.

L1 is reported beside DreamSim throughout, and the two do not always agree. The arms with
SEC_PATCH train on a different objective (0.7(1-SSIM)+0.3MSE on crops plus a band term), so their
*training* losses are not comparable to the others; the L1 probe and DreamSim are, because both are
measured after the fact on the same twelve held-out cuts.
"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, "/workspace/rebuild/project3/code/evaluate")
import realism

FN = os.environ.get("FN_ROOT", "/workspace/rebuild/worktree")
W = "/workspace/ovoxel_native"
dev = "cuda"
refs_h = realism._paths(os.path.join(FN, "secref_orraw_hsep"))
refs_v = realism._paths(os.path.join(FN, "secref_orraw_vsep"))

ARMS = [
    ("existing pipeline (route 1)",              f"{FN}/eval_orange",        None),
    ("existing pipeline (route 2)",              f"{FN}/eval_orange_r2",     None),
    ("r1 free RGB, trained  (ANCHOR=0)",         f"{W}/mv_free/eval_final",  "mv_free"),
    ("r1 free RGB, pinned   (ANCHOR=0)",         f"{W}/mv_pin/eval_final",   "mv_pin"),
    ("r1 decoder, initialised",                  f"{W}/r1_free/eval_init",   None),
    ("r1 decoder, trained",                      f"{W}/r1_free/eval_final",  "r1_free"),
    ("r1 decoder, pinned",                       f"{W}/r1_pin/eval_final",   "r1_pin"),
    ("r1 decoder, FLAT interior, trained",       f"{W}/r1flat_free/eval_final", "r1flat_free"),
    ("r1 decoder, FLAT interior, pinned",        f"{W}/r1flat_pin/eval_final",  "r1flat_pin"),
    ("r1 pinned + SEC_PATCH",                    f"{W}/r1_pin_patch/eval_final", "r1_pin_patch"),
    ("r1 pinned + VOXEL_SMOOTH",                 f"{W}/r1_pin_vs/eval_final",    "r1_pin_vs"),
    ("r1 pinned + both (full parity)",           f"{W}/r1_pin_full/eval_final",  "r1_pin_full"),
    ("r2 decoder, initialised",                  f"{W}/r2_free/eval_init",   None),
    ("r2 decoder, trained",                      f"{W}/r2_free/eval_final",  "r2_free"),
    ("r2 decoder, pinned",                       f"{W}/r2_pin/eval_final",   "r2_pin"),
    ("r2 pinned + both (full parity)",           f"{W}/r2_pin_full/eval_final",  "r2_pin_full"),
]

h = len(refs_h) // 2
print(f"  {'arm':<40} {'DS rh':>7} {'DS rv':>7} {'probe L1':>9}")
print(f"  {'the photographs, split in half':<40} "
      f"{realism._dreamsim(refs_h[:h], refs_h[h:], dev):>7.4f} "
      f"{realism._dreamsim(refs_v[:3], refs_v[3:], dev):>7.4f} {'':>9}")
res = {}
for tag, d, run in ARMS:
    ph = sorted(glob.glob(os.path.join(d, "rh*_init_0.png"))) if os.path.isdir(d) else []
    if not ph:
        print(f"  {tag:<40} {'(missing)':>7}")
        continue
    pv = sorted(glob.glob(os.path.join(d, "rv*_init_0.png")))
    a = realism._dreamsim(refs_h, ph, dev)
    b = realism._dreamsim(refs_v, pv, dev) if pv else float("nan")
    pr = ""
    if run and os.path.exists(f"{W}/{run}/hist.json"):
        pr = f"{json.load(open(f'{W}/{run}/hist.json'))['probe'][-1][1]:.5f}"
    res[tag] = (a, b)
    print(f"  {tag:<40} {a:>7.4f} {b:>7.4f} {pr:>9}")

print("\nwhat each mechanism is worth, on r1 pinned (baseline 0.0504)")
base = res.get("r1 decoder, pinned", (float('nan'),))[0]
for t in ("r1 pinned + SEC_PATCH", "r1 pinned + VOXEL_SMOOTH", "r1 pinned + both (full parity)"):
    if t in res:
        print(f"  {t:<40} {res[t][0]:>7.4f}   {res[t][0]-base:+.4f}")
print("\nthe interior initialisation, on r1")
for a, b in (("r1 decoder, trained", "r1 decoder, FLAT interior, trained"),
             ("r1 decoder, pinned", "r1 decoder, FLAT interior, pinned")):
    if a in res and b in res:
        print(f"  {b:<40} {res[b][0]:>7.4f}   {res[b][0]-res[a][0]:+.4f} vs seeded from f_dc")
print("EVAL_OK")
