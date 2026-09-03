# The measured gate: the lift is kept only where it lowers the distance to the object's OWN
# training photographs (which every object has); otherwise the asset keeps the fitted field.
# One rule, no threshold, never below the baseline; held-out photographs are never consulted.
import sys,re,math
txt=open(sys.argv[1]).read()
def row(name):
    m=re.search(name+r"\s+([\d.na]+)\s+([\d.na]+)",txt); return [float(v) if v!="nan" else math.nan for v in m.groups()] if m else [math.nan,math.nan]
ov,cy=row("O-Voxel"),row("cyl")
pairs=[(o,c) for o,c in zip(ov,cy) if not (math.isnan(o) or math.isnan(c))]
keep=all(c<o for o,c in pairs)
print(("LIFT" if keep else "FITTED"),"  O-Voxel",ov,"cyl",cy)
