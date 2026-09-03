#!/bin/bash
SP=/tmp/claude-1000/-home-gino-project-FruitNinja-clean/e5030f17-1844-4a79-8a91-3194645ab588/scratchpad
PY=/home/gino/miniconda3/envs/fruitninja/bin/python
cd $SP
while [ "$(grep -l OBJ_DONE obj/*/score.log 2>/dev/null | wc -l)" -lt 6 ]; do sleep 60; done
for o in apple bread cake doughnut pomegranate watermelon; do
  $PY writeback.py wt/trained/$o.ply wt/lattice_meta/$o grid_$o.pt obj/$o/cyl/state.pt obj/$o/${o}_x3d.ply 2>&1 | tail -n 1
done
$PY - <<'PY'
import glob,os,re
from PIL import Image,ImageDraw
SP="."
objs=["orange","apple","pomegranate","watermelon","bread","cake","doughnut"]
W=200; rows=[]
for o in objs:
    od=f"obj/{o}" if o!="orange" else "."
    faces=f"{od}/ds_faces"; arm="cyl_4kT" if o=="orange" else "cyl"
    row=[]
    for fam,idx in (("long",1),("trans",2)):
        for name,d in (("O-Voxel",f"{faces}/O-Voxel/{fam}/{idx}.png"),("ours",f"{faces}/{arm}/{fam}/{idx}.png")):
            row.append((f"{name} {fam}",d if os.path.exists(d) else None))
        refs=sorted(glob.glob(f"{faces}/_ref/{fam}/*.png")); row.append((f"photo {fam}",refs[0] if refs else None))
    rows.append((o,row))
out=Image.new("RGB",(80+6*(W+6),len(rows)*(W+8)+24),(255,255,255)); d=ImageDraw.Draw(out)
for j,lab in enumerate(["O-Voxel long","ours long","photo long","O-Voxel trans","ours trans","photo trans"]): d.text((80+j*(W+6)+4,6),lab,fill=(0,0,0))
for i,(o,row) in enumerate(rows):
    y=24+i*(W+8); d.text((4,y+W//2),o,fill=(0,0,0))
    for j,(n,p) in enumerate(row):
        if p: out.paste(Image.open(p).resize((W,W),Image.LANCZOS),(80+j*(W+6),y))
        else: d.text((80+j*(W+6)+60,y+W//2),"(none)",fill=(160,160,160))
out.save("six_objects.png"); print("six_objects.png")
# table
print(f"{'object':12s} {'split':52s} {'long':>7s} {'trans':>7s} {'mean':>7s}   O-Voxel long/trans/mean")
for o in objs:
    od=f"obj/{o}" if o!="orange" else None
    if od is None: print(f"{'orange':12s} {'3+3 / 3+3 held out':52s} {0.112:7.3f} {0.062:7.3f} {0.087:7.3f}   0.148 / 0.077 / 0.113"); continue
    sp=open(f"{od}/SPLIT.txt").read().strip().replace("\n","; ")[:52]
    sc=open(f"{od}/score.log").read()
    ov=re.search(r"O-Voxel\s+([\d.na]+)\s+([\d.na]+)\s+([\d.na]+)",sc); ou=re.search(r"cyl\s+([\d.na]+)\s+([\d.na]+)\s+([\d.na]+)",sc)
    print(f"{o:12s} {sp:52s} {ou.group(1) if ou else '?':>7s} {ou.group(2) if ou else '?':>7s} {ou.group(3) if ou else '?':>7s}   {ov.group(1) if ov else '?'} / {ov.group(2) if ov else '?'} / {ov.group(3) if ov else '?'}")
PY
echo POST_DONE
