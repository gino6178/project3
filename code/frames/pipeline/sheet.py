# The review sheet for one object: every lengthwise and transverse face and every exterior, labelled.
#   python sheet.py <aiobj root> <object>      -> <root>/<object>/sheet.png
import sys,os
from PIL import Image,ImageDraw
root,o=sys.argv[1],sys.argv[2]; W=150; im=Image.new("RGB",(W*9,W*3),"white"); d=ImageDraw.Draw(im)
for i,fam in enumerate(("long","trans")):
    for j in range(9):
        p=f"{root}/{o}/raw/{fam}_{j}.png"
        if os.path.exists(p): im.paste(Image.open(p).convert("RGB").resize((W,W)),(j*W,i*W)); d.text((j*W+3,i*W+3),f"{fam}_{j}",fill=(255,0,0))
for j in range(4):
    p=f"{root}/{o}/exterior/ext_{j}.png"
    if os.path.exists(p): im.paste(Image.open(p).convert("RGB").resize((W,W)),(j*W,2*W)); d.text((j*W+3,2*W+3),f"ext {j}",fill=(255,0,0))
im.save(f"{root}/{o}/sheet.png"); print(f"{root}/{o}/sheet.png")
