# Figure 3 of v0.2: the method as one diagram, every panel a real slice.  Row one is what goes in -- the
# shape the carrier is fitted on (three sources), the cut faces (two sources), their framing, the two
# priors; row two is the cylindrical state, the one reverse chain on it and the cuttable asset.
#   python method_overview_v02.py <dir of the v0.1 figure's images> <Fruit3D_Fusion root> <fig3_parts.py output> <out.svg>
# The first directory holds the slices embedded in assets/method_overview.svg, extracted in order (00..18).
import sys,os,io,base64
from PIL import Image
IMG,ROOT,PARTS,OUT=sys.argv[1:5]
def b64(im,s=None,crop=True,fmt="JPEG"):
    im=im.convert("RGB")
    if crop: w,h=im.size; m=min(w,h); im=im.crop(((w-m)//2,(h-m)//2,(w+m)//2,(h+m)//2))
    if s: im=im.resize((s,s) if crop else s,Image.LANCZOS)
    b=io.BytesIO(); im.save(b,fmt,quality=88); return f"data:image/{fmt.lower()};base64,"+base64.b64encode(b.getvalue()).decode()
old=lambda i: Image.open(f"{IMG}/{i:02d}.png")
sw=Image.open(f"{ROOT}/v01/asset/swissroll.png"); fo=Image.open(f"{ROOT}/v02/fruit/out/s_orange.png")
I={k:b64(old(i),224) for k,i in dict(pl=0,pt=1,car=2,cl=3,ct=4,n=8,eL=10,comb=12,fin=13,a0=15,rl=16,rt=17,ro=18).items()}
I.update(strip=b64(old(5),(496,124),crop=False),nstrip=b64(old(9),(192,48),crop=False),eT=b64(old(11),(192,60),crop=False),
         fstrip=b64(old(14),(232,58),crop=False),slL=b64(old(6),(132,160),crop=False),slT=b64(old(7),(368,104),crop=False),
         gl=b64(Image.open(f"{ROOT}/aiobj/swissroll/raw/long_0.png"),224),gt=b64(Image.open(f"{ROOT}/aiobj/swissroll/raw/trans_0.png"),224),
         ext=b64(Image.open(f"{ROOT}/aiobj/swissroll/exterior/ext_3.png"),224),mesh=b64(sw.crop((0,0,512,512)),224),
         orange=b64(fo.crop((0,0,512,512)),224),state=b64(Image.open(f"{PARTS}/state.png"),400),
         lo=b64(Image.open(f"{PARTS}/ilvr_low.png"),160),hi=b64(Image.open(f"{PARTS}/ilvr_high.png"),160),sm=b64(Image.open(f"{PARTS}/ilvr_sum.png"),160))
BL,OR,GR,INK,MUT="#1f5fa8","#c8621a","#3d7a3a","#16181b","#4f535a"
W,H=1840,966; o=[]
def T(x,y,t,cls="b",anchor="start",fill=None): o.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}"'+(f' fill="{fill}"' if fill else "")+f'>{t}</text>')
def lines(x,y,ls,cls="m",dy=17,fill=None):
    for i,l in enumerate(ls): T(x,y+i*dy,l,cls,fill=fill)
def R(x,y,w,h,cls="card"): o.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}"/>')
def im(k,x,y,w,h,frame=None):
    o.append(f'<image href="{I[k]}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none"/>')
    if frame: o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="none" stroke="{frame}" stroke-width="2"/>')
def arrow(d,c="#3a3d42",w=2.2,dash=None):
    o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"'+(f' stroke-dasharray="{dash}"' if dash else "")+f' marker-end="url(#m{c[1:]})"/>')
def panel(n,x,y,w,h,title,sub,col=INK):
    R(x,y,w,h,"panel"); o.append(f'<circle cx="{x+22}" cy="{y+24}" r="13" fill="{col}"/>'); T(x+22,y+29,n,"num","middle")
    T(x+44,y+30,title,"h"); T(x+44,y+50,sub,"m")
# ---------------- row one ----------------
Y=96; HH=400
panel("0",30,Y,450,HH,"Shape","the solid the carrier is fitted on: three sources",GR)
cards=[(Y+66,"a shape you define","an equation (sphere, torus,","box), or a solid built whole"),
       (Y+168,"generated: TRELLIS.2","one exterior view &#8594;","a textured mesh, one seed"),
       (Y+270,"a real object","captured whole: its released","Gaussians (the paper&#8217;s fruit)")]
for yy,t1,t2,t3 in cards: R(44,yy,282,92); T(132,yy+24,t1,"b"); lines(132,yy+44,[t2,t3])
# a shape you define: drawn, not photographed
yy=Y+66; o.append(f'<g transform="translate(54,{yy+16})" fill="none" stroke="{GR}" stroke-width="1.8"><ellipse cx="34" cy="16" rx="30" ry="12"/><ellipse cx="34" cy="16" rx="11" ry="4"/><rect x="14" y="36" width="24" height="22" rx="1"/><path d="M14,36 l10,-7 l24,0 l-10,7 M38,58 l10,-7 l0,-22"/><circle cx="58" cy="48" r="9"/></g>')
yy=Y+168; im("ext",50,yy+10,36,36); arrow(f"M68,{yy+48} L68,{yy+54}",w=1.4); im("mesh",72,yy+40,48,48)
yy=Y+270; im("orange",52,yy+12,72,68)
for yy in (Y+112,Y+214,Y+316): arrow(f"M328,{yy} L346,{Y+214}",c="#7a7f86",w=1.6)
R(350,Y+120,118,190); T(362,Y+144,"the carrier","b"); im("car",370,Y+156,78,78)
lines(362,Y+254,["occupancy lattice,","fitted to the","cut faces (&#167;3.1)"],dy=16)
T(409,Y+352,"constrains &#9316;","bl","middle",fill=GR); T(409,Y+370,"never trained on","s","middle")
panel("1",500,Y,350,HH,"Cut faces","2&#8211;3 per family, different specimens",INK)
T(516,Y+84,"real photographs","b"); im("pl",516,Y+94,104,104,BL); im("pt",630,Y+94,104,104,OR)
T(516,Y+230,"generated: Gemini 2.5 Flash Image","b"); im("gl",516,Y+240,104,104,BL); im("gt",630,Y+240,104,104,OR)
T(746,Y+150,"longitudinal","bl",fill=BL); T(746,Y+170,"transverse","bl",fill=OR)
lines(746,Y+282,["one request","per image"])
panel("2",870,Y,360,HH,"Canonicalise","every slice framed the same way",INK)
im("cl",890,Y+70,140,140,BL); im("ct",1070,Y+70,140,140,OR)
lines(890,Y+232,["canonical frame, 256&#178;, object fills 82%"])
arrow(f"M1140,{Y+214} L1140,{Y+262}",c=OR); T(1150,Y+246,"unwrap to (r, &#966;)","bl",fill=OR)
im("strip",890,Y+272,320,80,OR)
lines(890,Y+374,["the membranes become parallel stripes, &#966; wraps"])
panel("3",1250,Y,270,HH,"Two priors","single-image DDPM, cut faces only",INK)
for k,(yy,c,cl,t,sub) in enumerate([(Y+74,BL,"lf","&#949;<tspan baseline-shift='sub' font-size='11'>L</tspan>  longitudinal","3 cut faces &#183; 30k steps"),
                                    (Y+226,OR,"tf","&#949;<tspan baseline-shift='sub' font-size='11'>T</tspan>  transverse, polar","3 strips &#183; 4k steps")]):
    R(1266,yy,238,136,cl); T(1282,yy+28,t,"b")
    for j,(bw,bh) in enumerate([(14,40),(14,26),(14,16),(14,26),(14,40)]): o.append(f'<rect x="{1290+j*26}" y="{yy+46+(40-bh)/2}" width="{bw}" height="{bh}" rx="2" fill="{c}"/>')
    T(1282,yy+118,sub,"m")
panel("4",1540,Y,270,HH,"One cylindrical state","A[r, &#966;, z]: both families are exact slices",INK)
im("state",1555,Y+62,240,240)
T(1675,Y+336,"longitudinal: two &#966; columns &#8594; (s, z)","bl","middle",fill=BL)
T(1675,Y+356,"transverse: one z &#8594; the (r, &#966;) strip","bl","middle",fill=OR)
T(1675,Y+382,"the carrier, sampled in: start and mask M","m","middle")
for x0,x1 in ((480,500),(850,870),(1230,1250),(1520,1540)): arrow(f"M{x0+1},{Y+HH/2} L{x1-3},{Y+HH/2}")
# ---------------- row two ----------------
Y2=Y+HH+30; H2=424
panel("5",30,Y2,1230,H2,"One reverse chain on that state","from t&#8320; = 0.3T to 0, 100 DDIM steps; nothing is interpolated",INK)
cx=[50,250,490,750,1070]; cw=[180,220,240,300,170]; cy=Y2+70; ch=300
heads=["start: noised carrier","each family denoises","combine, by radius","layout from the carrier","the interior, t = 0"]
for x,w,hd in zip(cx,cw,heads): R(x,cy,w,ch); T(x+14,cy+26,hd,"b")
im("n",66,cy+40,148,148); im("nstrip",66,cy+196,148,38); lines(66,cy+258,["interior only;","outside M, the same noise"])
im("eL",266,cy+40,148,148,BL); T(266,cy+206,"&#949;<tspan baseline-shift='sub' font-size='10'>L</tspan> on 90 planes","bl",fill=BL)
im("eT",266,cy+220,198,52,OR); T(266,cy+290,"&#949;<tspan baseline-shift='sub' font-size='10'>T</tspan> on each height z (250)","bl",fill=OR)
im("comb",506,cy+40,110,110)
o.append(f'<g transform="translate(626,{cy+46})"><path d="M0,0 L0,110 L104,110" fill="none" stroke="#9aa0a6" stroke-width="1.2"/><path d="M2,6 C30,6 40,10 52,60 C62,100 74,104 102,104" fill="none" stroke="{INK}" stroke-width="2.2"/></g>')
T(506,cy+180,"w(r): longitudinal leads","bl",fill=BL); T(506,cy+198,"at the axis, transverse outside","bl",fill=OR)
lines(506,cy+234,["x&#770;&#8320; = w(r) x&#770;&#8320;&#7480; + (1&#8722;w(r)) x&#770;&#8320;&#7488;","the columella: a strip to one family,","a disc to the other"])
x5=764
im("lo",x5,cy+44,78,78); T(x5+84,cy+90,"+","h"); im("hi",x5+100,cy+44,78,78); T(x5+184,cy+90,"=","h"); im("sm",x5+200,cy+44,78,78)
for k,(l1,l2) in enumerate([("the carrier&#8217;s","low-pass"),("the priors&#8217;","detail"),("this step&#8217;s","state")]):
    T(x5+39+k*100,cy+142,l1,"s","middle"); T(x5+39+k*100,cy+158,l2,"bl","middle",fill=GR if k==0 else (INK if k==2 else BL))
lines(x5,cy+192,["each step keeps the carrier&#8217;s low","frequencies (ILVR, &#963; = 16 px) and the","priors&#8217; high ones; the shell is never","written. Re-noise to t&#8242; (DDIM), repeat."],cls="m",dy=18)
im("fin",1080,cy+40,150,150); im("fstrip",1080,cy+200,150,38); lines(1082,cy+262,["both families","agree on it"])
for x0,x1 in ((230,250),(470,490),(730,750),(1050,1070)): arrow(f"M{x0+1},{cy+ch/2} L{x1-3},{cy+ch/2}")
arrow(f"M{900},{cy+ch+2} L900,{cy+ch+26} L140,{cy+ch+26} L140,{cy+ch+4}",c="#7a7f86",w=1.8,dash="6 4")
T(528,cy+ch+20,"&#215; 100 steps","m","middle")
panel("6",1280,Y2,530,H2,"A cuttable asset","baked once, read at any angle",INK)
R(1296,Y2+70,240,150); T(1310,Y2+96,"write back, once","b"); im("a0",1310,Y2+108,96,96)
lines(1416,Y2+124,["interior cells &#8592; A","skin keeps its colour","682,100 rewritten","480,287 kept"],dy=18)
R(1548,Y2+70,246,150); T(1562,Y2+96,"closed-form planar cut","b"); lines(1562,Y2+120,["plane &#8745; cell edges,","exact roots","8.6&#8211;16.5 ms","0% unpainted"],dy=20)
T(1296,Y2+250,"read at any plane, supervised or not","b")
for j,(k,c,l) in enumerate([("rl",BL,"longitudinal"),("rt",OR,"transverse"),("ro",INK,"oblique")]): im(k,1296+j*104,Y2+262,94,94,c); T(1343+j*104,Y2+376,l,"bl","middle",fill=c)
R(1612,Y2+244,182,166); T(1624,Y2+266,"held-out DreamSim &#8595;","b")
for j,(l,v,c) in enumerate([("real floor",0.055,"#9aa0a6"),("FruitNinja",0.154,"#c4c7cc"),("carrier",0.113,"#c4c7cc"),("ours",0.087,BL)]):
    yy=Y2+278+j*30; T(1624,yy+12,l,"s"); o.append(f'<rect x="1624" y="{yy+18}" width="{v*700:.0f}" height="8" fill="{c}"/>'); T(1624+v*700+5,yy+26,f"{v:.3f}","s")
arrow(f"M1260,{Y2+H2/2} L1277,{Y2+H2/2}")
# the carrier, into the chain
for x,w in ((50,180),(750,300)): R(x+w-92,cy-10,84,22,"badge"); T(x+w-50,cy+6,"carrier &#9450;","bdg","middle")
# row one into row two
arrow(f"M1675,{Y+HH} L1675,{Y+HH+14} L1600,{Y+HH+14}",c="#3a3d42",w=0)  # spacer (no stroke)
mk="".join(f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in (INK,"#3a3d42","#7a7f86",OR,BL))
css=f'''text{{font-family:"Helvetica Neue",Arial,"Segoe UI",sans-serif}} .h{{font-size:18px;font-weight:700;fill:{INK}}} .b{{font-size:14.5px;font-weight:600;fill:{INK}}}
.t{{font-size:14px;fill:{INK}}} .m{{font-size:13px;fill:{MUT}}} .s{{font-size:12px;fill:{MUT}}} .bl{{font-size:13px;font-weight:600}} .num{{font-size:14px;font-weight:700;fill:#fff}}
.ttl{{font-size:26px;font-weight:700;fill:{INK}}} .sub{{font-size:15px;fill:{MUT}}}
.panel{{fill:#f8f9fa;stroke:#d5d9de;stroke-width:1.2;rx:12}} .card{{fill:#fff;stroke:#cdd1d6;stroke-width:1.2;rx:8}}
.badge{{fill:#e8f2e6;stroke:#3d7a3a;stroke-width:1.2;rx:11}} .bdg{{font-size:12px;font-weight:700;fill:#3d7a3a}}
.lf{{fill:#eef4fb;stroke:{BL};stroke-width:1.4;rx:8}} .tf{{fill:#fdf2e9;stroke:{OR};stroke-width:1.4;rx:8}}'''
head=f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" role="img" aria-label="Method overview: the shape the carrier is fitted on from three sources (a shape you define, a TRELLIS.2 mesh, a captured real object), cut faces from two sources (real photographs, Gemini), canonical framing, two single-image priors, one cylindrical state, one reverse chain and the cuttable asset">'
title=f'<text class="ttl" x="30" y="44">Interiors from two cut families</text><text class="sub" x="30" y="72">A shape and two or three cut faces per family, of different specimens &#8594; a product of two single-image diffusion priors over the slices of one cylindrical lattice. Every image is a real slice.</text>'
o=[x for x in o if 'stroke-width="0"' not in x]
open(OUT,"w").write(head+f"<defs>{mk}</defs><style>{css}</style><rect width='{W}' height='{H}' fill='#fff'/>"+title+"\n".join(o)+"</svg>")
print(OUT)
