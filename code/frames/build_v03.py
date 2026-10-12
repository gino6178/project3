# v0.3: the v0.2 page reorganised as a paper -- a main text of six sections and Appendices A-E -- with the
# text moved, not rewritten, except for the joins.  Figures, tables, equations and section references are
# renumbered by order of appearance.
import re
SRC="/home/gino/project/project3-site/3dfusion_v0.2.html"; DST="/home/gino/project/project3-site/3dfusion_v0.3.html"
s=open(SRC).read()
# ---- placeholders for every number that will move
s=re.sub(r'<span class="fnum">Figure (\d+)</span>',lambda m:f'<span class="fnum">Figure {{{{F:{m.group(1)}}}}}</span>',s)
s=re.sub(r'Figure&nbsp;(\d+)',lambda m:f'Figure&nbsp;{{{{F:{m.group(1)}}}}}',s)
s=re.sub(r'<p class="tnum">Table (\d+)</p>',lambda m:f'<p class="tnum">Table {{{{T:{m.group(1)}}}}}</p>',s)
s=re.sub(r'Table&nbsp;(\d+)',lambda m:f'Table&nbsp;{{{{T:{m.group(1)}}}}}',s)
s=re.sub(r'<div class="num">\((\d)\)</div>',lambda m:f'<div class="num">({{{{E:{m.group(1)}}}}})</div>',s)
for a,b in [("equation&nbsp;(6)","equation&nbsp;({{E:6}})"),("equation&nbsp;(1)","equation&nbsp;({{E:1}})"),("// equation (6)","// equation ({{E:6}})"),
            ("radius rule (6)","radius rule ({{E:6}})"),("make (5) computable","make ({{E:5}}) computable"),("approximation in (5)","approximation in ({{E:5}})"),
            ("what (4) asks for","what ({{E:4}}) asks for")]:
    assert a in s,a; s=s.replace(a,b)
def secref(m):
    word,body=m.group(1),m.group(2)
    return word+"&nbsp;"+re.sub(r"\d\.\d+",lambda k:"{{S:"+k.group(0)+"}}",body)
s=re.sub(r"(sections?)&nbsp;(\d\.\d+(?:(?:&ndash;|, | and |&nbsp;and&nbsp;)\d\.\d+)*)",secref,s)
L=s.splitlines()
def g(a,b): return "\n".join(L[a-1:b])
def one(i): return L[i-1]
def edit(t,a,b):
    assert a in t,(a[:80]); return t.replace(a,b,1)
# ---- the head: title, version, contents
head=g(1,111)
head=edit(head,"<title>3DFusion v0.2:","<title>3DFusion v0.3:")
head=edit(head,'<p class="venue">Preprint &middot; v0.2 draft</p>','<p class="venue">Preprint &middot; v0.3 draft</p>')
head=edit(head,'<a href="3dfusion_v0.1.html">Version 0.1</a>','<a href="3dfusion_v0.1.html">Version 0.1</a>\n    <a href="3dfusion_v0.2.html">Version 0.2 (the long form)</a>')
PRINT='''  /* print: a paper -- A4, 9.5pt, the main text in two columns with figures and tables across both */
  @page{size:A4;margin:14mm 13mm 16mm}
  @media print{ html,body{font-size:9.5pt;line-height:1.38} .wrap{max-width:none;padding:0} nav.toc,.links,footer{display:none}
    header h1{font-size:17pt;margin:0 0 4px} header .sub{font-size:9.5pt} .lead{font-size:9.5pt}
    h2{font-size:12pt;margin:12px 0 4px} h3{font-size:10.5pt;margin:9px 0 3px} p,li{margin:0 0 5px} ul{margin:4px 0 6px}
    .paper{column-count:2;column-gap:7mm;column-fill:auto}
    .paper figure,.paper .tw,.paper .teaser,.paper .eq,.paper div[style*="monospace"]{column-span:all}
    figure{margin:8px 0 10px} figcaption{font-size:8pt;line-height:1.3} p.mut,.tw p.mut{font-size:8pt}
    figure,.tw,table,.eq,div[style*="monospace"]{break-inside:avoid} figcaption,p.mut{break-before:avoid} h2,h3,p.sub-t{break-after:avoid}
    .tw{overflow:visible;margin:8px 0} table{font-size:7.6pt;width:100%} td,th{padding:2px 4px;overflow-wrap:anywhere}
    figure img{max-height:100mm;width:auto;max-width:100%;margin:0 auto;object-fit:contain}
    .teaser{grid-template-columns:2.2fr 1fr!important} .teaser img{max-height:62mm;width:auto;max-width:100%;margin:0 auto}
    .cutstrip{display:none} h2.appendix{break-before:page} .appendix-all{font-size:9pt} }
  .teaser{display:grid;grid-template-columns:1.9fr 1fr;gap:14px;align-items:center}
  @media(max-width:760px){.teaser{grid-template-columns:1fr}}
  p.sub-t{font-weight:600;margin:16px 0 6px}
  h2.appendix{border-top:2px solid #ddd;padding-top:22px;margin-top:56px}
</style>'''
head=edit(head,"</style>",PRINT)
toc=('<nav class="toc">\n  <a href="#abstract">Abstract</a><a href="#intro">1 Introduction</a><a href="#related">2 Related work</a><a href="#method">3 Method</a>'
     '<a href="#results">4 Experiments</a><a href="#limits">5 Limitations</a><a href="#conclusion">6 Conclusion</a><a href="#appendix">Appendices</a><a href="#refs">References</a>\n</nav>\n')
# ---- Figure 1: the teaser, the old Figures 1 and 2 side by side
cap1=re.search(r'<strong>The asset read at every angle.</strong>(.*?)</figcaption>',one(119)).group(1)
cap2=re.search(r'<strong>One method, a canonical coordinate per structure.</strong>(.*?)</figcaption>',one(124)).group(1)
teaser=f'''<figure style="margin:14px 0 26px">
  <div class="teaser"><img src="assets/v01_overview.png" alt="three objects and three canonical coordinate systems: a layer cake in Cartesian coordinates, a tree trunk in cylindrical coordinates whose rings unwrap to parallel stripes, a banana straightened along its declared centreline"><img src="assets/v02_orange.gif" alt="the lifted orange: turned whole, then read by a longitudinal plane turning about the axis, a transverse plane sweeping and an oblique plane tilting"></div>
  <figcaption><span class="fnum">Figure {{{{F:1}}}}</span><br><strong>A canonical coordinate per structure, and the asset it yields.</strong> <em>Left:</em>{cap2} <em>Right:</em>{cap1}</figcaption>
</figure>
'''
abstract=g(127,140)
intro=g(141,198)
# ---- related work: three subsections; dual-grid surface extraction goes to Appendix A
related=g(199,206)+"\n"+edit(g(207,210),"2.2&ensp;","2.2&ensp;")+"\n"+edit(g(215,218),"<h3>2.4&ensp;","<h3>2.3&ensp;")
# ---- method
m_head=one(219)
m_lead='''<p class="lead">Figure&nbsp;{{F:3}} is the whole of it. The carrier is fitted first (section&nbsp;3.4; Appendix&nbsp;A). Its interior is then sampled anew from a product of two single-image diffusion priors on a canonical lattice, with the carrier&#8217;s layout and shell as constraints on the sampling and not as training data (sections&nbsp;3.1&ndash;3.3), and written back into the carrier&#8217;s cells (section&nbsp;3.4). The cut operator and the solver interface are the carrier&#8217;s and are not touched.</p>'''
fig3=g(223,226)
s31="<h3>3.1&ensp;Overview and problem formulation</h3>\n\n"+g(285,300)
s32="<h3>3.2&ensp;Canonical coordinate systems</h3>\n\n"+one(303)+"\n\n"+one(335)
s33=("<h3>3.3&ensp;Slice diffusion as a product of experts</h3>\n\n"+g(310,333)+"\n\n"
     +'<p>Each family&#8217;s prior is a small single-image DDPM trained on that family&#8217;s two or three canonical photographs alone, the longitudinal one for 30k steps and the polar one for 4k; its architecture, the training recipe and the notation are in Appendix&nbsp;B.</p>')
s34=("<h3>3.4&ensp;The carrier: a decoupled discrete lattice</h3>\n\n"+g(252,256)+"\n\n"
     +'<p>It is fitted to the unposed cut photographs without poses, and a plane cuts it in closed form at machine precision; both are the carrier&#8217;s own and are given in Appendix&nbsp;A.</p>\n\n'+one(343))
# ---- experiments
e_head=one(349)+"\n\n"+one(351)
t4=g(436,447)
p367=one(367)
p367=edit(p367,"<p>Table&nbsp;{{T:1}} is the comparison. The cylinder is below the carrier on both faces and the mean,","<p>On the orange, where the real floor between photographs is 0.055, the cylinder is below the carrier on both faces and the mean (0.087 against 0.113),")
p367=edit(p367,"Every Cartesian arm is above the carrier.","Every Cartesian arm is above the carrier (Table&nbsp;{{T:2}}a).")
p449=one(449)
i=p449.index(" Both losses are on objects whos"); j=p449.index(" The apple and the orange, where the carrier"); k=p449.index(" Six of seven are below the carrier")
p449a=p449[:i]+p449[j:k]+" Six of seven are below the carrier; the watermelon is above it, and the two losses and the watermelon are analysed together in section&nbsp;4.5.</p>"
s41=("<h3>4.1&ensp;Main results: real objects and baselines</h3>\n\n"+one(429)+"\n\n"+t4+"\n\n"+p449a+"\n\n"+p367+"\n\n"+g(451,457)+"\n\n"+one(461))
# Table 2: the benchmark, one table -- (a) DreamSim and volume L1 per object, (b) the growth field
def rows(html):
    out={}
    for r in re.findall(r"<tr>(.*?)</tr>",html[html.index("<tbody>"):]):
        c=[re.sub("<[^>]+>","",x) for x in re.findall(r"<td>(.*?)</td>",r)]; out[c[0].split()[0]]=c[1:]
    return out
t7=g(503,505); t8=g(521,523); d7=rows(t7); d8=rows(t8)
names={"wood1":"wood1 concentric","wood2":"wood2 perturbed","wood2k":"wood2k knotted","wood3":"wood3 bent by shear","wood3r":"wood3r arc-bent","wood5":"wood5 strongly bent, sliced normal","wood4":"wood4 curved and knotted","Swiss":"Swiss roll","honeycomb":"honeycomb","maki":"maki roll","layer":"layer cake"}
alias8={"Swiss":"Swiss","maki":"maki","layer":"layer"}
tb=""
for k,nm in names.items():
    a=d7[k]; b=d8[alias8.get(k,k)] if alias8.get(k,k) in d8 else d8.get(k)
    av=[None if "mdash" in x else float(x) for x in a]; bv=[None if "mdash" in x else float(x) for x in b]
    ba=min(x for x in av if x is not None); bb=min(x for x in bv if x is not None); cells=[]
    for x,y in zip(av,bv):
        if x is None: cells.append("<td>&mdash;</td>"); continue
        sx=f"<strong>{x:.3f}</strong>" if x==ba else f"{x:.3f}"; sy=f"<strong>{y:.4f}</strong>" if y==bb else f"{y:.4f}"
        cells.append(f"<td>{sx} / {sy}</td>")
    tb+=f"<tr><td>{nm}</td>"+"".join(cells)+"</tr>"
t9=g(549,552)
t9b=t9   # the growth-field table, in Appendix D
t2=('<div class="tw"><p class="tnum">Table {{T:7}}</p><p class="sub-t">DreamSim to the truth&#8217;s faces / volume L1</p><table>\n<thead><tr><th>object</th><th>carrier</th><th>cylinder</th><th>Cartesian</th><th>curvilinear</th></tr></thead>\n<tbody>'
    +tb+'</tbody></table></div>\n<p class="mut" style="margin-top:-8px">DreamSim: the mean over the unseen heights, the unseen azimuths and the oblique planes. Volume L1: over the interior, colour in [0,1]. Both lower better. Every column is the paper&#8217;s carrier and priors; only the lattice differs; curvilinear is the parallel-transported frame, run where a centreline is declared. Every face kind, with orientation error, PSNR, SSIM and LPIPS, is in Appendix&nbsp;D.</p>\n')
p519=edit(one(519),"<p>Table&nbsp;{{T:8}} is the whole-volume view.","<p>Table&nbsp;{{T:7}} is also the whole-volume view, with the per-face numbers in Appendix&nbsp;D.")
p534=edit(one(534),"<p>","<p><strong>The coordinate system follows the structure.</strong> ")
p547=edit(one(547),"<p>","<p><strong>Structure, read from the growth field.</strong> ")
p534=p534.replace("</p>"," The strongly bent wood5, sliced normal to its axis, is the case the parallel-transported frame exists for, and there it is the best lattice on every face kind (Appendix&nbsp;D).</p>")
p554=one(554); cut=p554.index(" Ring continuity,"); p554a=p554[:cut]+"</p>"; p554b="<p>"+p554[cut+1:]
s42=("<h3 id=\"gtbench\">4.2&ensp;A benchmark with full 3-D truth</h3>\n\n"+one(499)+"\n\n"+one(501)+"\n\n"+t2+"\n\n"+g(514,517)+"\n\n"+p519+"\n\n"+p534+"\n\n"+p547+"\n\n"+p554a)
# Table 3: ablations -- (a) the sampler's design and settings on the orange, (b) how many photographs
t2old=g(378,391); t3old=g(394,403); t1old=g(355,366)
extra=""
for r in re.findall(r"<tr>.*?</tr>",t1old,re.S):
    if any(k in r for k in ("separate chains","re-sliced every step","intersection sync")): extra+=r+"\n"
t2a=t2old.replace('<p class="tnum">Table {{T:2}}</p>','<p class="tnum">Table {{T:2}}</p><p class="sub-t">(a) the sampler, on the orange</p>')
t2a=t2a.replace("</tbody></table></div>","</tbody></table></div>",1)
# the three design rows of the old Table 1 (mean only where it was measured) go under the Cartesian rows
def widen(r):
    c=re.findall(r"<td[^>]*>(.*?)</td>",r,re.S)
    if len(c)==4: return f"<tr><td>{c[0]}</td><td>{c[1]}</td><td>{c[2]}</td><td>{c[3]}</td><td>a design that did not survive</td></tr>\n"
    return f"<tr><td>{c[0]}</td><td colspan=\"4\">{c[1]}</td></tr>\n"
t2a=edit(t2a,"</tbody></table></div>","".join(widen(r) for r in re.findall(r"<tr>.*?</tr>",extra,re.S))+"</tbody></table></div>")
t3b=t3old.replace('<p class="tnum">Table {{T:3}}</p>','<p class="sub-t">(b) how many photographs per family</p>')
p392=one(392).replace("Table&nbsp;{{T:3}}","Table&nbsp;{{T:2}}b"); p376=one(376).replace("Table&nbsp;{{T:2}}","Table&nbsp;{{T:2}}a")
p404=one(404); p411=one(411)
diag=('''<figure style="margin:16px 0 26px">
  <img src="assets/v03_diag.jpg" style="width:100%;border-radius:4px" alt="three diagnostics on the orange: the longitudinal prior alone restoring a carrier slice from 30, 50 and 90% noise; the polar prior overtrained; the copier test">
  <figcaption><span class="fnum">Figure {{F:8}}</span><br><strong>Three diagnostics, on the orange.</strong> <strong>(a)</strong> The longitudinal prior alone, in-plane: a carrier slice restored from 30%, 50% and 90% noise at eight and thirty thousand steps; at 30&ndash;50% the longer prior draws segment walls and a crisp columella, at 90% both collapse &mdash; a patch prior has no global layout, and the carrier is where the layout comes from. <strong>(b)</strong> The polar prior overtrained: at 30k the membranes are fixed in number and evenly spaced, the three training strips reproduced. <strong>(c)</strong> The copier test: cross-conditioned priors fed the slice itself reproduce it, fed zeros collapse to noise &mdash; none of them had learned a prior (Appendix&nbsp;C).</figcaption>
</figure>''')
p558=one(558).replace("</p>"," Inputs from six trunks keep the ordering of the same-specimen benchmark: against a specimen-to-specimen floor of 0.008 / 0.013 / 0.007 (unseen height / azimuth / oblique) the carrier scores 0.113 / 0.129 / 0.136, the cylinder 0.071 / 0.128 / 0.103 and the Cartesian lattice 0.100 / 0.113 / 0.126 (Table&nbsp;{{T:10}}, Figure&nbsp;{{F:15}}).</p>")
p625=one(625).replace("<p>","<p><strong>The carrier is a data term.</strong> ",1); RF=" <em>Re-fitting it.</em>"; cut=p625.index(RF)
p625b="<p><strong>Re-fitting the carrier.</strong>"+p625[cut+len(RF):]; p625=p625[:cut]+"</p>"
s43=("<h3>4.3&ensp;Diagnostics and ablations</h3>\n\n"+p376+"\n\n"+t2a+"\n"+t3b+"\n\n"+p392+"\n\n"+p404+"\n\n"+p411+"\n\n"+diag+"\n\n"+p558+"\n\n"+p625)
# 4.4: the asset and its cost; the generated route
fruit=('''<figure style="margin:20px 0 22px">
  <img src="assets/v02_fruit7.jpg" loading="lazy" style="width:100%;max-width:920px;margin:0 auto;display:block;border-radius:4px" alt="the seven objects as finished assets: exterior, lengthwise cut, cross-section and oblique cut">
  <figcaption><span class="fnum">Figure {{F:fruit}}</span><br><strong>The seven objects as finished assets.</strong> '''+re.sub(r"<[^>]*>","",one(483)).replace("and below each turning and then cut as in Figure&nbsp;{{F:1}}","each turning and then cut in Appendix&nbsp;D").strip()+'''</figcaption>
</figure>''')
p465=one(465)
p578=one(578)
p583=one(583)
p585=one(585).replace("</p>"," The lattices score, lengthwise / across / mean against a floor of 0.080 / 0.043 / 0.061: the carrier 0.154 / 0.253 / 0.204, the cylinder 0.156 / 0.271 / 0.214, the Cartesian lattice 0.140 / 0.260 / 0.200, the translation-only curvilinear lattice 0.148 / 0.241 / 0.194 and the parallel-transported one 0.150 / 0.249 / 0.199 (Table&nbsp;{{T:11}}).</p>")
eleven='''<p><strong>Eleven objects of varied shape.</strong> The same path, with one setting for all, runs on eleven further generated objects chosen for their shape as much as their interior &mdash; rolls, layered and radial vegetables, a geode, a terrazzo block, a cheese wedge, a bagel, an ice-cream cone and a custard pudding (Appendix&nbsp;D: Table&nbsp;{{T:13}}, Figures&nbsp;{{F:19}}&ndash;{{F:20}}). The declared lattice improves on the carrier on nine of the eleven, and on the mean from 0.291 to 0.268; the two losses, the carrot and the ice-cream cone, are an interior with little layout to add and a family whose generated slices disagree with each other (floor 0.252).</p>'''
s44=("<h3>4.4&ensp;End to end: the asset, its cost, and generated inputs</h3>\n\n"+p465+"\n\n"+g(467,477)+"\n\n"+fruit+"\n\n"+'<p><strong>From generated inputs.</strong> The same program runs with no object and no photographer, from the generated source of Figure&nbsp;{{F:3}}: a short oak log generated as images and lifted to an asset (Appendix&nbsp;D), a banana on the curvilinear lattice, and eleven objects of varied shape.</p>'+"\n\n"+p585+"\n\n"+g(598,601)+"\n\n"+one(603)+"\n\n"+eleven)
# 4.5: failures, gathered
k=p449.index(" Both losses are on objects whos"); j=p449.index(" The apple and the orange, where the carrier")
p449loss=p449[k:j].strip(); k2=p449.index(" Six of seven are below the carrier"); p449wm=p449[k2:].replace("</p>","").strip()
lim=lambda i: re.sub(r"^\s*<li>","",one(i)).replace("</li>","")
s45=("<h3>4.5&ensp;Discussion and failure analysis</h3>\n\n"
     "<p>Every loss in this paper falls into one of three classes, and each class is a property of the method rather than of one object.</p>\n\n"
     "<p><strong>Discrete compact structures are smeared.</strong> "+re.sub(r"^<strong>.*?</strong>\s*","",lim(672))+" "+p449wm+"</p>\n\n"
     "<p><strong>Thin, low-contrast structure is smoothed away.</strong> "+re.sub(r"^<p><strong>.*?</strong>\s*","",one(605)).replace("</p>","")+" Lengthwise / across / mean against a floor of 0.036 / 0.120 / 0.078, the carrier scores 0.220 / 0.259 / 0.240, the cylinder 0.267 / 0.260 / 0.264 and the Cartesian lattice 0.255 / 0.331 / 0.293 (Table&nbsp;{{T:12}}).</p>\n\n"
     +g(616,619)+"\n\n"+one(621)+"\n\n"
     "<p><strong>Where the carrier fails, the lift inherits it.</strong> "+p449loss+" "+re.sub(r"^<strong>.*?</strong>\s*","",lim(674))+" "+re.sub(r"^<strong>.*?</strong>\s*","",lim(683))+"</p>")
# 5: four limitations; the rest go to Appendix E
s5=('<h2 id="limits">5&ensp;Limitations and future work</h2>\n\n<ul>\n'
    '  <li><strong>Anisotropic smearing of discrete compact structures.</strong> The bias that makes membranes stationary makes seeds non-stationary, and a patch prior adds texture under a layout without inventing one. A prior that represents compact structures as objects, or a third family of cuts that sees them end-on, is the open extension (section&nbsp;4.5).</li>\n'
    '  <li><strong>Dependence on the carrier&#8217;s macro-layout.</strong> The lift sharpens and textures the carrier&#8217;s layout and does not correct it: where the carrier fits badly (the bread, the cake, the bent wood3r) or has not caught a layout (the honeycomb), the asset inherits it, and where the carrier is already near the floor (the watermelon) the prior has little to add. Its rings follow the family&#8217;s distribution, not one specimen&#8217;s placement (Table&nbsp;{{T:9}}).</li>\n'
    '  <li><strong>The coordinate system and its axis are declared.</strong> The polar axis, the canonical lattice and, for a curved object, the centreline and its frame are chosen by the user from the object&#8217;s organisation and how it was cut; nothing infers them, and the canonical frame assumes an object that fills its bounding square.</li>\n'
    '  <li><strong>Low-contrast micro-structure is lost.</strong> A patch prior trained on two or three photographs under-weights a thin line of nearly the ground&#8217;s colour, and the lift scores below the carrier it started from (the onion).</li>\n'
    '</ul>\n<p>Appendix&nbsp;E keeps the full list of narrower observations &mdash; one-family objects, the resampling floor of the write and read, the cylinder&#8217;s longitudinal bands, which curvilinear frame wins, the carrier&#8217;s configuration and the protocol.</p>')
s6=('<h2 id="conclusion">6&ensp;Conclusion</h2>\n\n'+one(663)+"\n\n"+one(665)+"\n\n"+one(667))
refs=g(688,848)
# ---- appendices
apA=('<h2 class="appendix" id="appendix">Appendix A&ensp;The carrier lattice and the closed-form planar cut</h2>\n\n'
     +edit(g(211,214),"<h3>2.3&ensp;Surface extraction and dual grid encodings</h3>","<h3>A.1&ensp;Surface extraction and dual grid encodings</h3>")+"\n\n<h3>A.2&ensp;Pose-free supervision and the closed-form cut</h3>\n\n"+g(258,281)+"\n\n"+one(345))
apB=('<h2 class="appendix">Appendix B&ensp;Notation, priors and the pipeline&#8217;s arrays</h2>\n\n<h3>B.1&ensp;Notation</h3>\n\n'+g(235,248)+"\n\n<h3>B.2&ensp;The priors</h3>\n\n"+one(339)+"\n\n<h3>B.3&ensp;The pipeline as arrays</h3>\n\n"+g(228,231)+"\n\n"+g(305,308))
apC=('<h2 class="appendix">Appendix C&ensp;Diagnostics: what went wrong on the way</h2>\n\n'+g(369,372)+"\n\n"+one(418)+"\n\n"+one(425)+"\n\n"+p625b)
apD=('<h2 class="appendix">Appendix D&ensp;Full results</h2>\n\n<h3>D.1&ensp;Every real object, both families</h3>\n\n'+g(431,434)+"\n\n"+one(482).replace('<div class="cutstrip">','<div class="cutstrip" style="margin-top:12px">')+'\n<p class="mut" style="margin-top:-14px">The seven assets of Figure&nbsp;{{F:fruit}}, turning and then cut as in Figure&nbsp;{{F:1}}.</p>'
     +"\n\n<h3>D.2&ensp;The benchmark, every face kind</h3>\n\n"
     +re.sub(r"</?details[^>]*>|<summary[^>]*>.*?</summary>","",g(507,512),flags=re.S).replace('<div class="tw"><table>','<div class="tw"><p class="tnum">Table {{T:7full}}</p><table>',1)+"\n"
     +re.sub(r"</?details[^>]*>|<summary[^>]*>.*?</summary>","",g(525,530),flags=re.S).replace('<div class="tw"><table>','<div class="tw"><p class="tnum">Table {{T:8full}}</p><table>',1)+"\n\n"
     +"<h3>D.3&ensp;The strongly bent trunk, the longitudinal bands, the growth field</h3>\n\n"+one(536)+"\n\n"+g(538,541)+"\n\n"+one(543)+"\n\n"+t9b+"\n\n"+p554b
     +"\n\n<h3>D.4&ensp;Cross-specimen inputs</h3>\n\n"+g(560,572)+"\n\n<h3>D.5&ensp;Generated objects</h3>\n\n"+p578+"\n\n"+g(580,581)+"\n\n"+p583+"\n\n"+g(587,596)+"\n\n"+one(629)+"\n\n"+g(631,647)+"\n\n"+one(649)+"\n\n"+g(651,657))
apE=('<h2 class="appendix">Appendix E&ensp;Extended limitations and failures</h2>\n\n'+g(607,614)+"\n\n<ul>\n"+g(672,685)+"\n</ul>")
foot=g(849,855)
doc="\n\n".join([head,toc,teaser,'<div class="paper">',abstract,intro,related,m_head,m_lead,fig3,s31,s32,s33,s34,e_head,s41,s42,s43,s44,s45,s5,s6,"</div>",apA,apB,apC,apD,apE,refs,foot])
# ---- numbering, by order of appearance
def number(doc,kind,label_re,merge):
    order=[]
    for m in re.finditer(label_re,doc):
        k=merge.get(m.group(1),m.group(1))
        if k not in order: order.append(k)
    num={k:i+1 for i,k in enumerate(order)}
    def rep(m):
        k=merge.get(m.group(1),m.group(1))
        assert k in num,(kind,m.group(1))
        return str(num[k])
    return re.sub(r"\{\{"+kind+r":([\w]+)\}\}",rep,doc),num
doc,fn=number(doc,"F",r'<span class="fnum">Figure \{\{F:(\w+)\}\}',{"2":"1","9":"8","10":"8"})
doc,tn=number(doc,"T",r'<p class="tnum">Table \{\{T:(\w+)\}\}',{"8":"7","1":"4","3":"2"})
doc,en=number(doc,"E",r'<div class="num">\(\{\{E:(\d)\}\}\)',{})
SMAP={"1.1":"1.1","1.2":"1.2","1.3":"1.3","1.4":"1.4","1.5":"1.5","2.1":"2.1","2.2":"2.2","2.3":"A","2.4":"2.3","3":"3","3.0":"B","3.1":"3.4","3.2":"3.1","3.3":"3.3","3.4":"B","3.5":"3.4","4.1":"4.1","4.2":"4.3","4.3":"4.1","4.4":"4.1","4.5":"4.4",
      "4.6":"4.2","4.7":"4.2","4.8":"4.2","4.9":"4.3","4.10":"4.4","4.11":"4.3","4.12":"D","5.2":"5","5.1":"6"}
def sec(m):
    v=SMAP[m.group(1)]; return v
doc=re.sub(r"\{\{S:(\d\.\d+)\}\}",sec,doc)
doc=re.sub(r"(sections?)&nbsp;([A-E])\b",lambda m:"Appendix&nbsp;"+m.group(2),doc)
doc=re.sub(r"sections?&nbsp;(4\.2)&ndash;4\.3","sections&nbsp;4.2&ndash;4.3",doc)
for a,b in [("sections&nbsp;3.1&ndash;B)","sections&nbsp;3.1&ndash;3.3)"),("(sections&nbsp;4.2, 4.4, D)","(sections&nbsp;4.2 and 4.4; Appendix&nbsp;D)"),
            ("stationary, and section&nbsp;3.3 gives the","stationary, and section&nbsp;3.2 gives the"),("curved-axial interiors (section&nbsp;3.3)","curved-axial interiors (section&nbsp;3.2)"),
            ("canonical coordinate</em> (section&nbsp;3.3)","canonical coordinate</em> (section&nbsp;3.2)"),("Curvilinear is the parallel-transported frame of section&nbsp;3.3","Curvilinear is the parallel-transported frame of section&nbsp;3.2")]:
    assert a in doc,a; doc=doc.replace(a,b)
assert "{{" not in doc, re.findall(r"\{\{[^}]*\}\}",doc)[:5]
open(DST,"w").write(doc)
print("figures",fn); print("tables",tn); print("equations",en)
