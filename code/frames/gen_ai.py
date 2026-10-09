# Cross-section photographs of non-fruit objects, generated (Gemini 2.5 Flash Image on Vertex AI).
# Every image is a separate request with its own seed, so every image is a different specimen --
# which is the method's own premise: unposed cuts of different specimens, never registered.
#
#   python gen_ai.py <outdir> [object ...]
#     outdir/<object>/raw/{long,trans}_<k>.png   k = 0..NPER-1; the ones that are not an orthographic cut face
#     (perspective, text, the wrong face) are rejected by eye before prep, and the rejections are listed in REJECT.txt
import os,sys,json,base64,subprocess,time,urllib.request,concurrent.futures as cf
P=os.environ.get("GCP_PROJECT","project-6d34f8a9-8150-4e6a-ae7"); MODEL="gemini-2.5-flash-image"
NPER=int(os.environ.get("NPER","9"))
URL=f"https://aiplatform.googleapis.com/v1/projects/{P}/locations/global/publishers/google/models/{MODEL}:generateContent"
STYLE=("Photorealistic macro photograph, flat orthographic top-down view with no perspective: only the flat cut surface "
       "is visible, none of the object's sides or depth. The cut face is centered and fills about 80% of the frame. "
       "Plain pure white seamless background, soft even light, no shadow, no props, no hands, no knife, no plate, "
       "no text, no labels, no numbers.")
OBJS={
 "log":       ("A short thick section of a young fast-grown oak log, as long as it is wide, split lengthwise exactly through the pith and lying flat. The camera looks straight down at the flat split face: a square, with a few broad straight vertical grain bands, a dark pith line down the centre and a strip of bark along the left and right edges",
               "A short thick section of a young fast-grown oak log seen end-on. The camera looks straight at the flat round end-grain face: about twelve broad, clearly separated growth rings around a small pith, a few radial cracks, a ring of bark around the edge"),
 "maki":      ("A thick futomaki sushi roll cut lengthwise in half through its centre, lying flat. The camera looks straight down at the flat cut face: a long rectangle, standing vertically in the frame, with stripes of fillings running top to bottom, white rice on both sides and dark nori along both long edges",
               "A thick futomaki sushi roll cut crosswise. The camera looks straight at the flat round cut face: dark nori ring, white rice, colourful fillings in the middle"),
 "honeycomb": ("A square block of natural honeycomb cut vertically, parallel to the comb cells. The camera looks straight at the flat cut face: long narrow straight wax cell tubes running from bottom to top like vertical stripes, golden honey between thin pale wax walls, wax caps along the top and bottom edges. No hexagons are visible on this face",
               "A square block of natural honeycomb cut horizontally, across the cells. The camera looks straight at the flat cut face: a square of hexagonal wax cells filled with golden honey"),
 "strata":    ("A square block of layered sedimentary sandstone cut vertically. The camera looks straight at the flat cut face: a square with many thin horizontal wavy layers of sand, rust, cream and grey rock, slightly undulating",
               "A square block of sedimentary sandstone cut horizontally along one bedding layer. The camera looks straight at the flat cut face: a square of mostly one rock colour with fine grain and a few patches of the neighbouring layers"),
 "terrazzo":  ("A square block of terrazzo cut vertically. The camera looks straight at the flat polished cut face: a square of coloured marble chips of many sizes set in grey cement",
               "A square block of terrazzo cut horizontally. The camera looks straight at the flat polished cut face: a square of coloured marble chips of many sizes set in grey cement"),
 "cheese":    ("A square block of Emmental cheese cut vertically. The camera looks straight at the flat cut face: a pale yellow square with round holes of various sizes",
               "A square block of Emmental cheese cut horizontally. The camera looks straight at the flat cut face: a pale yellow square with round holes of various sizes"),
 "onion":     ("A brown onion cut lengthwise in half from root to tip, lying flat. The camera looks straight down at the flat cut face, root at the bottom and tip at the top: nested curved white layers converging at the root and at the tip, a thin brown skin around the edge",
               "A brown onion cut crosswise through its equator. The camera looks straight down at the flat round cut face: concentric white rings, a thin brown skin around the edge"),
}
OBJS.update({
 "swissroll": ("A thick Swiss roll sponge cake cut lengthwise in half through its centre, lying flat. The camera looks straight down at the flat cut face: a rectangle, standing vertically in the frame, with stripes of cream and sponge running top to bottom",
               "A thick Swiss roll sponge cake cut crosswise. The camera looks straight at the flat round cut face: a spiral of golden sponge and white cream"),
 "egg":       ("A hard-boiled egg cut lengthwise in half from tip to tip, lying flat. The camera looks straight down at the flat cut face, the long axis vertical in the frame: white albumen around an oval yellow yolk",
               "A hard-boiled egg cut crosswise through its middle. The camera looks straight down at the flat round cut face: a ring of white albumen around a round yellow yolk"),
 "cabbage":   ("A red cabbage cut in half from top to bottom through the core, lying flat. The camera looks straight down at the flat cut face, the core vertical in the frame: purple and white leaf layers folding around a pale central core",
               "A red cabbage cut crosswise through its middle. The camera looks straight down at the flat round cut face: tightly packed purple and white leaf layers swirling around the centre"),
 "carrot":    ("A thick carrot cut lengthwise in half through its centre, lying flat. The camera looks straight down at the flat cut face, the carrot vertical in the frame: orange flesh with a paler central core running top to bottom",
               "A thick carrot cut crosswise. The camera looks straight at the flat round cut face: an orange outer ring and a paler star-shaped central core"),
 "candle":    ("A thick round pillar candle made of many horizontal coloured wax layers, cut lengthwise in half through its centre. The camera looks straight at the flat cut face: a rectangle standing vertically, horizontal bands of coloured wax and the wick running down the centre",
               "A thick round pillar candle made of coloured wax cut crosswise. The camera looks straight at the flat round cut face: one wax colour with faint concentric pour rings and the wick dot in the centre"),
 "bamboo":    ("A thick green bamboo stalk segment cut lengthwise in half through its centre, lying flat. The camera looks straight down at the flat cut face, standing vertically in the frame: the hollow interior, a solid node wall across the middle, fibrous pale walls on both sides",
               "A thick green bamboo stalk cut crosswise through a node. The camera looks straight at the flat round cut face: a solid pale fibrous disc with a green outer skin ring"),
 "cinnamonroll": ("A tall cinnamon roll cut vertically in half through its centre. The camera looks straight at the flat cut face: layers of soft bread and dark cinnamon swirl stacked side by side, standing upright",
               "A cinnamon roll seen from the top after slicing off the top, the flat round cut face: a spiral of soft bread and dark brown cinnamon sugar"),
 "gobstopper": ("A large gobstopper jawbreaker candy cut in half through its centre. The camera looks straight at the flat round cut face: many thin concentric coloured sugar layers around a small white centre",
               "A large gobstopper jawbreaker candy cut in half through its centre. The camera looks straight at the flat round cut face: many thin concentric coloured sugar layers around a small white centre"),
 "golfball":  ("A golf ball cut in half through its centre. The camera looks straight at the flat round cut face: a thin white dimpled cover, a thin mantle layer and a large coloured rubber core",
               "A golf ball cut in half through its centre. The camera looks straight at the flat round cut face: a thin white dimpled cover, a thin mantle layer and a large coloured rubber core"),
 "geode":     ("A round geode rock cut in half through its centre and polished. The camera looks straight at the flat cut face: a grey rock rind, bands of agate, and sparkling quartz crystals lining a hollow centre",
               "A round geode rock cut in half through its centre and polished. The camera looks straight at the flat cut face: a grey rock rind, bands of agate, and sparkling quartz crystals lining a hollow centre"),
 "baseball":  ("A baseball cut in half through its centre. The camera looks straight at the flat round cut face: white leather cover with red stitches at the edge, tightly wound grey and white yarn layers, a small cork and rubber centre",
               "A baseball cut in half through its centre. The camera looks straight at the flat round cut face: white leather cover with red stitches at the edge, tightly wound grey and white yarn layers, a small cork and rubber centre"),
 "tiramisu":  ("A square block of tiramisu cut vertically. The camera looks straight at the flat cut face: a square with horizontal layers of coffee-soaked ladyfingers and white mascarpone cream, cocoa powder on top",
               "A square block of tiramisu cut horizontally through a cream layer. The camera looks straight at the flat cut face: a square of white mascarpone cream with patches of coffee-soaked sponge"),
 "marble":    ("A square block of white Carrara marble cut vertically and polished. The camera looks straight at the flat cut face: a white square with thin grey veins running diagonally",
               "A square block of white Carrara marble cut horizontally and polished. The camera looks straight at the flat cut face: a white square with thin grey veins"),
})
OBJS["cable"]=("A short thick section of a four-core power cable, as long as it is wide, sawn lengthwise exactly through its centre and lying flat. The camera looks straight down at the flat sawn face: a square. Two of the four conductors are cut along their length: each appears over the FULL height of the face as one straight solid copper band with a thin line of its insulation colour on each side (brown on the left conductor, blue on the right). The copper is never stripped or exposed at the ends and the insulation never covers the copper: every horizontal line through the face shows the same layers. White filler between and around the conductors, a thick black outer jacket band along the left and right edges",
               "A four-core power cable sawn crosswise. The camera looks straight at the flat round sawn face, which fills the frame: exactly four equal round solid copper conductors arranged in a square, each in a ring of coloured insulation (brown, black, grey, blue), white filler between them, a thick black outer jacket ring")
OBJS["banana"]=("A ripe yellow Cavendish banana, peel on, sliced lengthwise exactly through its centre along its whole curved length, lying flat with its curve in the picture plane and its stem at the top. The camera looks straight down at the flat cut face: the curved ivory flesh with the faint darker central band of the locules and a few tiny dark seed specks along it, a thin yellow peel along both curved edges",
                 "A ripe yellow Cavendish banana, peel on, sliced crosswise perpendicular to its length with a sharp knife. The camera looks straight down at the flat round cut face, centered and filling the frame: thin yellow peel with a pale inner layer, ivory flesh, and the soft translucent three-lobed locule pattern of a real banana with a few tiny dark seed specks in the centre")
EXT={
 "banana":"A whole ripe yellow Cavendish banana standing upright on its flower end, stem at the top, its gentle curve seen from the side",
 "cable":"A short thick section of a black four-core power cable, as long as it is wide, standing upright on one cut end showing the four coloured cores",
 "log":"A short thick section of a young oak log, as long as it is wide, standing upright on one flat end",
 "maki":"A thick futomaki sushi roll standing upright on one cut end",
 "onion":"A whole brown onion standing upright, root at the bottom",
 "honeycomb":"A square block of natural honeycomb, the comb cells running vertically",
 "strata":"A square block of layered sedimentary sandstone with horizontal layers",
 "terrazzo":"A square block of polished terrazzo",
 "cheese":"A square block of Emmental cheese",
 "swissroll":"A thick Swiss roll sponge cake standing upright on one end",
 "egg":"A whole peeled hard-boiled egg standing upright on its end",
 "cabbage":"A whole red cabbage, core at the bottom",
 "carrot":"A short thick section of carrot standing upright on one cut end",
 "candle":"A thick round pillar candle made of many horizontal coloured wax layers, standing upright",
 "bamboo":"A short thick green bamboo stalk segment standing upright",
 "cinnamonroll":"A tall cinnamon roll standing upright",
 "gobstopper":"A large round gobstopper jawbreaker candy",
 "golfball":"A white golf ball",
 "geode":"A round rough grey geode rock, uncut",
 "baseball":"A baseball with red stitching",
 "tiramisu":"A square block of tiramisu with cocoa powder on top",
 "marble":"A square block of white Carrara marble with grey veins",
}
EXT_STYLE=(" Seen from a three-quarter view slightly from above, the whole object in frame and centered. "
           "Photorealistic product photograph, plain pure white seamless background, soft even light, no shadow, no props, no text.")

def token(): return subprocess.check_output(["gcloud","auth","print-access-token"],text=True,env={**os.environ,"PATH":os.path.expanduser("~/google-cloud-sdk/bin")+":"+os.environ["PATH"]}).strip()
TOK=token()
def gen(prompt,seed,path):
    if os.path.exists(path): return path
    body=json.dumps({"contents":[{"role":"user","parts":[{"text":prompt}]}],
                     "generationConfig":{"responseModalities":["IMAGE"],"seed":seed,"imageConfig":{"aspectRatio":"1:1"}}}).encode()
    for a in range(5):
        try:
            r=urllib.request.Request(URL,data=body,headers={"Authorization":"Bearer "+TOK,"Content-Type":"application/json"})
            d=json.load(urllib.request.urlopen(r,timeout=180))
            for p in d["candidates"][0]["content"]["parts"]:
                if "inlineData" in p: open(path,"wb").write(base64.b64decode(p["inlineData"]["data"])); return path
        except Exception as e: print("  retry",path,e,flush=True); time.sleep(10*(a+1))
    return None
if __name__=="__main__":
    out=sys.argv[1]; names=sys.argv[2:] or list(OBJS); jobs=[]
    for n in names:
        lon,tra=OBJS[n]; d=f"{out}/{n}/raw"; os.makedirs(d,exist_ok=True)
        for k in range(NPER):
            jobs.append((f"{lon}. {STYLE}",100+k,f"{d}/long_{k}.png"))
            jobs.append((f"{tra}. {STYLE}",200+k,f"{d}/trans_{k}.png"))
        e=f"{out}/{n}/exterior"; os.makedirs(e,exist_ok=True)
        for k in range(4): jobs.append((EXT[n]+"."+EXT_STYLE,400+k,f"{e}/ext_{k}.png"))
    with cf.ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda j: gen(*j),jobs): print("  ", r, flush=True)
