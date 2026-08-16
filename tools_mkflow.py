"""Emit the pipeline figure.

Hand-placing every string in an SVG is how text ends up outside the box that was drawn for it:
the box is a number and the string is a number of characters, and nothing checks that the second
fits inside the first. So the boxes are declared with their content here and the wrapping is
measured, at the same per-character widths the page's own font stack gives, which is what makes
the figure survive an edit to any line of it.
"""
import textwrap

W_S, W_T, W_TH, W_M = 5.55, 6.75, 7.9, 6.5      # px per character, by class
ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def esc(t):
    for a, b in ESC.items():
        t = t.replace(a, b)
    return t.replace("\x00", "&")                # \x00 marks a pre-escaped entity


def wrap(text, width_px, per_char):
    n = max(8, int(width_px / per_char))
    return textwrap.wrap(text, n) or [""]


class Fig:
    def __init__(self):
        self.out = []

    def add(self, s):
        self.out.append(s)

    def box(self, x, y, w, h, cls="b", r=4):
        self.add(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/>')

    def text(self, x, y, s, cls="s", per=W_S, lh=15.5, width=None, anchor=None):
        """Returns the y after the last line, so a caller can stack blocks without arithmetic."""
        a = f' text-anchor="{anchor}"' if anchor else ""
        lines = wrap(s, width, per) if width else [s]
        for i, ln in enumerate(lines):
            self.add(f'<text class="{cls}" x="{x}" y="{y + i * lh}"{a}>{esc(ln)}</text>')
        return y + (len(lines) - 1) * lh

    def block(self, x, y, w, title, body, eqs=None, cls_t="t"):
        """A titled paragraph inside a box of width w, with its equation numbers set right."""
        per = W_T if cls_t == "t" else W_TH
        y = self.text(x, y, title, cls_t, per, 16, w)
        y = self.text(x, y + 17, body, "s", W_S, 15.5, w)
        if eqs:
            self.add(f'<text class="eq" x="{x + w}" y="{y + 16}" text-anchor="end">{eqs}</text>')
            y += 16
        return y

    def grid(self, x, y, cols, rows, step):
        g = [f'<g class="gl" transform="translate({x},{y})">']
        for r in range(rows + 1):
            g.append(f'<path d="M0,{r * step} H{cols * step}"/>')
        for c in range(cols + 1):
            g.append(f'<path d="M{c * step},0 V{rows * step}"/>')
        g.append("</g>")
        self.add("\n".join(g))

    def cells(self, x, y, step, coords, cls="oc"):
        r = "".join(f'<rect x="{a * step}" y="{b * step}" width="{step}" height="{step}"/>'
                    for a, b in coords)
        self.add(f'<g class="{cls}" transform="translate({x},{y})">{r}</g>')

    def key(self, x, y, cls, label):
        self.add(f'<rect class="{cls}" x="{x}" y="{y - 9}" width="11" height="11"/>')
        self.text(x + 18, y, label)

    def arrow(self, d, cls="l"):
        self.add(f'<path class="{cls}" d="{d}"/>')

    def lane(self, y, h, tag):
        self.box(0, y, 1160, h, "lane")
        self.add(f'<text class="lanet" x="14" y="{y + 20}">{esc(tag)}</text>')


f = Fig()
HEAD = '''<div class="flow">
<svg viewBox="0 0 1160 1348" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="The pipeline: two routes to one lattice, the two structures it carries, and the
                 cutting, drawing and physics that read it">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7.5" markerHeight="7.5"
            orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8a8a8a"/></marker>
    <marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
            orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#a8412a"/></marker>
    <style>
      .lane { fill:#fbfbfb; stroke:#efefef; stroke-width:1 }
      .lanet{ font:600 10.5px -apple-system,Segoe UI,Helvetica,Arial,sans-serif; fill:#a0a0a0;
              letter-spacing:1.5px }
      .b   { fill:#ffffff; stroke:#dcdcdc; stroke-width:1 }
      .ba  { fill:#fdf5f2; stroke:#e0b6aa; stroke-width:1 }
      .bb  { fill:#f3f8fc; stroke:#bcd5e7; stroke-width:1 }
      .hub { fill:#fbe6de; stroke:#c9836c; stroke-width:1.4 }
      .t   { font:600 13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif; fill:#111 }
      .th  { font:600 15px -apple-system,Segoe UI,Helvetica,Arial,sans-serif; fill:#111 }
      .s   { font:11.5px -apple-system,Segoe UI,Helvetica,Arial,sans-serif; fill:#5c5c5c }
      .m   { font:600 11px ui-monospace,Menlo,Consolas,monospace; fill:#a8412a }
      .eq  { font:italic 11px Georgia,serif; fill:#9a9a9a }
      .l   { stroke:#8a8a8a; stroke-width:1.3; fill:none; marker-end:url(#ar) }
      .la  { stroke:#a8412a; stroke-width:1.6; fill:none; marker-end:url(#arw) }
      .gl  { stroke:#e4e4e4; stroke-width:0.8; fill:none }
      .oc  { fill:#fbe6de; stroke:#e0b6aa; stroke-width:0.8 }
      .sk  { fill:#e2b09a; stroke:#c9836c; stroke-width:0.8 }
      .cut { stroke:#1b6ca8; stroke-width:1.9; fill:none }
      .dot { fill:#c9836c }
    </style>
  </defs>'''

# ---------------------------------------------------------------- A: what arrives
f.lane(6, 186, "A · WHAT ARRIVES")
f.box(34, 34, 520, 146, "ba")
f.block(52, 58, 300, "Route 1 · a released reconstruction",
        "Structure and appearance are both already in the file, and the interior it carries is "
        "somebody else's generated interior.")
f.text(52, 158, "prefilled/trained_gs/*.ply", "m", W_M)
f.add('<g transform="translate(400,58)">'
      '<circle cx="52" cy="50" r="46" fill="none" stroke="#eddcd5" stroke-width="1"/>'
      '<g class="dot">' + "".join(
          f'<circle cx="{52 + 46 * __import__("math").cos(a * 0.3927):.1f}" '
          f'cy="{50 + 46 * __import__("math").sin(a * 0.3927):.1f}" r="1.8"/>' for a in range(16))
      + '<circle cx="52" cy="38" r="1.4"/><circle cx="63" cy="58" r="1.4"/>'
        '<circle cx="40" cy="60" r="1.4"/></g>'
      '<text class="s" x="-6" y="116">points on a surface</text>'
      '<text class="s" x="-6" y="131">with nothing behind it</text></g>')

f.box(606, 34, 520, 146, "bb")
f.block(624, 58, 300, "Route 2 · nothing but a description",
        "No reconstruction was ever made. A signed-distance field and six rendered views are all "
        "that exists of the object.")
f.text(624, 158, "φ(x) + six views", "m", W_M)
f.add('<g transform="translate(972,58)">'
      '<circle cx="52" cy="50" r="33" fill="#eaf1f7" stroke="#8fb5d0" stroke-width="1.2"/>'
      '<circle cx="52" cy="50" r="43" fill="none" stroke="#cfe0ec" stroke-width="0.8" '
      'stroke-dasharray="3 3"/>'
      '<circle cx="52" cy="50" r="23" fill="none" stroke="#cfe0ec" stroke-width="0.8" '
      'stroke-dasharray="3 3"/>'
      '<g fill="#1b6ca8">'
      '<rect x="48" y="-4" width="9" height="6" rx="1"/><rect x="48" y="98" width="9" height="6" rx="1"/>'
      '<rect x="-4" y="47" width="6" height="9" rx="1"/><rect x="98" y="47" width="6" height="9" rx="1"/>'
      '<rect x="12" y="12" width="8" height="6" rx="1"/><rect x="84" y="82" width="8" height="6" rx="1"/></g>'
      '<text class="s" x="0" y="116">a field, and six</text>'
      '<text class="s" x="0" y="131">known cameras</text></g>')

# ---------------------------------------------------------------- B: to the lattice
f.lane(200, 232, "B · TO THE LATTICE")
f.arrow("M294,180 L294,232")
f.arrow("M866,180 L866,232")

for x, title, body, eq in [
    (34, "quantise", "One cell per primitive, floored from a corner and not from a centre.", "(4)"),
    (296, "close, then fill", "A photographed surface leaves voids behind it. These remove them.", "(5)"),
    (606, "occupancy from the field",
     "φ < 0 is inside; the band −βhₑ < φ < 0 is skin.", "(6)"),
    (868, "project the six views",
     "Every camera is known, so colour is a lookup weighted by incidence.", "(7)"),
]:
    f.box(x, 240, 258, 84)
    f.block(x + 16, 262, 226, title, body, eq)

f.arrow("M852,282 L864,282")

f.text(34, 348, "Solid only after a repair step: closing adds 11.7% of the orange's cells, and the "
                "uncut object goes from 561 pieces to one.", width=500)
f.text(606, 348, "Solid by construction: closing adds 0.1%, the uncut object is one piece, and the "
                 "shell thickness β is an argument to the builder.", width=500)
f.arrow("M294,392 L294,412 L556,412 L556,440", "la")
f.arrow("M866,392 L866,412 L604,412 L604,440", "la")

# ---------------------------------------------------------------- C: the lattice
f.lane(444, 240, "C · ONE TWO-LEVEL LATTICE")
f.box(34, 474, 1092, 196, "hub")
y = f.block(56, 500, 420, "The same structure, whichever route reached it",
            "Downstream reads three things and asks nothing else: occupancy at two levels, face "
            "adjacency, and one learned feature per cell. Which route produced them is not "
            "recoverable from here on.", cls_t="th")
f.text(56, y + 24, "V₀ ∪ V₁  A  fₖ ∈ ℝ⁸",
       "m", W_M)
f.add('<text class="eq" x="56" y="%d">(1) (2) (3)</text>' % (y + 42))
f.text(56, y + 68, "1.74M cells quantised or 1.07M generated for the watermelon, against the "
                   "7.96M primitives of the released model route 1 quantises.", width=420)

CX, CY, ST = 560, 494, 17
f.grid(CX, CY, 9, 9, ST)
f.cells(CX, CY, ST, [(2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3),
                     (4, 3), (5, 3), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (2, 5), (3, 5),
                     (4, 5)], "oc")
f.add('<g class="sk" transform="translate(%d,%d)">' % (CX, CY) + "".join(
    f'<rect x="{a * ST / 2}" y="{b * ST / 2}" width="{ST / 2}" height="{ST / 2}"/>'
    for a, b in [(4, 1), (5, 1), (6, 1), (7, 1), (2, 3), (3, 3), (8, 3), (9, 3),
                 (1, 6), (1, 7), (10, 6), (10, 7), (2, 10), (3, 10), (8, 10), (9, 10),
                 (4, 12), (5, 12), (6, 12), (7, 12)]) + "</g>")
f.key(740, 520, "oc", "level 0, spacing h₀, the interior")
f.key(740, 548, "sk", "level 1, spacing hₑ, the skin band")
f.text(740, 582, "hₑ = h₀ / 2ℓ with refine = 2. Level 1 exists only in the skin "
                 "band, so the interior is coarse and the surface is not.", width=370)
f.text(740, 636, "One feature vector per cell, decoded to colour on demand.", width=370)

f.arrow("M300,670 L300,706")
f.arrow("M860,670 L860,706")

# ---------------------------------------------------------------- D: two structures
f.lane(700, 168, "D · TWO STRUCTURES, ONE OCCUPANCY")
f.box(34, 730, 536, 124)
y = f.block(52, 754, 340, "The cube hierarchy · the physical volume",
            "Answers where material is, what is connected to what, and which piece a point belongs "
            "to. A cube's support is its cell, so a gap is a hole and not a coverage parameter.")
f.text(52, y + 20, "Never rendered by splatting.")
f.add('<g transform="translate(424,758)">'
      '<path d="M0,62 H18 V46 H36 V30 H54 V14 H72 V0" stroke="#c9836c" stroke-width="1.6" '
      'fill="none"/><text class="s" x="-14" y="82">a staircase at the cell size</text></g>')

f.box(590, 730, 536, 124)
y = f.block(608, 754, 340, "The O‑Voxel dual grid · the visible boundary",
            "One dual vertex per boundary cell, joined across shared faces. Explicit polygons, so "
            "coverage comes from triangles and not from primitives dense enough to hide gaps.")
f.text(608, y + 20, "Takes no part in the physics.")
f.add('<g transform="translate(980,758)">'
      '<path d="M0,60 C18,54 34,40 50,20 C58,10 64,4 70,0" stroke="#1b6ca8" stroke-width="1.6" '
      'fill="none"/><g fill="#1b6ca8"><circle cx="0" cy="60" r="2.5"/><circle cx="18" cy="54" r="2.5"/>'
      '<circle cx="36" cy="38" r="2.5"/><circle cx="52" cy="18" r="2.5"/><circle cx="68" cy="2" r="2.5"/>'
      '</g><text class="s" x="0" y="82">the same boundary, placed</text></g>')

# ---------------------------------------------------------------- E: what reads it
f.lane(872, 466, "E · WHAT READS THE LATTICE")
f.arrow("M300,854 L300,900")
f.arrow("M860,854 L860,900")

# E1 cutting
f.box(34, 906, 350, 420, "ba")
f.text(52, 932, "Cutting", "th", W_TH)
f.text(52, 950, "discrete on the lattice, analytic on the boundary", width=310)
GX, GY, GS = 52, 962, 18
f.grid(GX, GY, 16, 4, GS)
f.add('<g fill="#1b6ca8" fill-opacity="0.16" transform="translate(%d,%d)">' % (GX, GY) + "".join(
    f'<rect x="{i * GS}" y="{54 - i * 7}" width="{GS}" height="{GS}"/>' for i in range(8))
    + "</g>")
f.add(f'<path class="cut" d="M{GX},{GY + 66} L{GX + 288},{GY + 2}"/>')
y = f.block(52, 1080, 310, "1 · a sign per leaf",
            "An adjacency survives when its two leaves agree, and a piece is a connected "
            "component of what is left.", "(8) (9)")
y = f.block(52, y + 24, 310, "2 · refine only the band",
            "O(N²ᐟ³) of the volume, and returned when the cut is not live.")
f.block(52, y + 24, 310, "3 · the face is a polygon, not an approximation",
        "The plane against the cell's twelve edges, in closed form: every vertex on the plane to "
        "0.00e+00.", "(10)")

# E2 drawing
f.box(400, 906, 316, 420)
f.text(418, 932, "Drawing", "th", W_TH)
f.text(418, 950, "the boundary, and the volume behind it", width=280)
f.add('<g transform="translate(418,964)">'
      '<rect x="6" y="8" width="86" height="64" class="oc" fill-opacity="0.55"/>'
      '<path d="M6,64 L58,18" stroke="#3a3a3a" stroke-width="1.3" fill="none" stroke-dasharray="4 3"/>'
      '<path d="M58,18 L92,36" stroke="#3a3a3a" stroke-width="1.3" fill="none" stroke-dasharray="4 3"/>'
      '<path d="M30,40 L16,26" stroke="#8a8a8a" stroke-width="1" fill="none"/>'
      '<path d="M78,26 L88,12" stroke="#8a8a8a" stroke-width="1" fill="none"/>'
      '<circle cx="58" cy="18" r="3.4" fill="#a8412a"/>'
      '<text class="s" x="106" y="30">each plane through the</text>'
      '<text class="s" x="106" y="46">cell adds a term; the</text>'
      '<text class="s" x="106" y="62">vertex is the minimiser</text></g>')
y = f.block(418, 1080, 280, "the dual vertex is a quadric's minimiser",
            "Ten numbers per cell, additive over the planes it covers, which is what makes an "
            "adaptive grid a collapse.", "(11)")
y = f.block(418, y + 24, 280, "collapse to a tolerance",
            "The error is read off the same quadric that placed the vertex.", "(12)")
f.block(418, y + 24, 280, "the volume is drawn by lookup",
        "The smallest cell containing a pixel, decoded through the shared head, trilinear between "
        "cell centres.", "(18)")

# E3 physics
f.box(732, 906, 394, 420, "bb")
f.text(750, 932, "Physics", "th", W_TH)
f.text(750, 950, "materials, contact, and the surface that follows them", width=360)
PX, PY, PS = 750, 962, 16
f.grid(PX, PY, 9, 4, PS)
f.add('<g fill="#e0b6aa" fill-opacity="0.9" transform="translate(%d,%d)">' % (PX, PY) + "".join(
    f'<rect x="{i * PS}" y="{48 - i * 6}" width="{PS}" height="{PS}"/>' for i in range(7))
    + "</g>")
f.add(f'<path class="cut" d="M{PX},{PY + 58} L{PX + 144},{PY + 2}"/>')
f.text(910, 978, "A leaf the plane crosses is occupied as a whole, so occupancy alone claims "
                 "material on both sides of the cut.", width=210)
f.add('<text class="s" x="910" y="1040" fill="#a8412a">One sign test removes that band.</text>')
y = f.block(750, 1080, 356, "a stiffness per class, not one modulus",
            "Classes come from the same learned feature. Shell fraction and brightness order them, "
            "and the ordering is the claim.", "(14)")
y = f.block(750, y + 22, 356, "can the solver see the shell at all?",
            "τ = tₛₕₑₗₗ / Δx has to reach about two, and route 2 "
            "can build for it because β is an argument. It is 0.87 to 2.00 as built.", "(15)")
f.block(750, y + 22, 356, "contact is three integer operations",
        "A floor division, a table lookup and one sign comparison. 2,698 false contacts at rest "
        "become 0, and the surface follows the particles by weights fixed at bind time.",
        "(13) (16) (17)")

print(HEAD + "\n  " + "\n  ".join(f.out) + "\n</svg>\n</div>")
