"""Render the paper to a two-column CGF-style PDF.

The page is written for a browser: KaTeX typesets the mathematics client-side, the layout is one
wide column, and the figures are sized for scrolling. None of that survives a print renderer, which
runs no JavaScript. So this does three things and nothing else:

  1. replaces the LaTeX fragments with a Unicode transcription, because a PDF that cannot show its
     own equations is not a backup of anything;
  2. swaps the screen stylesheet for a Computer Graphics Forum one -- A4, two columns, 9pt serif,
     the section numbering the page already carries;
  3. drops the parts of the page that exist only on screen (the table of contents, the video
     figures, the BibTeX block).

It is a backup rather than a submission: the real submission needs the eurographics LaTeX class,
and this exists so that the paper survives the website.

    python tools_makepdf.py index.html paper_cgf.pdf              # the whole record
    python tools_makepdf.py index.html paper_sub.pdf --submission  # cut to a venue's length

`--submission` drops the sections tagged `data-supp` in the page. The page is the full record and
the record is worth keeping: three of the four claims this paper withdrew were found by someone
reading a result against the code that produced it, which a ten-page paper does not carry. So the
selection lives here rather than in the manuscript, and the two outputs disagree about length
without disagreeing about anything else.
"""
import html
import re
import sys

# Enough of LaTeX to read the eighteen equations this paper has. A transcription, not a typesetter:
# where a construct has no Unicode form -- fractions, big sums with limits -- it becomes the form a
# reader would say aloud, which is what a backup needs.
SYM = {
    r"\rho": "ρ", r"\sigma": "σ", r"\tau_c": "τ_c", r"\tau_s": "τ_s", r"\tau": "τ",
    r"\beta_s": "β_s", r"\beta_m": "β_m", r"\beta": "β", r"\phi": "φ", r"\gamma": "γ",
    r"\Delta": "Δ", r"\Pi": "Π", r"\Omega": "Ω", r"\varepsilon": "ε", r"\epsilon": "ε",
    r"\mathcal{V}": "V", r"\mathcal{S}": "S", r"\mathcal{A}": "A", r"\mathcal{R}": "R",
    r"\mathcal{N}": "N", r"\mathcal{O}": "O", r"\mathbf{x}": "x", r"\mathbf{n}": "n",
    r"\mathbf{k}": "k", r"\mathbf{o}": "o", r"\mathbf{d}": "d", r"\mathbf{a}": "a",
    r"\mathbf{b}": "b", r"\mathbf{c}": "c", r"\mathbf{t}": "t", r"\mathbf{A}": "A",
    r"\mathbf{\delta}": "δ", r"\delta": "δ", r"\ell": "ℓ", r"\le": "≤", r"\ge": "≥",
    r"\times": "×", r"\cdot": "·", r"\in": "∈", r"\notin": "∉", r"\cup": "∪", r"\cap": "∩",
    r"\emptyset": "∅", r"\wedge": "∧", r"\neg": "¬", r"\gg": "≫", r"\ll": "≪",
    r"\gtrsim": "≳", r"\approx": "≈", r"\sum": "Σ", r"\prod": "Π", r"\nabla": "∇",
    r"\lVert": "‖", r"\rVert": "‖", r"\lfloor": "⌊", r"\rfloor": "⌋", r"\partial": "∂",
    r"\operatorname{sgn}": "sgn", r"\operatorname{leaf}": "leaf", r"\operatorname{occ}": "occ",
    r"\operatorname{dist}": "dist", r"\operatorname{parent}": "parent", r"\complement": "∁",
    r"\oplus": "⊕", r"\ominus": "⊖", r"\qquad": "   ", r"\quad": "  ", r"\,": " ", r"\;": " ",
    r"\!": "", r"\left": "", r"\right": "", r"\bigl": "", r"\bigr": "", r"\Bigl": "",
    r"\Bigr": "", r"\textstyle": "", r"\displaystyle": "", r"\mathbb{1}": "1",
    r"\top": "ᵀ", r"\text": "", r"\max": "max", r"\min": "min", r"\pi": "π",
}


def demath(t):
    """LaTeX fragment to a readable Unicode line."""
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\tfrac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
    t = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", t)
    for a, b in sorted(SYM.items(), key=lambda kv: -len(kv[0])):
        t = t.replace(a, b)
    t = re.sub(r"_\{([^{}]*)\}", r"_\1", t)
    t = re.sub(r"\^\{([^{}]*)\}", r"^\1", t)
    t = t.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", t).strip()


CSS = """
@page { size: A4; margin: 20mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page); font: 8pt Georgia, serif; color:#444 } }
html { font: 9pt/1.32 Georgia, "Times New Roman", serif; color:#111 }
body { column-count: 2; column-gap: 6mm; text-align: justify; hyphens: auto }
h1 { column-span: all; font: bold 17pt Georgia, serif; text-align:center; margin:0 0 2mm }
.sub { column-span: all; text-align:center; font-style:italic; font-size:10pt; margin:0 0 1mm }
.authors, .venue { column-span: all; text-align:center; font-size:9pt; margin:0 }
.venue { margin-bottom: 4mm }
h2 { font: bold 10.5pt Georgia, serif; margin: 3.5mm 0 1.4mm; break-after: avoid }
h3 { font: bold 9.5pt Georgia, serif; margin: 3mm 0 1.2mm; break-after: avoid }
h4 { font: italic bold 9pt Georgia, serif; margin: 2.4mm 0 1mm; break-after: avoid }
p { margin: 0 0 1.6mm }
p.lead { font-size: 9pt }
.eq { margin: 1.8mm 0; break-inside: avoid; display: flex; gap: 2mm; align-items: baseline }
.eq .body { flex: 1; font-style: italic; text-align: center; font-size: 8.6pt }
.eq .num { font-size: 8pt; color:#444 }
table { width: 100%; border-collapse: collapse; font-size: 7.4pt; margin: 1.6mm 0;
        break-inside: avoid }
th, td { border-top: 0.4pt solid #bbb; padding: 0.7mm 1mm; text-align: left; vertical-align: top }
thead th { border-bottom: 0.4pt solid #555; font-weight: bold }
figure { margin: 2mm 0; break-inside: avoid; text-align: center }
/* A figure that fills a column in a browser fills a third of a page in print, and this paper has
   fifteen of them. Bounding the height rather than the width keeps a tall multi-row sheet from
   taking a page to itself while a wide one still spans its column. */
figure img { max-width: 100%; max-height: 62mm; width: auto; height: auto }
figcaption { font-size: 7.4pt; color:#333; margin-top: 0.8mm }
.note { border-left: 1pt solid #888; padding-left: 2mm; font-size: 8.2pt; margin: 1.8mm 0;
        break-inside: avoid }
.mut { color:#555 }
ol.refs { font-size: 7.6pt; padding-left: 4mm; margin: 0 }
ol.refs li { margin-bottom: 0.5mm }
ul { padding-left: 4mm; margin: 0 0 1.6mm }
li { margin-bottom: 0.8mm }
svg { max-width: 100%; max-height: 150mm; height: auto }
.flow svg { max-height: 150mm }
.flow { break-inside: avoid; margin: 2mm 0 }
code { font: 7.6pt "DejaVu Sans Mono", monospace }
"""


def main(src, out, submission=False):
    s = open(src, encoding="utf-8").read()

    if submission:
        # Drop, not summarise: a section that survives at a third of its length is worse than a
        # pointer to the full version, and the full version is a file away.
        n = len(re.findall(r'data-supp="1"', s))
        s = re.sub(r"<(h[234]|div|figure|p|section)[^>]*data-supp=\"1\"[^>]*>.*?</\1>",
                   "", s, flags=re.S)
        s = re.sub(r"<!--supp-->.*?<!--/supp-->", "", s, flags=re.S)
        print(f"  submission cut: {n} tagged blocks")

    # screen-only furniture
    s = re.sub(r"<nav class=\"toc\">.*?</nav>", "", s, flags=re.S)
    s = re.sub(r"<div class=\"links\">.*?</div>", "", s, flags=re.S)
    s = re.sub(r"<h2 id=\"bibtex\">.*?(?=</div>\s*</body>|\Z)", "", s, flags=re.S)
    s = re.sub(r"<script.*?</script>", "", s, flags=re.S)
    s = re.sub(r"<style>.*?</style>", "", s, flags=re.S)
    s = re.sub(r"<link[^>]*>", "", s)
    # a video has no still frame to print; its caption still carries the result
    s = re.sub(r"<video[^>]*>.*?</video>", "", s, flags=re.S)

    # display equations, then inline
    s = re.sub(r"\$\$(.*?)\$\$", lambda m: html.escape(demath(m.group(1))), s, flags=re.S)
    s = re.sub(r"\\\((.*?)\\\)", lambda m: html.escape(demath(m.group(1))), s, flags=re.S)

    s = s.replace("</head>", f"<style>{CSS}</style></head>")

    from weasyprint import HTML
    HTML(string=s, base_url=src).write_pdf(out)
    print(f"  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], submission="--submission" in sys.argv)
