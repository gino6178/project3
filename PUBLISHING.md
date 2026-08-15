# How this page gets published

The whole mechanism is a git push. GitHub Pages serves `main` of `gino6178/project3` at
`https://gino6178.github.io/project3/`, so publishing is committing and pushing — there is no
build, no CI, no deploy step, and nothing to install.

## The setup that already exists

```
/home/gino/project/project3-site/     the working copy
  index.html                          the entire page, one file
  assets/                             every figure, PNG
  .nojekyll                           see below
  README.md
  PUBLISHING.md                       this file

remote   git@github.com:gino6178/project3.git   (SSH, not HTTPS)
branch   main
author   gino6178 <gino6178@gmail.com>          (set per-repo, not globally)
```

Authentication is the SSH key at `~/.ssh/id_ed25519`, already registered with GitHub. Check it
without pushing anything:

```bash
ssh -T git@github.com
# Hi gino6178! You've successfully authenticated, but GitHub does not provide shell access.
```

That message is success. GitHub never gives shell access, so the second half is not an error.

`.nojekyll` is an empty file and it matters: without it GitHub runs Jekyll over the repository,
and Jekyll ignores files and directories whose names begin with an underscore. Nothing here is
named that way today, but the failure mode is a figure that silently 404s, which is worth one
empty file to rule out.

## Publishing

```bash
cd /home/gino/project/project3-site
git add -A
git commit -m "what changed and why"
git push origin main
```

Then wait. The push returns immediately; Pages rebuilds afterwards, usually in 10–60 seconds
and occasionally longer. Poll for the change rather than guessing:

```bash
# wait for a phrase you just added to appear
until curl -s https://gino6178.github.io/project3/ | grep -q "the phrase"; do sleep 15; done

# and check every new asset actually resolves -- a 404 here is the common failure
for f in skin_options pxcell cutface; do
  curl -s -o /dev/null -w "$f %{http_code}\n" \
    https://gino6178.github.io/project3/assets/$f.png
done
```

Both are worth doing every time. A push can succeed while an image is missing, because git is
perfectly happy to commit an `<img src>` pointing at a file that was never added.

## Before pushing

Two checks catch almost everything that has gone wrong so far.

**The HTML closes.** The page is hand-written and a dropped `</div>` will render as a
progressively narrower column rather than as an error:

```bash
cd /home/gino/project/project3-site
python3 -c "
import html.parser
class P(html.parser.HTMLParser):
    def __init__(s): super().__init__(); s.stack=[]; s.err=[]
    def handle_starttag(s,t,a):
        if t not in ('img','br','meta','link','hr','input'): s.stack.append(t)
    def handle_endtag(s,t):
        if s.stack and s.stack[-1]==t: s.stack.pop()
        else: s.err.append(t)
p=P(); p.feed(open('index.html').read())
print('unclosed', p.stack[:4], 'mismatched', p.err[:4])
"
# unclosed [] mismatched []
```

**Every `<img>` has a file.** Orphan references and orphan files are both worth knowing about:

```bash
python3 - <<'EOF'
import os, re
html = open('index.html').read()
used = set(re.findall(r'src="assets/([^"]+)"', html))
have = set(os.listdir('assets'))
print('referenced but missing:', sorted(used - have) or 'none')
print('present but unused:   ', sorted(have - used) or 'none')
EOF
```

## Regenerating the figures

The figures are not drawn by hand and should not be edited by hand. One script rebuilds all of
them and prints, as it goes, the numbers its captions quote:

```bash
cd /home/gino/project/fruitninja2
python3 -m cube_ovoxel.figures \
  --runs /path/to/runs \
  --out  /home/gino/project/project3-site/assets
```

Write the captions from that output, not from memory. A figure whose generating script no
longer exists cannot be checked against the code it claims to depict, and the caption drifts
from the picture without either of them looking wrong — which is exactly what happened to the
compositing figure, whose caption said "the original Gaussians" while the picture was a flat
grey silhouette for weeks.

## The MathJax on the page

Equations are written as `\(...\)` inline and `\[...\]` display, rendered by KaTeX's
auto-render loaded from the CDN in `<head>`. Two consequences:

- a backslash in the HTML source is a backslash in the maths, so nothing needs doubling;
- Markdown code spans do **not** work — the page is HTML, so use `<code>` tags. Writing
  `` `foo` `` leaves literal backticks on the published page.

## Starting a new project page

The three pages so far are the same shape, and project2 is the one to copy:

```bash
cp -r /home/gino/project/project2-site /home/gino/project/project4-site
cd /home/gino/project/project4-site
rm -rf .git assets/*
git init -b main
git remote add origin git@github.com:gino6178/project4.git
git config user.name gino6178
git config user.email gino6178@gmail.com
```

Then create the repository on GitHub as **public** (Pages is not served from a private
repository on a free account), push, and enable Pages in *Settings → Pages* with source
**"Deploy from a branch"**, branch `main`, folder `/ (root)`. The first build takes a few
minutes; every one after that is under a minute.

## What is deliberately not in the repository

Only `index.html`, `assets/*.png` and the two markdown files. No code, no weights, no training
runs, no `.pt` files. The repository is the write-up; the work lives in
`/home/gino/project/fruitninja2` and its handover folder. Keeping it that way is what makes the
repository small enough to clone in a second and safe to have public.
