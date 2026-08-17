"""Does each exterior reference show the face it is named after?

Every failure of appearance in this project so far has been traced back to a reference image
rather than to the model that consumed it, and each time it was found after a training run, by
looking at the result. This looks before.

The test needs no ground truth, because the six prompts already state what the six faces are
supposed to be. So score every reference against every prompt with CLIP and ask one question:
does an image score highest against its own prompt? A `down` reference that is really another
`up` will prefer the `up` prompt, and say so. The margin is reported too -- winning by 0.001
is not the same claim as winning by 0.05 -- and a view whose prompt is genuinely near-identical
to another's (the orange's four sides are word-for-word the same) is excluded from the test
rather than failed by it, since there is nothing there to distinguish.

The one thing it cannot catch is a face whose prompt is itself wrong, which is why the prompt
that decided each verdict is printed next to it.

Read the margin, not the verdict, and do not use this alone. Measured on the three sets in the
repository it calls the doughnut's `up` WRONG at -0.0112, and that reference is correct -- it is
the one object whose skin came out right. It also fails the watermelon's `right` and `back` at
-0.0037 and -0.0005, which is noise. CLIP does not reliably encode "seen from directly above",
so only a large negative margin means anything, and even then it is a reason to look at the
image rather than a verdict on it. The orange's `up` at -0.0311 is the one call here that
independently held up.

    python method/common/pipeline/check_refs.py REF_DIR PROMPTS.json
"""
import os as _os
_FN_ROOT = _os.environ.get("FN_ROOT", "/home/gino/project/FruitNinja_clean")

import json
import sys

import torch
from PIL import Image

sys.path += [_FN_ROOT, _FN_ROOT + "/gaussian-splatting"]

MODEL = "openai/clip-vit-base-patch32"


def main(ref_dir, prompts_json, model_name=MODEL):
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(model_name)

    prompts = json.load(open(prompts_json))
    names = [n for n in prompts if _os.path.exists(_os.path.join(ref_dir, f"{n}_ref.png"))]
    if not names:
        raise SystemExit(f"{ref_dir} has no *_ref.png matching {prompts_json}")

    with torch.no_grad():
        t = proc(text=[prompts[n] for n in names], return_tensors="pt",
                 padding=True, truncation=True).to(dev)
        tf = model.get_text_features(**t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        ims = [Image.open(_os.path.join(ref_dir, f"{n}_ref.png")).convert("RGB") for n in names]
        i = proc(images=ims, return_tensors="pt").to(dev)
        imf = model.get_image_features(**i)
        imf = imf / imf.norm(dim=-1, keepdim=True)
    S = (imf @ tf.T).cpu()                        # image x prompt

    # A prompt that is word-for-word another prompt cannot lose to it or beat it, so the faces
    # that share wording are grouped and the test asks only that the image prefers its group.
    group = {}
    for n in names:
        group.setdefault(prompts[n].strip(), []).append(n)
    of = {n: group[prompts[n].strip()] for n in names}

    bad = 0
    print(f"  {ref_dir}  ({len(names)} references)")
    for a, n in enumerate(names):
        own = [names.index(m) for m in of[n]]
        best_own = max(float(S[a, k]) for k in own)
        others = [(float(S[a, k]), names[k]) for k in range(len(names)) if k not in own]
        if not others:
            print(f"    {n:<6} no distinguishing prompt, skipped")
            continue
        best_other, who = max(others)
        ok = best_own > best_other
        bad += (not ok)
        mark = "ok  " if ok else "WRONG"
        print(f"    {n:<6} {mark} own {best_own:.4f}   best other {best_other:.4f} ({who})"
              f"   margin {best_own - best_other:+.4f}"
              + ("" if ok else f"   <- looks like `{who}`"))
    print(f"  {len(names) - bad}/{len(names)} references prefer their own face")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main(sys.argv[1], sys.argv[2]) else 0)
