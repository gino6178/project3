import torch
import os
import math
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

class ImageProcessor:
    @staticmethod
    def pil_to_numpy(images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        # Ensure images are in RGB mode
        images = [image.convert('RGB') for image in images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)  # Shape: (N, H, W, C)
        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        if images.ndim == 3:
            images = images[None, ...]  # Add batch dimension
        images = images.transpose(0, 3, 1, 2)  # Shape: (N, C, H, W)
        images = torch.from_numpy(images)
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = images.cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # Shape: (N, H, W, C)
        images = np.clip(images, 0, 1)  # Ensure values are in [0, 1]
        # Remove multiplication by 255 here
        return images  # Values remain in [0, 1]

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> Union[List[Image.Image], Image.Image]:
        if images.ndim == 3:
            images = images[None, ...]  # Add batch dimension
        # Multiply by 255 only once here
        images = (images * 255).astype(np.uint8)
        # Ensure images have the correct color mode
        pil_images = [Image.fromarray(image, mode='RGB') for image in images]
        if len(pil_images) == 1:
            return pil_images[0]
        else:
            return pil_images

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        print("1")
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        print("2")
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        print("3")
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def return_img(local_latent, pipe):
    local_latent = local_latent.clone().detach()
    local_latent.requires_grad_(False)
    image_local = pipe.decode_latents(local_latent)
    return ImageProcessor.numpy_to_pil(image_local)

def return_img(local_latent, pipe):
    local_latent = local_latent.clone().detach()
    local_latent.requires_grad_(False)
    image_local = pipe.vae.decode(local_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image_local = image_local.detach()
    return pipe.image_processor.postprocess(image_local)[0]

_CANON = {}
_PHOTOS = {}
# Give the whole transverse family one photograph, so its planes agree about where the
# segment walls are. See the note in one_step_sds_orange.
# How the transverse family agrees about angle. "" leaves each plane with its own photograph,
# which is what produced walls that decorrelate in six depths. "share" gives the family one
# photograph rescaled into each disc, which is exact for an object whose transverse structure is
# purely axial -- a citrus's segment walls -- and wrong for one with localised features, since
# everything local is then extruded and a watermelon grows columns of seed. "align" keeps each
# photograph and rotates it to the family's phase, which preserves the local content and only
# partly agrees, because photographs of different specimens have different numbers of walls and
# no rotation reconciles ten with twelve.
# (27) by default, which falls through to (11) when nothing has been solved for a
# reference set. "align" forces (11), "share" the single shared phase, "" the raw
# photograph at whatever angle it was taken.
REF_PHASE_MODE = os.environ.get("REF_PHASE_MODE", "solve")


# Which plane is being supervised, and how many there are. Set by the training loop before
# each reference is fetched; left at the default it reproduces the released behaviour.
_PLANE = {"idx": 0, "n": 1}


def set_plane(idx, n=1):
    _PLANE["idx"], _PLANE["n"] = int(idx), max(int(n), 1)


# Where the run is, so the reference can change with it. Same module-global idiom as
# _PLANE: the two call sites in the trainer already pass their state this way, and
# threading an iteration number through one_step_sds_orange's signature would touch
# every caller including the released trainer.
_ITER = {"j": 0, "total": 3000}


def set_iter(j, total):
    _ITER["j"], _ITER["total"] = int(j), max(int(total), 1)



# --- when to regenerate -------------------------------------------------------------------
#
# The paper regenerates "until the reconstruction losses for all slices converge below a
# predefined threshold". An absolute threshold has to be chosen per object, per resolution and
# per point count, so in practice it becomes a fixed interval and the fixed interval is wrong:
# FruitNinja's twenty iterations are enough for 7.96M free primitives and not for 826k lattice
# cells, which at iteration twenty still carry 34.7% of their initial error and are still
# improving at seventy.
#
# So gate on the *rate*, which is dimensionless. With r_j the mean per-plane residual and
#
#     rbar_j = (1 - a) rbar_{j-1} + a r_j                      (a = REF_CONV_EMA)
#     g_j    = (rbar_{j-W} - rbar_j) / rbar_{j-W}              (W = REF_CONV_WINDOW)
#
# g_j is the fraction of the error still present W iterations ago that the last W iterations
# removed. Regenerate when g_j < REF_CONV_TAU, which reads "this reference has stopped paying"
# and needs no per-object number. The EMA is there because a single iteration's residual moves
# a few percent on crop sampling alone, and one lucky window would otherwise trigger it.
_CONV = {"ema": None, "hist": [], "last": -10 ** 9, "fire_iter": None,
         "fired_any": False}


def note_residual(j, r):
    """Record one iteration's mean residual and decide whether the next one regenerates.

    The decision is taken here, once per iteration, and latched -- not inside `past_warmup`,
    which every plane calls separately. A test that mutates state cannot live there: the first
    plane of the iteration would consume the trigger and the other twenty-five would be
    supervised against the old target, which is a half-regenerated run and worse than either.
    """
    a = float(os.environ.get("REF_CONV_EMA", "0.3"))
    _CONV["ema"] = r if _CONV["ema"] is None else (1.0 - a) * _CONV["ema"] + a * r
    _CONV["hist"].append((j, _CONV["ema"]))
    if float(os.environ.get("REF_CONV_TAU", "0")) > 0 and converged_enough(j):
        _CONV["fire_iter"] = j + 1
        _CONV["fired_any"] = True


def converged_enough(j):
    """Has the fit to the current reference stopped improving fast enough to be worth keeping?"""
    w = int(os.environ.get("REF_CONV_WINDOW", "10"))
    tau = float(os.environ.get("REF_CONV_TAU", "0.02"))
    warm = int(os.environ.get("REF_WARMUP", "0"))
    if tau <= 0 or len(_CONV["hist"]) <= w or j - _CONV["last"] < w or j < warm:
        return False
    past = _CONV["hist"][-1 - w][1]
    now = _CONV["hist"][-1][1]
    if past <= 0:
        return False
    g = (past - now) / past
    # Converged means the residual has stopped falling, not that it is rising. A negative g is
    # the transient after the target was last replaced: the new reference is a different image,
    # so the residual jumps, and "improved by less than tau" is then satisfied trivially. Left
    # unguarded that is self-sustaining -- regenerate, residual jumps, regenerate again -- and
    # it fired at 87, 97, 107, 117 with g of -2%, -146%, -17%, which is the failure the whole
    # criterion was meant to replace, now produced by the criterion itself.
    #
    # So require the fit to be at the bottom of its own curve since the last regeneration, not
    # merely flat: the residual must be within delta of the best it has reached since then.
    since = [v for (k, v) in _CONV["hist"] if k > _CONV["last"]]
    floor = min(since) if since else now
    delta = float(os.environ.get("REF_CONV_DELTA", "0.05"))
    if 0.0 <= g < tau and now <= (1.0 + delta) * floor:
        print(f"  converged j={j}: last {w} iterations removed {100 * g:.2f}% of the residual "
              f"(threshold {100 * tau:.1f}%), at {now / floor:.3f} of its best since the last "
              f"regeneration, regenerating", flush=True)
        _CONV["last"] = j
        return True
    return False


def past_warmup():
    """Has the photograph handed over to the released target yet?

    The two ways of making a target fail in opposite directions. A photograph carries interior
    structure that diffusion of the model's own render cannot invent, but it is pasted in as a
    stencil -- the render supplies only the disc to paste into -- so nothing in the loss
    couples neighbouring cells and the field keeps a layer of per-cell noise. The released
    target is the opposite: it is the model's own render refined, so it cannot add anything,
    but it is self-consistent by construction and it converges.

    So the photograph runs only long enough to place the structure, and everything after it is
    the released method unchanged -- same path, same annealing, same interval, same settings.
    That also makes the comparison clean: one difference, stated as one number.
    """
    # Default past any run length: under REF_PHOTO the photograph is the target for the
    # whole run, which is what the three finished objects were trained with. The handover
    # is opt-in because the loop it hands over to hollowed the interior out -- 71% of cells
    # below opacity 0.5 -- see notes/closed-loop-target.md.
    warm = int(os.environ.get("REF_WARMUP", "10000000"))
    # Under the convergence gate the warmup is only a floor. Which iteration regenerates was
    # decided at the end of the previous one and latched, so this is a read and every plane in
    # that iteration sees the same answer.
    if float(os.environ.get("REF_CONV_TAU", "0")) > 0 and warm < 10 ** 9:
        return _ITER["j"] >= warm and _ITER["j"] == _CONV.get("fire_iter")
    return _ITER["j"] >= warm


def sampling_strength():
    """img2img strength for this regeneration, annealed to nothing under REF_ANNEAL.

    A closed loop at constant strength diverges, and it took a run to see it. Each
    regeneration takes the current render and pushes it along whatever direction the prompt
    names -- "detailed cross section" -- the model fits that, and the next regeneration starts
    from the pushed position and pushes again. There is no fixed point, and the measurement
    says so plainly: over thirty iterations past the handover the gradient rose 78% and the
    angular contrast reached three times its warmup value, but the speckle rose 53% alongside
    them. Structure and noise amplifying together is not refinement. The renders end up a dark
    over-saturated orange webbed with invented white veining, and each target is more extreme
    than the render it came from.

    One sample cannot show this -- it is what accumulates over dozens of regenerations -- which
    is why the still-image check at 0.30 looked clean and was.

    The released method never meets this because its own annealing, 30 - j//100, drives its
    target to identity. Same shape here: the handover strength is what was chosen and
    verified, and every regeneration after it is weaker, so the loop gain decays and the fixed
    point exists.
    """
    s0 = float(os.environ.get("REF_STRENGTH", "0.6"))
    if os.environ.get("REF_ANNEAL", "0") != "1":
        return s0
    warm = int(os.environ.get("REF_WARMUP", "0"))
    j, total = _ITER["j"], _ITER["total"]
    t = min(max((j - warm) / max(total - warm, 1), 0.0), 1.0)
    return max(s0 * 0.5 * (1.0 + math.cos(math.pi * t)), 0.02)


def _sds_prompt(view_cut):
    """The prompt the section sampler is conditioned on, for this cut direction.

    The default described a navel orange, in detail, and nothing in a run had to override it.
    A watermelon run that turned the reference regeneration on therefore had its references
    resampled as oranges -- and came back with an orange interior, radial membranes and all,
    scoring *better* on a texture measure than the released watermelon while being the wrong
    fruit. The default stays, because the orange is what most of this work is, but it is now
    one line to change and the change is visible in the run script.

    `{view_cut}` in the environment value is substituted, so one string covers both families
    and can carry the phrase a DreamBooth fine-tune was bound to. That binding is the whole
    point of fine-tuning: a checkpoint trained on "the horizontal cross-sectional view of a
    watermelon" contributes nothing to a prompt that never says it.
    """
    p = os.environ.get("SDS_PROMPT", "")
    if p:
        return p.replace("{view_cut}", view_cut)
    return (f"macro photo of the {view_cut} cross section of a navel orange, "
            f"white central pith core, white radial segment membranes dividing the "
            f"wedges, translucent orange juice vesicles, white spongy albedo under "
            f"the peel, detailed")


def _photos_in(spec):
    """The colour photographs in a reference directory, and nothing else.

    This took every file the directory held. Generating depth maps beside the photographs --
    `orange1.png` next to `orange1_depth.png` -- therefore doubled the reference set with
    greyscale images, and because the names sort adjacent, every second plane was supervised
    against a depth map as though it were a colour target. The section came out in desaturated
    horizontal bands, one per alternating plane, and the run that produced it looked like a
    training failure rather than a directory listing.

    Anything that is not a photograph of a cross-section does not belong in the reference set,
    so require an image extension and drop the derived maps by name.
    """
    import glob
    return [p for p in glob.glob(os.path.join(spec, "*"))
            if os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg", ".webp")
            and not os.path.splitext(os.path.basename(p))[0].endswith(("_depth", "_mask",
                                                                       "_alpha", "_normal"))]


def _photo(spec):
    """One real cross-section photograph, chosen for the plane being supervised.

    This used to return files[0] and nothing else, so all forty-eight planes were matched
    against a single image and the other nineteen in the directory were never opened. Two
    consequences, both measured. The colour came from whichever photograph happened to sort
    first, and for the watermelon that is the darkest of the twenty -- flesh at (0.81, 0.14,
    0.17) against the others' 0.86 to 0.98 -- so the model converged to a red darker than any
    other melon in the set while faithfully fitting the one it was shown. And after about
    five iterations there was nothing left to learn: twenty-six planes aligned to one image
    carry one image's worth of information, and the held-out score sat flat from there.

    Spreading the photographs across the planes gives each depth its own target, which is
    also what the object is: sections at different depths do not look alike.
    """
    import glob
    files = sorted(_photos_in(spec)) if os.path.isdir(spec) else [spec]
    k = (_PLANE["idx"] * len(files) // _PLANE["n"]) % len(files)
    # The continuous assignment, which is equation (14) of the paper. The rule above is
    # piecewise constant -- integer division gives two or three adjacent planes the same
    # photograph, so the interior has no reason to differ between them and every reason to
    # change where the block does. Continuous instead: a plane at t = j M_f / N_f takes
    # photograph floor(t) and the next, both brought onto a common disc by _blend_canonical,
    # mixed at the fractional part.
    #
    # It is deterministic. A plane still sees one fixed target for the whole run, so a cell
    # shared by two families meets the same conflict on every pass -- which is what makes the
    # assignment worth optimising at all, and what REF_RANDOM_ASSIGN gives up.
    #
    # On by default, which is what the page has always said this pipeline does. It was
    # implemented, measured, and absent from this file, so every model in the repository until
    # now was trained by the block rule instead. REF_DEPTH_BLEND=0 restores that rule, which is
    # what the numbers measured before this change were produced under.
    if os.environ.get("REF_DEPTH_BLEND", "1") == "1" and len(files) > 1:
        t = _PLANE["idx"] * len(files) / max(_PLANE["n"], 1)
        k0 = int(t) % len(files)
        k1 = (k0 + 1) % len(files)
        w = float(t - int(t))
        key = (spec, "blend", k0, k1, round(w, 3))
        if key not in _PHOTOS:
            _PHOTOS[key] = _blend_on_disc(files[k0], files[k1], w)
        return _PHOTOS[key]
    key = (spec, k)
    if key not in _PHOTOS:
        _PHOTOS[key] = Image.open(files[k]).convert("RGB")
    return _PHOTOS[key]


def _blend_canonical(path, size=512, frac=0.38):
    """One photograph with its section centred and scaled to a fixed radius.

    Named _blend_canonical, not _canonical: this file already has a _canonical with a different
    signature, and defining a second one silently replaced it.

    Two references cannot be blended as they arrive: they are different fruits, framed by hand, and
    their discs differ in centre and radius. Overlaying them directly leaves a ring where one disc
    reaches and the other does not, which the section loss would then try to reproduce.
    """
    import numpy as np
    im = Image.open(path).convert("RGB")
    cy, cx, r = _disc(im)
    scale = (frac * size) / max(r, 1e-6)
    w, h = im.size
    im2 = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    out = Image.new("RGB", (size, size), (255, 255, 255))
    out.paste(im2, (int(round(size / 2 - cx * scale)), int(round(size / 2 - cy * scale))))
    return np.asarray(out, np.float32) / 255.0


def _blend_on_disc(pa, pb, w):
    """(1-w) of one reference and w of the next, both on a common disc."""
    import numpy as np
    a, b = _blend_canonical(pa), _blend_canonical(pb)
    m = np.clip((1.0 - w) * a + w * b, 0, 1)
    return Image.fromarray((m * 255).astype("uint8"))


def _disc(img):
    """Centre and radius of the object against its own border colour."""
    import numpy as np
    from scipy import ndimage
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.
    h, w, _ = a.shape
    k = max(2, h // 20)
    bg = np.median(np.concatenate([a[:k].reshape(-1, 3), a[-k:].reshape(-1, 3)]), 0)
    m = ndimage.binary_fill_holes(np.abs(a - bg).max(2) >= 0.10)
    if m.sum() < 100:
        return (h / 2, w / 2, h / 4)
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return (cy, cx, float(np.percentile(rr, 98)))


def _canonical(pipe, image, depth, view_cut):
    """Sample the shared cross-section once and keep it for every later view."""
    import torch as _t
    if "img" in _CANON:
        return _CANON["img"]
    prompt = _sds_prompt(view_cut)
    g = _t.Generator(pipe.device).manual_seed(int(os.environ.get("REF_SEED", "1234")))
    r = pipe(prompt=prompt, image=image, depth_map=depth,
             negative_prompt="watermark, text, letters, logo, signature, border",
             strength=float(os.environ.get("REF_STRENGTH", "0.6")),
             guidance_scale=10, num_inference_steps=50, generator=g, return_dict=False)
    _CANON["img"] = r[0][0] if isinstance(r, tuple) else r.images[0]
    return _CANON["img"]


def _angular_profile(img, nb=360, r_lo=0.25, r_hi=0.80):
    """Mean brightness as a function of angle, over the annulus the walls live in.

    The walls are the only thing in a transverse section that is a function of angle alone, so
    averaging over radius inside an annulus isolates them from the pith at the centre and the
    peel at the rim.
    """
    import numpy as np
    a = np.asarray(img.convert("L"), dtype=np.float32) / 255.
    m = a < 0.98
    if m.sum() < 400:
        return None
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    R = np.sqrt(m.sum() / np.pi)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    rr = np.hypot(yy - cy, xx - cx) / R
    th = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)
    sel = m & (rr > r_lo) & (rr < r_hi)
    if sel.sum() < 400:
        return None
    b = (th[sel] * nb).astype(int) % nb
    prof = np.bincount(b, a[sel], minlength=nb) / np.maximum(np.bincount(b, minlength=nb), 1)
    prof = prof - prof.mean()
    n = np.linalg.norm(prof)
    return prof / n if n > 1e-9 else None


_PHASE = {}


def _shared_phase(spec):
    """The family's first photograph, for every plane in it."""
    files = sorted(_photos_in(spec)) if os.path.isdir(spec) else [spec]
    key = (spec, 0, "share")
    if key not in _PHASE:
        _PHASE[key] = Image.open(files[0]).convert("RGB")
    return _PHASE[key]


def _solved(spec):
    """The phases and the assignment equation (27) chose, if they have been solved for.

    (11) sets each transverse phase by cross-correlating that photograph against a fixed member of
    its own family. That is greedy twice over: it never consults the longitudinal family, whose
    planes cross every one of these, and it never revisits a choice. (27) minimises the two
    families' disagreement on the lines they share, over the phases and over which photograph is
    shown at which depth, before a gradient is taken -- on the orange it takes the chordal cost
    from 0.2667 to 0.2362.

    Returned only when the directory still holds the photographs it was solved over. A reference
    set that has gained or lost an image since silently invalidates a permutation, and a silently
    wrong assignment is the failure this whole area keeps producing.
    """
    import numpy as np
    f = os.path.join(spec, "phase_opt.npz") if os.path.isdir(spec) else ""
    if not f or not os.path.isfile(f):
        return None
    z = np.load(f, allow_pickle=False)
    now = [os.path.basename(p) for p in sorted(_photos_in(spec))]
    was = [str(x) for x in z["files"]]
    if now != was:
        print(f"  phase_opt.npz is stale for {spec}: {len(was)} photographs solved, "
              f"{len(now)} present -- falling back to the greedy alignment")
        return None
    return z["phases"], z["perm"]


def _phase_aligned(spec):
    """This plane's own photograph, turned to the family's angular phase.

    The first photograph in the directory defines the phase; every other is rotated by the shift
    that maximises the circular cross-correlation of its angular profile with that one. Rotation
    is the only degree of freedom used, so nothing about a photograph is invented or discarded --
    the seeds, the pith and the flesh it happens to show all survive, at a different angle.
    """
    import numpy as np
    files = sorted(_photos_in(spec)) if os.path.isdir(spec) else [spec]
    k = (_PLANE["idx"] * len(files) // _PLANE["n"]) % len(files)
    key = (spec, k, "phase")
    if key in _PHASE:
        return _PHASE[key]

    ref_key = (spec, 0, "phase")
    ref = _PHASE.get(ref_key)
    if ref is None:
        ref = Image.open(files[0]).convert("RGB")
        _PHASE[ref_key] = ref
    if k == 0:
        return ref

    img = Image.open(files[k]).convert("RGB")
    pr, pi = _angular_profile(ref), _angular_profile(img)
    if pr is None or pi is None:
        _PHASE[key] = img
        return img
    # circular cross-correlation, by FFT
    cc = np.fft.irfft(np.fft.rfft(pr) * np.conj(np.fft.rfft(pi)), n=len(pr))
    shift = int(np.argmax(cc))
    deg = 360.0 * shift / len(pr)
    out = img.rotate(-deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    _PHASE[key] = out
    return out


def _solved_photo(spec):
    """This plane's photograph under equation (27): its assignment, at its solved phase.

    Falls through to (11) when nothing has been solved for this reference set, so a run that has
    not had `stage_phases` still trains rather than failing, and says which it used.
    """
    import numpy as np
    got = _solved(spec)
    if got is None:
        return _phase_aligned(spec)
    phases, perm = got
    files = sorted(_photos_in(spec)) if os.path.isdir(spec) else [spec]
    k = (_PLANE["idx"] * len(files) // _PLANE["n"]) % len(files)
    k = int(perm[k]) if k < len(perm) else k
    key = (spec, k, "solved")
    if key in _PHOTOS:
        return _PHOTOS[key]
    deg = float(np.degrees(phases[k])) if k < len(phases) else 0.0
    img = Image.open(files[k]).convert("RGB")
    if abs(deg) > 1e-6:
        img = img.rotate(-deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    _PHOTOS[key] = img
    return img


def _fit_disc(canon, target):
    """Rescale the canonical section so its disc matches the target view's disc."""
    import numpy as np
    ccy, ccx, cr = _disc(canon)
    tcy, tcx, tr = _disc(target)
    s = max(tr / max(cr, 1e-6), 1e-3)
    nw, nh = max(int(canon.width * s), 8), max(int(canon.height * s), 8)
    sc = canon.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", target.size, (255, 255, 255))
    out.paste(sc, (int(round(tcx - ccx * s)), int(round(tcy - ccy * s))))
    return out


def _sample(pipe, image, depth, view_cut):
    """One img2img draw: the released path's sampler, named so the anchor can use it too."""
    prompt = _sds_prompt(view_cut)
    # One seed for every view. Sampling each cross-section independently gives references
    # that are individually good and mutually contradictory: their angular profiles correlate
    # -0.09 across views, so the radial membranes sit at different angles in each one and the
    # 3D model can only converge to their average.
    g = torch.Generator(pipe.device).manual_seed(int(os.environ.get("REF_SEED", "1234")))
    r = pipe(prompt=prompt, image=image, depth_map=depth,
             negative_prompt="watermark, text, letters, logo, signature, border",
             strength=sampling_strength(), guidance_scale=10, num_inference_steps=50,
             generator=g, return_dict=False)
    return r[0][0] if isinstance(r, tuple) else r.images[0]


def _blend(a, b, w, mode=None):
    """b mixed into a by weight w.

    `pixel` is a straight average and costs exactly what the anchor was added to protect. The
    two images carry their detail in different places -- a is what the sampler made of this
    object, b is a photograph of another one -- so averaging them cancels the high frequencies
    while the low ones, which are aligned, survive. Measured on the run this was written for:
    the sampled reference carries 85.1e-3 of detail against the photograph's 70.7, and their
    half-and-half average carries 52.5. The anchor was holding the colour and throwing away
    the detail it was supposed to be protecting.

    `colour` blends the low frequencies only and puts a's own detail back on top. Drift is a
    low-frequency phenomenon -- the reference goes muddy and desaturated, not grainy -- so
    that is the band the restoring force belongs in.
    """
    import numpy as _np
    from PIL import Image as _I
    import cv2 as _cv
    x = _np.asarray(a).astype(_np.float32)
    y = _np.asarray(b).astype(_np.float32)
    if (mode or os.environ.get("REF_ANCHOR_MODE", "pixel")) == "colour":
        k = int(os.environ.get("REF_ANCHOR_SIGMA", "12"))
        lx = _cv.GaussianBlur(x, (0, 0), k)
        ly = _cv.GaussianBlur(y, (0, 0), k)
        out = (x - lx) + (lx * (1.0 - w) + ly * w)
    else:
        out = x * (1.0 - w) + y * w
    return _I.fromarray(out.clip(0, 255).astype(_np.uint8))



def _lock_shared_axis(gen, render, view_cut):
    """Give back the line every vertical section shares, from what the volume already holds.

    The ten vertical training planes all contain the object's axis, so that axis is the same line
    in all ten references and the model draws it ten times without knowing that. Measured on the
    stored references it disagrees with itself by 0.113 there, while a single section varies along
    the same line by 0.091: the conflict between the references is larger than the structure inside
    one of them, and the volume can only average it away afterwards.

    The complementary fix to sharing the initial noise, which does this for horizontal sections by
    making their angular patterns agree. Here the sampled section keeps everything except a
    feathered band about the axis, which is restored from the render -- that is, from what the
    other nine sections have already agreed on through the cells they share. On the stored
    references a band of eight pixels removes 54 to 58% of the disagreement before a gradient is
    taken, and what is left is the render's own, which is the point: the supervision stops fighting
    the volume about a line the volume has already decided.

    Off by default. It changes what the supervision is, so a run with it and a run without it are
    different experiments and the paper reports which is which.
    """
    import numpy as np
    from PIL import Image
    half = int(os.environ.get("REF_AXIS_LOCK", "0"))
    if half <= 0 or view_cut != "vertical":
        return gen
    g = np.asarray(gen.convert("RGB"), np.float32)
    r = np.asarray(render.convert("RGB").resize(gen.size), np.float32)
    W = g.shape[1]
    x = np.arange(W)
    w = np.clip(1.0 - np.abs(x - W // 2) / float(half), 0.0, 1.0)[None, :, None]
    return Image.fromarray(np.clip(g * (1 - w) + r * w, 0, 255).astype(np.uint8))


def one_step_sds_orange(image, depth, total_epochs, pipe, view_cut):
    # SD-2-depth operates at 512x512 and its VAE downsamples 8x. When the caller
    # renders below that (ablation runs use 256 for speed) the latent is too small
    # for the UNet to synthesise structure and the reference decodes to colour mush.
    # Upscale for generation, hand the result back at the caller's size. Applied
    # identically in every ablation variant, so it cannot bias the comparison.
    _orig = image.size
    if _orig != (512, 512):
        image = image.resize((512, 512))

    # The reference this returns is only ever used as an MSE/SSIM target, so it does not
    # have to come from SDS -- and SDS is the reason it has no structure. Averaging the
    # score over timesteps drawn uniformly from t_range washes out high frequencies: the
    # SDS reference is a smooth pink disc with a single white centre at 30 or 60 epochs
    # and under either prompt, while ordinary img2img sampling of the same model, same
    # prompt and same depth condition returns the radial segment membranes a real orange
    # cross-section has. Sample properly for the target.
    _pdir = os.environ.get("REF_PHOTO", "")
    if _pdir and view_cut == "vertical" and os.environ.get("REF_PHOTO_V", ""):
        _pdir = os.environ["REF_PHOTO_V"]
    elif _pdir and view_cut != "horizontal":
        _pdir = ""
    if _pdir:
        # Align the family's photographs to one angular phase, rather than sharing one.
        #
        # `_photo` spreads the collected photographs across the planes so each depth gets its
        # own, which is right about one thing and wrong about another. Sections at different
        # depths really do differ -- in the radius of the disc, in how much pith shows, and in
        # where the seeds are -- and they do not differ in the *angles* of the segment walls,
        # because a wall is a surface running the length of the fruit and every transverse cut
        # meets the same ones. Handing adjacent depths photographs of different specimens asks
        # them for unrelated angular patterns, and a lattice with one colour per cell answers
        # with their average.
        #
        # Measured on a trained model: the angular profile of one transverse section correlates
        # +0.448 with the next depth and +0.077 six depths away, which is where two photographs
        # of different oranges sit (+0.034). The walls were not a surface; they were a per-plane
        # decoration.
        #
        # Sharing one photograph across the family fixes the phase and breaks something else:
        # everything localised is then extruded along the axis, and a watermelon grows columns
        # of seed rather than seeds. So keep each plane's own photograph and rotate it to the
        # family's phase. What lines up is the part that should -- the angular profile, which is
        # what the walls are -- and what stays particular to each photograph is everything the
        # profile averages over.
        if REF_PHASE_MODE and view_cut == "horizontal":
            _p = (_shared_phase(_pdir) if REF_PHASE_MODE == "share"
                  else _solved_photo(_pdir) if REF_PHASE_MODE == "solve"
                  else _phase_aligned(_pdir))
        else:
            _p = _photo(_pdir)
        # Neither strength setting works *as the only target*: at 0.1 the generation stays
        # registered to the render but adds no structure, and at 0.6 it invents the structure
        # and also invents a three-quarter viewpoint with visible rind thickness, so the MSE
        # compares pixels belonging to different parts of the fruit and the model can only
        # fit their average. The repo already ships flat top-down photographs of real
        # cross-sections -- the same geometry the h-views render -- so those carry the
        # structure, and diffusion is kept out of the structure question.
        #
        # It is not kept out of the *coherence* question. The photograph runs for the warmup
        # only, long enough to place the structure, and then this returns nothing: control
        # falls through to the released path below, with its own annealing and its own
        # settings, exactly as the released trainer runs it.
        _fitted = _fit_disc(_p, image)
        if not past_warmup():
            return _fitted.resize(_orig) if _fitted.size != _orig else _fitted
        # Past the warmup the target is regenerated from the model's own render, which is what
        # stops the transverse and longitudinal families pulling against each other: both are
        # then derived from one 3-D state rather than from two fixed images that were never
        # asked to agree.
        #
        # On its own that loop has no fixed point. Its change has a persistent direction --
        # toward whatever the prompt calls detail -- so halving the strength halves the speed
        # along the same path and nothing pulls back: iterated twelve times the speckle goes up
        # ninefold at 0.30 and still ninefold at 0.15, under a neutral prompt as readily as a
        # directed one, and with a correct depth condition as readily as a wrong one.
        #
        # REF_ANCHOR supplies the restoring force by blending the generated target back toward
        # the photograph, which is the fixed reference the whole run is supposed to agree with.
        # The loop keeps its consistency and gains a fixed point. Offline, twelve iterations
        # reach 9.2x the starting speckle at anchor 0, 7.4x at 0.25 and 5.5x at 0.50, the last
        # of those still rising by only 0.2 over the final four steps -- and that test assumes
        # the model reproduces its target exactly, which a lattice of fixed cells cannot.
        # Which sampler runs past the warmup is REF_SAMPLING's to decide, exactly as it is
        # when REF_PHOTO is unset. This called _sample unconditionally, so REF_SAMPLING=0
        # selected SDS everywhere except here and a run asking for SDS silently got img2img
        # at REF_STRENGTH -- 0.6 by default, the value measured to resample rather than
        # refine. Falling through when REF_SAMPLING is off puts the choice back where it
        # belongs; the anchor stays on the img2img path, which is the one it was measured on.
        if os.environ.get("REF_SAMPLING", "1") == "1":
            _anchor = float(os.environ.get("REF_ANCHOR", "0"))
            _gen = _sample(pipe, image, depth, view_cut)
            _out = _blend(_gen, _fitted, _anchor) if _anchor > 0 else _gen
            return _out.resize(_orig) if _out.size != _orig else _out

    if os.environ.get("REF_CANONICAL", "0") == "1" and view_cut == "horizontal":
        # An orange's segment walls run vertically, so every horizontal section shows the
        # same angular pattern at a different radius. Sampling each section on its own
        # leaves the membranes at unrelated angles (cross-view angular correlation +0.09,
        # +0.33 once the seed is shared) and the 3D model averages them into flat colour.
        # Generate the pattern once and rescale it into each section's disc instead:
        # correlation +0.999, and every view now asks for the same 3D structure.
        _c = _canonical(pipe, image, depth, view_cut)
        _out = _fit_disc(_c, image)
        return _out.resize(_orig) if _out.size != _orig else _out

    if os.environ.get("REF_SAMPLING", "1") == "1":
        _prompt = _sds_prompt(view_cut)
        # One seed for every view. Sampling each cross-section independently gives
        # references that are individually good and mutually contradictory: their
        # angular profiles correlate -0.09 across views, so the radial membranes sit at
        # different angles in each one and the 3D model can only converge to their
        # average, which is the uniform orange we measured (27 white voxels in 559k).
        # Sharing the initial noise raises that correlation to +0.52 -- an orange's
        # segment walls run vertically, so every horizontal section should show the same
        # angular pattern anyway.
        _g = torch.Generator(pipe.device).manual_seed(
            int(os.environ.get("REF_SEED", "1234")))
        _r = pipe(prompt=_prompt, image=image, depth_map=depth,
                  negative_prompt="watermark, text, letters, logo, signature, border",
                  strength=sampling_strength(),
                  guidance_scale=10, num_inference_steps=50,
                  generator=_g, return_dict=False)
        _img = _r[0][0] if isinstance(_r, tuple) else _r.images[0]
        _img = _lock_shared_axis(_img, image, view_cut)
        return _img.resize(_orig) if _img.size != _orig else _img
    depth = depth.to(pipe.device)
    cur_t = [0.02, 0.98]
    clip = 1
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor = image_tensor.to(pipe.device)
    init_latents = pipe.vae.encode(image_tensor)
    init_latents = retrieve_latents(init_latents)
    init_latents = pipe.vae.config.scaling_factor * init_latents
    init_latents = init_latents.detach().clone().requires_grad_(True)
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.1)  # Choose a learning rate
    for e in range(total_epochs):
        optimizer.zero_grad()
        step_ratio = min(1, e / total_epochs)
        # The released prompt names the fruit and nothing else, and SD2 answers with a
        # uniformly pink pulp: the generated references carry a small white centre but no
        # radial segment membranes, so the model cannot learn a structure its guidance
        # never shows. Naming the parts an orange cross-section actually has puts them
        # back. (strength= below is inert here -- get_sds_latent samples its timestep from
        # t_range and the get_timesteps call that would use strength is commented out.)
        _prompt = _sds_prompt(view_cut)
        grad = pipe.get_sds_latent(
            _prompt,
            image=image_tensor,
            depth_map=depth,
            strength=0.1,
            num_inference_steps=100,
            init_latents=init_latents,
            t_range = cur_t,
            guidance_scale=10,
            step_ratio=None
        )
        grad.clamp(-clip, clip)
        target = (init_latents - grad).detach()
        loss = 0.5 * F.mse_loss(init_latents.float(), target, reduction='sum') / init_latents.shape[0]
        loss.backward()
        optimizer.step()
    _r = return_img(init_latents, pipe)
    if _r.size != _orig:
        _r = _r.resize(_orig)
    return _r


_EXT = {}


def exterior_ref(image, depth, pipe, prompt, seed=1234, strength=None, fresh=False):
    """A reference for the outside of the object, sampled once and reused per view.

    The cross-section branch is supervised against photographs or generated sections;
    the exterior branch is supervised against a render of the model it started from, which
    preserves whatever appearance that model already had. That works when the input was a
    scan. When the shell is generated instead, the input has no appearance -- ours starts
    flat grey and is coloured by the cross-section photograph's outer ring, which is the
    cut rim, not the intact peel -- and the branch then holds the wrong colour in place for
    the whole run, because no other view ever sees the outside.

    Sampling the exterior the same way the sections are sampled closes that gap: the depth
    map registers the generation to the silhouette actually being rendered, so the result
    is a peel on this object rather than a picture of a fruit.

    One sample serves every view. Sampling per view gives references that are individually
    plausible and mutually inconsistent, which is the same failure the cross-section branch
    hit -- their angular profiles correlated -0.09 across views, and the model can only fit
    the average.
    """
    import torch as _t
    orig = image.size
    if orig != (512, 512):
        image = image.resize((512, 512))
    # A generated shell starts flat grey, so unlike the cross-section branch there is no
    # appearance to preserve and the sampler has to supply all of it; the depth map is what
    # holds the result on this silhouette, not the input colours.
    if strength is None:
        strength = float(os.environ.get("EXT_STRENGTH", "0.95"))
    # `fresh` resamples instead of reusing. The cache is right for the one-shot use this was
    # written for -- one sample, refitted per view, so every view shows the same fruit -- and
    # wrong for refinement, where the point is that the input has changed. Without it
    # EXT_REF_INTERVAL was inert: three regenerations returned the identical image, colour and
    # gradient equal to three decimals.
    if fresh or "img" not in _EXT:
        g = _t.Generator(pipe.device).manual_seed(int(seed))
        # The sampler's prior for a fruit against a disc-shaped depth map is a cut one:
        # under a short negative ("cross section, cut, sliced") at guidance 10 every view
        # came back as a cross-section, for both fruits. Naming the parts a cut fruit shows
        # -- halved, wedge, segments, pulp, flesh, seeds -- and raising guidance to 12 gives
        # an intact fruit under all four prompts tried, and the peel's dimpling with it.
        r = pipe(prompt=prompt, image=image, depth_map=depth,
                 negative_prompt=os.environ.get("EXT_NEG",
                     "cross section, cut in half, sliced, halved, wedge, segments, pulp, "
                     "interior, flesh, seeds, watermark, text"),
                 strength=float(strength), guidance_scale=12, num_inference_steps=50,
                 generator=g, return_dict=False)
        _new = r[0][0] if isinstance(r, tuple) else r.images[0]
        if not fresh:
            _EXT["img"] = _new
    else:
        _new = _EXT["img"]
    # A refinement is already registered to this view, so fitting it to the disc again would
    # only resample it; the cached one-shot still needs placing.
    out = _new if fresh else _fit_disc(_EXT["img"], image)
    return out.resize(orig) if out.size != orig else out
