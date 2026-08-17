"""A material per cell, read off the lattice the appearance already lives on.

A fruit is not one material. Peel resists and flesh gives, and a simulation that gives both the
same Young's modulus produces the one behaviour neither of them has -- a uniform jelly that
wobbles as a whole. The usual reason to accept that is that there is nothing in the
representation to attach a second material to: a filled point cloud carries a colour per
primitive and no statement about what any of them *is*.

The lattice already carries both things this needs. Which level a cell belongs to says whether
it is skin or interior, which is a boundary that was measured from the object's own radial
density rather than assumed. And the cell's decoded colour says what the interior is made of
where the interior is not uniform -- rind, pith and flesh are different colours in every object
in the dataset, which is the whole premise of generating the interior in the first place.

So the field here is a rule, not a network: level decides skin from interior, and colour decides
what kind of interior. That is deliberately the weakest version. It costs nothing, it is
inspectable, and if a learned head is worth having later then this is what it has to beat.

The numbers are order-of-magnitude, from the range used for soft solids in graphics MPM: a
citrus peel is stiffer than its flesh by roughly a decade, and bread crust by rather more.
"""
import numpy as np
import torch

C0 = 0.28209479177387814


PRESETS = {
    # name: (skin E, interior E, skin density, interior density, nu)
    "orange": dict(E_skin=8.0e6, E_soft=6.0e5, rho_skin=750.0, rho_soft=950.0, nu=0.35),
    "watermelon": dict(E_skin=2.0e7, E_soft=4.0e5, rho_skin=900.0, rho_soft=950.0, nu=0.35),
    "doughnut": dict(E_skin=3.0e6, E_soft=3.0e5, rho_skin=450.0, rho_soft=250.0, nu=0.30),
    "uniform": dict(E_skin=1.2e6, E_soft=1.2e6, rho_skin=800.0, rho_soft=800.0, nu=0.35),
    # Deliberately beyond anything physical: if a four-decade contrast between shell and
    # interior does not change the motion, the field is not reaching the solver and no
    # amount of choosing better values will help.
    "extreme": dict(E_skin=1.0e8, E_soft=1.0e4, rho_skin=800.0, rho_soft=800.0, nu=0.35),
}


def _rgb(gaussians_or_dc):
    dc = gaussians_or_dc
    if hasattr(dc, "_features_dc"):
        dc = dc._features_dc.detach()
    return (dc.squeeze(1) * C0 + 0.5).clamp(0, 1)


def material_field(colour, level, preset="orange", colour_rule=None):
    """Per-cell (E, nu, density) from the lattice level and the decoded colour.

    colour  (N, 3) in [0, 1]
    level   (N,)   0 for interior cells, 1 for skin cells

    Two rules, in order. The level is the structural one and applies to every object: the fine
    level is the shell the projection painted, so it is the peel. The colour rule then grades
    the interior, because "interior" is not one thing -- an orange's pith is not its flesh, and
    a doughnut's crust is not its crumb. It is written as a soft interpolation on lightness
    rather than a threshold: a threshold puts a discontinuity wherever it happens to fall, and
    MPM does not need one to produce a boundary that reads.
    """
    p = PRESETS[preset]
    # "uniform" is the control the others are read against, so it has to be genuinely uniform:
    # leaving the colour grading on gave its interior a decade of variation and there was
    # nothing left to compare to.
    if colour_rule is None:
        colour_rule = preset != "uniform"
    dev = colour.device
    n = colour.shape[0]
    lvl = level.reshape(-1)[:n].to(dev).float()

    E = torch.where(lvl > 0.5,
                    torch.full((n,), p["E_skin"], device=dev),
                    torch.full((n,), p["E_soft"], device=dev))
    rho = torch.where(lvl > 0.5,
                      torch.full((n,), p["rho_skin"], device=dev),
                      torch.full((n,), p["rho_soft"], device=dev))

    if colour_rule:
        # Pale interior is structure -- pith, albedo, rind under the skin, the white ring a
        # watermelon has inside its green. It is stiffer than the flesh beside it and lighter.
        # `t` runs 0 at the interior's median lightness to 1 at white.
        lum = colour.mean(1)
        med = float(lum[lvl < 0.5].median()) if (lvl < 0.5).any() else 0.5
        t = ((lum - med) / max(1e-3, 1.0 - med)).clamp(0, 1)
        soft = lvl < 0.5
        E = torch.where(soft, E * (1.0 + 9.0 * t), E)          # up to 10x within the interior
        rho = torch.where(soft, rho * (1.0 - 0.25 * t), rho)

    nu = torch.full((n,), p["nu"], device=dev)
    return E, nu, rho


def summarise(E, rho, level):
    lvl = level.reshape(-1)[:E.shape[0]].to(E.device)
    for name, m in (("skin", lvl > 0.5), ("interior", lvl < 0.5)):
        if not m.any():
            continue
        e = E[m]
        print(f"    {name:9s} {int(m.sum()):>8,} cells   E {float(e.min()):.2e} "
              f"- {float(e.max()):.2e} (median {float(e.median()):.2e})   "
              f"density {float(rho[m].median()):.0f}")


# Order-of-magnitude ranges per category, in pascals. These are not measurements of the
# particular fruit in front of us and are not claimed to be: appearance cannot give an absolute
# modulus, and a table keyed by what the object *is* supplies the one thing clustering cannot.
# Anything narrower would be a false precision. The upper end is also bounded by what the solver
# can integrate -- the wave speed is sqrt(E/rho) and the step has to resolve it, so a stiffer
# class costs a smaller dt rather than being free (E = 1e8 diverges at dt = 3e-4 on this grid).
CATEGORY_RANGES = {
    "fruit":  dict(E=(2.0e5, 8.0e6), rho=(900.0, 750.0), nu=0.35),
    "bread":  dict(E=(1.0e5, 3.0e6), rho=(250.0, 450.0), nu=0.30),
    "generic": dict(E=(2.0e5, 5.0e6), rho=(800.0, 800.0), nu=0.35),
}


def material_from_labels(labels, colour, level, category="fruit", report=True):
    """Per-cell material from a class labelling, in two separable steps.

    First a *relative* field: each class gets a number in [0, 1] saying how stiff it is
    compared with the others in this object. That part is derived, and it is derived from the
    two things the representation actually knows -- whether a class sits on the shell, and how
    pale it is, pale interior being structure (pith, albedo, crust) rather than flesh.

    Then an *absolute* scale, from the category's range. Splitting it this way keeps the claim
    honest in both directions: the ordering is ours and is checkable against the object, the
    magnitude is a table lookup and is not passed off as a measurement.
    """
    rng = CATEGORY_RANGES[category]
    dev = colour.device
    n = colour.shape[0]
    lab = labels.reshape(-1)[:n].to(dev).long()
    lvl = level.reshape(-1)[:n].to(dev).float()
    K = int(lab.max()) + 1

    skin_frac = torch.stack([(lvl[lab == j] > 0.5).float().mean() if (lab == j).any()
                             else torch.zeros((), device=dev) for j in range(K)])
    lum = torch.stack([colour[lab == j].mean() if (lab == j).any()
                       else torch.zeros((), device=dev) for j in range(K)])
    lum_n = (lum - lum.min()) / (lum.max() - lum.min()).clamp_min(1e-6)
    # A class on the shell is the shell whatever its colour; among interior classes, pale is
    # structural. Weighted so the first fact dominates and the second only orders the rest.
    r = (0.65 * skin_frac + 0.35 * lum_n).clamp(0, 1)
    r = (r - r.min()) / (r.max() - r.min()).clamp_min(1e-6)

    E_lo, E_hi = rng["E"]
    rho_lo, rho_hi = rng["rho"]
    E_c = E_lo + r * (E_hi - E_lo)
    rho_c = rho_lo + r * (rho_hi - rho_lo)
    if report:
        for j in range(K):
            print(f"    class {j}: skin {float(skin_frac[j]):.2f}  lightness {float(lum_n[j]):.2f}"
                  f"  -> r {float(r[j]):.2f}  E {float(E_c[j]):.2e}  rho {float(rho_c[j]):.0f}")
    nu = torch.full((n,), rng["nu"], device=dev)
    return E_c[lab], nu, rho_c[lab]
