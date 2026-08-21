"""Turn a longitudinal plane about the object's axis, and turn its camera with it.

The longitudinal family is ten fixed sheets through the axis and it never moves, so it reaches
15.9% of the orange's cells against the transverse family's 92.4% -- 84% of the object has no
longitudinal supervision at all, and a held-out longitudinal cut passes almost entirely through
cells only the other family ever constrained.

`JITTER_V` tried to fix that by sliding the plane along its normal, which is the transverse
family's degree of freedom and not this one's: it takes the plane off the axis, and every
longitudinal photograph is a CENTRAL section, so the target became a picture of a cut the plane was
no longer taking. Measured, it cost the transverse family 36%.

Turning the plane about the axis instead keeps it through the axis, so it is still a central
section and the photographs are still the right kind, while it sweeps the cells between the ten
azimuths -- the ones nothing has ever crossed.

The camera is the existing one, rotated, rather than a new one built here: the framing the
references were mapped onto stays exactly what it was, and only the direction changes.
"""
import numpy as np


def rodrigues(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def turn(mvp, n, d, axis, centre, ang):
    """The plane and the camera, both turned by `ang` about `axis` through `centre`.

    `mvp` is the row-vector matrix the rasteriser is fed: clip = [x, 1] @ mvp. The new camera has
    to see y where the old saw R^-1 (y - c) + c, and for a rotation R^-1 = R^T, so the affine that
    does it has R itself in its upper block under the row convention. Getting that transpose wrong
    turns the object instead of the camera and the two come apart, which is what `check` is for.
    """
    R = rodrigues(axis, ang)
    n = np.asarray(n, float)
    c = np.asarray(centre, float)
    n2 = R @ n
    d2 = d + float(n @ c) - float(n2 @ c)
    A = np.eye(4)
    A[:3, :3] = R
    A[3, :3] = c - c @ R
    return (A @ np.asarray(mvp, float)).astype(np.float32), n2.astype(np.float32), float(d2)


def check(mvp, n, d, axis, centre):
    """Turning by nothing must give back exactly what went in, and by a full turn as well."""
    m0, n0, d0 = turn(mvp, n, d, axis, centre, 0.0)
    ok = np.allclose(m0, np.asarray(mvp, np.float32), atol=1e-5) \
        and np.allclose(n0, np.asarray(n, np.float32), atol=1e-6) and abs(d0 - d) < 1e-6
    m1, n1, d1 = turn(mvp, n, d, axis, centre, 2 * np.pi)
    ok = ok and np.allclose(m1, np.asarray(mvp, np.float32), atol=1e-4)
    # and the turned plane still has to contain the axis: a point on the axis stays on the plane
    p = np.asarray(centre, float)
    m2, n2, d2 = turn(mvp, n, d, axis, centre, 0.7)
    on = abs(float(n2 @ p) + d2) < 1e-6 if abs(float(n @ p) + d) < 1e-6 else True
    return ok and on
