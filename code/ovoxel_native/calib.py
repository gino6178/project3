"""Calibrate the orientation convention before any claim rests on it: put lines in at a known
angle and see what the detector says."""
import numpy as np, cv2
from scipy import ndimage

N = 512
yy, xx = np.mgrid[0:N, 0:N]
mask = ((yy - N/2)**2 + (xx - N/2)**2) < (N*0.42)**2


def spectrum(res, mask, nb=180):
    e = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    res = np.where(e, res, 0.0)
    en = []
    for a in range(nb):
        rot = ndimage.rotate(res, a, reshape=False, order=1, mode="constant")
        col = rot.sum(0)
        en.append(float(np.abs(col - col.mean()).mean()))
    en = np.array(en)
    return (en - en.mean()) / (en.std() + 1e-9)


for true_deg in (0, 30, 45, 82, 90, 135):
    # lines whose DIRECTION is true_deg measured from +x, image-y up
    t = np.radians(true_deg)
    # phase varies along the normal to the line direction
    u = (xx - N/2) * (-np.sin(t)) + (-(yy - N/2)) * np.cos(t)
    img = np.where(mask, np.sin(2*np.pi*u/9.0), 0.0)
    z = spectrum(img, mask)
    p = int(np.argmax(z))
    print(f"  lines drawn at {true_deg:>3} deg  ->  detector peak {p:>3} deg  z={z[p]:+.1f}   "
          f"(reported - true = {(p - true_deg) % 180:>3})")
