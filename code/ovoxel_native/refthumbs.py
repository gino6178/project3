"""One thumbnail of each family's reference, for the picker to show beside the cut."""
import os
import re
import sys

import numpy as np
import cv2

sys.path.insert(0, "/workspace/ovoxel_native")
import refsel

FN = "/workspace/rebuild/worktree"
OBJDIR = "/workspace/rebuild/project3/code/objects"
OBJS = ["orange_sp", "watermelon_sp", "apple1_sp", "bread_sp", "cake2_sp",
        "pomegranate2_sp", "doughnut"]
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for obj in OBJS:
    for which, tag in (("REF_H", "h"), ("REF_V", "v")):
        m = re.search(rf"^{which}=(\S+)", open(os.path.join(OBJDIR, f"{obj}.conf")).read(), re.M)
        d = os.path.join(FN, m.group(1))
        a = np.asarray(refsel.as_array(
            (refsel.solved_photo if which == "REF_H" else refsel.photo)(d, 0, 1), 224),
            np.float32)
        cv2.imwrite(os.path.join(out, f"ref_{obj}_{tag}.jpg"),
                    (np.clip(a, 0, 1) * 255).astype(np.uint8)[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {obj}", flush=True)
