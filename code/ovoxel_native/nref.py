import os
import refsel
FN = "/workspace/rebuild/worktree"
for d in ("secref_orraw_hsep", "secref_orraw_vsep"):
    p = os.path.join(FN, d)
    fs = sorted(refsel.photos_in(p))
    print(f"  {d}: {len(os.listdir(p))} files on disk, {len(fs)} used as references")
    for f in fs:
        print("     ", os.path.basename(f))
