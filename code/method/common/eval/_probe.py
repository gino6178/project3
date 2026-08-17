import sys, numpy as np, torch
from plyfile import PlyData
e = PlyData.read(sys.argv[1]).elements[0]
c = np.stack([e['f_dc_0'], e['f_dc_1'], e['f_dc_2']], 1) * 0.28209479177387814 + 0.5
lv = torch.load(sys.argv[2]).numpy().reshape(-1)[:len(c)]
print(f"  {len(c):,} rows, cell_level {len(lv):,}")
for k, nm in ((0, 'coarse/interior'), (1, 'fine/skin')):
    m = lv == k
    if m.any():
        print(f"  {nm:<16} mean {c[m].mean(0).round(3).tolist()}  std {c[m].std(0).round(4).tolist()}")
