import numpy as np
import numexpr

with open(r"iplist1.txt", mode="rb") as f:
    aa = np.array(f.read().splitlines())
with open(r"iplist2.txt", mode="rb") as f:
    bb = np.array(f.read().splitlines())
if not aa.flags["C_CONTIGUOUS"]:
    aa = np.ascontiguousarray(aa)
a = np.lib.stride_tricks.as_strided(aa, (len(aa), len(bb)), (aa.itemsize, 0))
boollist = numexpr.evaluate("a==bb")
c = a[boollist]
d = np.where(boollist)
r1 = bb[d[1]]
r2 = aa[d[0]]

# python puro
# result = [(i1, i2) for i1 in enumerate(aa) for i2 in enumerate(bb) if i1[1] == i2[1]]
