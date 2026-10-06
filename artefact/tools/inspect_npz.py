# SPDX-License-Identifier: GPL-3.0-or-later
import sys, glob, zipfile, numpy as np
"""Print keys, dtypes, shapes and value ranges of session files.
Run: python tools/inspect_npz.py [file.npz ...]   (default: recordings/*.npz)"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (sys.argv[1:] or sorted(glob.glob(str(ROOT / "recordings" / "*.npz")))):
    print("="*80); print(p)
    z = zipfile.ZipFile(p); print("zip testzip():", z.testzip(), "| members:", [(i.filename, i.file_size, i.compress_type) for i in z.infolist()])
    d = np.load(p, allow_pickle=False)
    for k in d.files:
        a = d[k]
        line = f"  {k:14s} dtype={a.dtype} shape={a.shape}"
        if a.dtype.kind in "fiu" and a.size > 1:
            line += f" min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g} std={a.std():.4g}"
            if a.dtype.kind == "f": line += f" nan={int(np.isnan(a).sum())} inf={int(np.isinf(a).sum())}"
        else:
            line += f" value={a.tolist() if a.size < 10 else a[:5]}"
        print(line)
    X, y = d["eeg_data"], d["labels"]
    names = d["class_names"].tolist()
    print("  label counts:", {names[int(k)]: int((y == k).sum()) for k in np.unique(y)})
    print("  per-channel (axis=ch) over all trials: min / max / mean / std")
    for c in range(X.shape[2]):
        v = X[:, :, c]
        print(f"   ch{c:02d} {v.min():12.2f} {v.max():12.2f} {v.mean():12.2f} {v.std():10.2f}  nunique={len(np.unique(v))}")
    # trials that are constant / padded tail
    flat = [(i, int((np.abs(np.diff(X[i], axis=0)).sum(axis=1) == 0).sum())) for i in range(len(X))]
    print("  trials with >0 identical consecutive samples (padded tail?):", [(i,n) for i,n in flat if n>0][:20], "...count", sum(1 for _,n in flat if n>0))
    allconst = [i for i in range(len(X)) if np.all(X[i] == X[i][0])]
    print("  fully constant trials:", allconst)
