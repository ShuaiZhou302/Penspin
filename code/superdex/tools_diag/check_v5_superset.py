"""Rows of every v4 split are identical in the v5 dataset (v5 only adds train_f / shift_f)."""
import numpy as np
a = dict(np.load('runs/pairs_v4/dataset_mu1_v4.npz')); b = dict(np.load('runs/pairs_v5/dataset_mu1_v5.npz'))
m = np.isin(b['split'], np.unique(a['split']))
for k in a:
    if a[k].ndim and len(a[k]) == len(a['split']):
        assert np.array_equal(a[k], b[k][m]), k
print('V5_SUPERSET_OK', int(m.sum()), 'rows of', len(m), {s: int((b['split'] == s).sum()) for s in ('train_f', 'shift_f')})
