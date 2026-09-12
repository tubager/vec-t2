import types
import numpy as np
import scipy.sparse as sp

def to_dense(X):
    return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)

def _missing(*a, **k):
    raise NotImplementedError("shim")

t1_metrics = types.SimpleNamespace(
    pseudobulk_pearson=_missing, pseudobulk_pearson_celltype=_missing,
    composition_jsd=_missing, rbf_mmd=_missing, deg_recovery=_missing,
    train_frozen_probe=_missing, to_dense=to_dense,
)
