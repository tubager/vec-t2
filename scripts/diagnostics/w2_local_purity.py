"""Which side is wrong: local purity or neighbourhood-pb dispersion?

Re-pairing X to coords is the ONE move that changes neighborhood_mmd (25% of total)
and nothing else: expr/state depend only on the X multiset, d2/ODS/scale only on the
coordinate point set. So the direction of the fix must be measured, not guessed.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import numpy as np, anndata as ad
from sklearn.neighbors import NearestNeighbors
from common.core_metrics import mmd_unbiased
from t2.clusters import labels_to_clusters
from t2.io import to_dense

K = 15


def nbpb(X, C, k=K):
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(C).kneighbors(C)
    return X[idx[:, 1:]].mean(1)


def purity(cl, C, k=K):
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(C).kneighbors(C)
    nb = cl[idx[:, 1:]]
    return float(np.mean(nb == cl[:, None]))


def disp(P):
    """mean pairwise euclidean distance among neighbourhood pseudobulks (dispersion)."""
    Q = P[np.random.default_rng(0).choice(len(P), min(2000, len(P)), replace=False)]
    Q = Q - Q.mean(0)
    return float(np.sqrt((Q ** 2).sum(1).mean()))


A = ad.read_h5ad("data/E8.25_late.h5ad"); B = ad.read_h5ad("data/E9.5.h5ad"); T = ad.read_h5ad("data/E8.75.h5ad")
XA, XB, XT = (to_dense(z.X).astype(np.float64) for z in (A, B, T))
cA = np.array(labels_to_clusters(A.obs["celltype"], "heart"))
cB = np.array(labels_to_clusters(B.obs["celltype"], "heart"))
cT = np.array(labels_to_clusters(T.obs["celltype"], "heart"))
CA, CB, CT = (np.asarray(z.obsm["spatial_3D"], float) for z in (A, B, T))
P = ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad")
XP = to_dense(P.X).astype(np.float64); CP = np.asarray(P.obsm["spatial_3D"], float)

# recover our per-row cluster by exact match to the anchors
def idxmap(X, cl):
    d = {}
    for i, r in enumerate(X.astype(np.float32)):
        d.setdefault(r.tobytes(), cl[i])
    return d
dA, dB = idxmap(XA, cA), idxmap(XB, cB)
cP = np.array([dA.get(r.tobytes(), dB.get(r.tobytes(), "?")) for r in XP.astype(np.float32)])
print(f"our rows with a recovered cluster label: {np.mean(cP!='?'):.3f}")

print("\n--- local purity (fraction of the 15 neighbours in the same cluster) ---")
rng = np.random.default_rng(0)
for nm, cl, C in (("E8.25 (real tissue)", cA, CA), ("E8.75 (real tissue)", cT, CT), ("E9.5 (real tissue)", cB, CB)):
    s = rng.choice(len(cl), 5000, replace=False)
    print(f"  {nm:22s} n=5000 subsample: {purity(cl[s], C[s]):.4f}")
m = cP != "?"
print(f"  {'OURS (w2 pipeline)':22s} n={m.sum()}: {purity(cP[m], CP[m]):.4f}")

print("\n--- neighbourhood-pseudobulk dispersion (mean ||nbpb - mean||) ---")
for nm, X, C in (("E8.25", XA, CA), ("E8.75 TRUTH", XT, CT), ("E9.5", XB, CB), ("OURS", XP, CP)):
    s = rng.choice(len(X), min(5000, len(X)), replace=False)
    print(f"  {nm:12s} {disp(nbpb(X[s], C[s])):.4f}")

print("\n--- W2 NFS under re-pairings that keep BOTH multisets fixed ---")
PT, POp = nbpb(XP, CP), nbpb(XT, CT)
def nfs(Cpair):
    return mmd_unbiased(nbpb(XP, Cpair), POp, n=2000)
print(f"  baseline (pipeline OT pairing)      : {mmd_unbiased(PT, POp, n=2000):.5f}")
for nm, perm in (("random shuffle", rng.permutation(len(XP))),):
    Cn = CP[perm]
    print(f"  {nm:34s}: {nfs(Cn):.5f}")
# cluster-coherent: keep each cluster's coordinate SUBSET, but re-draw which cell gets which
# coordinate inside it (pure within-cluster shuffle -> purity unchanged, tests X-order noise)
perm2 = np.arange(len(XP))
for c in np.unique(cP):
    s = np.flatnonzero(cP == c)
    perm2[s] = s[rng.permutation(len(s))]
print(f"  within-cluster shuffle (purity=)   : {nfs(CP[perm2]):.5f}")
