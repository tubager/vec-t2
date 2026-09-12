"""HONEST W2 gate for the clock re-target machinery (gamma=0: only two anchors).

Board target E8.5 sits between E8.25 and E8.75 and our live X is a verbatim pick
mixture of those two datasets (75.4% / 24.6%, measured). W2 = anchors E8.25+E9.5,
truth E8.75, so the SAME machinery can be scored against a real truth with no
curvature term (2 anchors -> gamma unidentifiable). If the machinery cannot beat
the uncorrected pipeline pb here, it must not be submitted.

Also tests a second, independent lever: per-gene variance calibration.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import numpy as np, anndata as ad
from common.core_metrics import mmd_unbiased, variogram_score, de_score, de_direction, pb_rel_err
from T2.metrics import neighborhood_mmd
from t2.clusters import labels_to_clusters, cluster_order
from t2.expression import CompositionT2
from t2.io import to_dense

E = 1e-3
A = ad.read_h5ad("data/E8.25_late.h5ad"); B = ad.read_h5ad("data/E9.5.h5ad"); T = ad.read_h5ad("data/E8.75.h5ad")
XA, XB, XT = (to_dense(z.X).astype(np.float64) for z in (A, B, T))
cA = np.array(labels_to_clusters(A.obs["celltype"], "heart"))
cB = np.array(labels_to_clusters(B.obs["celltype"], "heart"))
cT = np.array(labels_to_clusters(T.obs["celltype"], "heart"))
order = cluster_order("heart")
P = ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad")
XP = to_dense(P.X).astype(np.float64); CP = np.asarray(P.obsm["spatial_3D"], float)
tC = np.asarray(T.obsm["spatial_3D"], float)
pt = XT.mean(0); px = XP.mean(0)
w = 0.4  # (8.75-8.25)/(9.5-8.25)
rel = lambda v: np.linalg.norm(v - pt) / np.linalg.norm(pt)

# --- identify each pipeline row's source stage + cluster (verbatim anchor rows)
def rowmap(X, cl):
    d = {}
    for i, r in enumerate(X):
        d.setdefault(r.astype(np.float32).tobytes(), (i, cl[i]))
    return d
dA, dB = rowmap(XA, cA), rowmap(XB, cB)
src, pcl = [], []
for r in XP:
    k = r.astype(np.float32).tobytes()
    if k in dA: src.append("A"); pcl.append(dA[k][1])
    elif k in dB: src.append("B"); pcl.append(dB[k][1])
    else: src.append("?"); pcl.append("?")
src = np.array(src); pcl = np.array(pcl)
print(f"pipeline rows sourced: A {np.mean(src=='A'):.3f}  B {np.mean(src=='B'):.3f}  unmatched {np.mean(src=='?'):.3f}")
pi_our = {k: float(np.mean(pcl == k)) for k in order}
pi_T = {k: float(np.mean(cT == k)) for k in order}
comp = CompositionT2("heart", {"A": 8.25, "B": 9.5}); comp.fit({"A": A.obs["celltype"], "B": B.obs["celltype"]}, {"A": 8.25, "B": 9.5})
pi_est = comp.probs(8.75)

mA = {k: (XA[cA == k].mean(0) if (cA == k).sum() else None) for k in order}
mB = {k: (XB[cB == k].mean(0) if (cB == k).sum() else None) for k in order}
def clock(k, mode):
    a, b = mA[k], mB[k]
    if a is None: return b
    if b is None: return a
    if mode == "lin": return (1 - w) * a + w * b
    return np.exp((1 - w) * np.log(a + E) + w * np.log(b + E)) - E

print("\n--- pb rel_err vs truth E8.75 (lower is better) ---")
print(f"  our pipeline (uncorrected)          : {rel(px):.4f}")
for mode in ("lin", "geo"):
    for nm, pi in (("pi_our", pi_our), ("pi_est", pi_est), ("pi_truth", pi_T)):
        v = np.zeros(len(pt)); tot = 0.0
        for k in order:
            p = pi.get(k, 0.0)
            if p <= 0: continue
            m = clock(k, mode)
            if m is None: continue
            v += p * m; tot += p
        print(f"  clock-{mode} cluster means, {nm:9s}    : {rel(v/tot):.4f}")

# --- build the correction from the pi_our / clock-lin target (the submittable recipe)
tgt = np.zeros(len(pt)); tot = 0.0
for k in order:
    p = pi_our.get(k, 0.0)
    if p <= 0: continue
    m = clock(k, "lin")
    if m is None: continue
    tgt += p * m; tot += p
tgt /= tot
c = np.log(tgt + E) - np.log(px + E)
print(f"\ncorrection c_g: sd {c.std():.4f}  mean {c.mean():+.4f}  |max| {np.abs(c).max():.4f}")

def evaluate(Xl, tag):
    r = pb_rel_err(Xl, XT)
    print(f"  {tag:28s} rel {r:.4f}  de {de_score(Xl,XT,XA)['score']:.4f}  ddir {de_direction(Xl,XT,XA):.4f}"
          f"  mmd {mmd_unbiased(Xl,XT,seed=0):.5f}  var {variogram_score(Xl,XT,seed=0):.5f}"
          f"  NFS {neighborhood_mmd(Xl,CP,XT,tC):.5f}")

print("\n--- W2 metrics: multiplicative clock re-target, lambda grid ---")
evaluate(XP, "lambda=0 (baseline)")
for lam in (0.25, 0.5, 0.75, 1.0):
    Xl = np.maximum(np.exp(np.log(XP + E) + lam * c) - E, 0.0)
    evaluate(Xl, f"lambda={lam:.2f}")

# --- second lever: per-gene variance calibration (mean-preserving) ---
vA, vB, vT, vP = XA.var(0), XB.var(0), XT.var(0), XP.var(0)
vhat = (1 - w) * vA + w * vB
print("\n--- per-gene variance ---")
print(f"  truth var (mean over genes) {vT.mean():.4f} | anchor-clock vhat {vhat.mean():.4f} | ours {vP.mean():.4f}")
rt = vP / np.maximum(vT, 1e-12); rh = vP / np.maximum(vhat, 1e-12)
print(f"  ours/truth  ratio: median {np.median(rt):.3f}  IQR {np.percentile(rt,25):.3f}-{np.percentile(rt,75):.3f}")
print(f"  ours/vhat   ratio: median {np.median(rh):.3f}  IQR {np.percentile(rh,25):.3f}-{np.percentile(rh,75):.3f}")
mx = XP.mean(0)
print("\n--- W2 metrics: variance calibration toward anchor-clock vhat ---")
for lam in (0.25, 0.5, 1.0):
    s = np.power(np.maximum(vhat, 1e-12) / np.maximum(vP, 1e-12), 0.5 * lam)
    Xl = np.maximum(mx + s * (XP - mx), 0.0)
    evaluate(Xl, f"var-calib lambda={lam:.2f}")
print("--- W2 metrics: variance calibration toward TRUTH var (oracle ceiling) ---")
s = np.power(np.maximum(vT, 1e-12) / np.maximum(vP, 1e-12), 0.5)
evaluate(np.maximum(mx + s * (XP - mx), 0.0), "var-calib oracle")
