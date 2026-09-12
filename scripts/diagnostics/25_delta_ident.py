"""Stage-dataset offset (delta) identifiability test for T2 (doc t2_p2_val §11.6).

Model per gene g, shared cluster k, stage S:
    log(m_{S,k,g} + e) = a_{k,g} + b_{k,g} t_S + delta_{S,g}

Algebra (checked here): the design has a TWO-dimensional null space --
  (i) delta_S += c            , a_k -= c          (constant gauge)
  (ii) delta_S += q t_S       , b_k -= q          (linear-in-t gauge)
so only the CURVATURE of delta in t is identifiable, i.e. the single combination
    rho_g = delta_M - (1-th) delta_A - th delta_B ,  th = (t_M-t_A)/(t_B-t_A)
Fix both gauges by writing delta(t) = gamma_g (t-t_A)(t-B), which vanishes at the
two outer stages. Then rho_g = gamma_g (t_M-t_A)(t_M-t_B) and the prediction at any
t* is gauge-invariant. Everything below uses TRAINING data only (no target file).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, anndata as ad
from t2.clusters import labels_to_clusters, cluster_order
from t2.io import to_dense

E = 1e-3
MIN_CELLS = 100


def stage_table(files, times, setting, panel):
    out = {}
    for stage, path in files.items():
        a = ad.read_h5ad(path)
        assert list(a.var_names) == panel, f"{path}: panel order mismatch"
        c = np.array(labels_to_clusters(a.obs["celltype"], setting))
        X = to_dense(a.X).astype(np.float64)
        out[stage] = (X, c, float(times[stage]))
    return out


def cluster_means(tab, order):
    means, counts = {}, {}
    for stage, (X, c, t) in tab.items():
        means[stage] = {}
        counts[stage] = {}
        for k in order:
            m = c == k
            counts[stage][k] = int(m.sum())
            means[stage][k] = X[m].mean(0) if m.sum() else None
    return means, counts


def fit_gamma(Y, ts, wts):
    """Y: (K,S) log-means for one gene; ts: (S,) times; wts: (K,S) weights.
    Basis: [1_k, t*1_k (k=1..K)] + [(t-tA)(t-tB)] -> params 2K+1."""
    K, S = Y.shape
    tA, tB = ts[0], ts[-1]
    cols = []
    for k in range(K):
        cols.append(((np.arange(K) == k)[:, None] * np.ones(S)[None, :]).ravel())
        cols.append(((np.arange(K) == k)[:, None] * ts[None, :]).ravel())
    quad = ((ts - tA) * (ts - tB))[None, :] * np.ones((K, 1))
    cols.append(quad.ravel())
    D = np.stack(cols, 1)
    W = np.sqrt(wts.ravel())[:, None]
    beta, *_ = np.linalg.lstsq(D * np.sqrt(wts.ravel())[:, None], Y.ravel() * np.sqrt(wts.ravel()), rcond=None)
    resid = Y.ravel() - D @ beta
    return beta, resid, D


def analyse(name, files, times, setting, panel_path, t_target, order_all):
    panel = [l.strip() for l in open(panel_path) if l.strip()]
    tab = stage_table(files, times, setting, panel)
    means, counts = cluster_means(tab, order_all)
    stages = list(files)
    ts = np.array([tab[s][2] for s in stages])
    idx = np.argsort(ts)
    stages = [stages[i] for i in idx]
    ts = ts[idx]
    shared = [k for k in order_all
              if all(means[s][k] is not None and counts[s][k] >= MIN_CELLS for s in stages)]
    K = len(shared)
    print(f"\n=== {name} ===")
    print("stages (sorted by t):", [(s, tab[s][2]) for s in stages])
    print(f"shared clusters with >= {MIN_CELLS} cells in all stages: K={K} -> {shared}")
    print("counts:", {s: {k: counts[s][k] for k in shared} for s in stages})
    if K < 4:
        print("TOO FEW shared clusters; abort")
        return None
    Y = np.stack([np.stack([np.log(means[s][k] + E) for s in stages]) for k in shared])  # (K,S,G)
    W = np.stack([np.stack([np.full(len(panel), counts[s][k]) for s in stages]) for k in shared], 1).astype(float)
    W = np.stack([np.stack([np.full(len(panel), float(counts[s][k])) for s in stages]) for k in shared])  # (K,S,G)
    G = len(panel)

    # ---- per-cluster residual from the outer-stage line, evaluated at middle stage
    tA, tM, tB = ts
    th = (tM - tA) / (tB - tA)
    r = Y[:, 1, :] - ((1 - th) * Y[:, 0, :] + th * Y[:, 2, :])          # (K,G)
    rho = r.mean(0)                                                      # identifiable combination
    sd_k = r.std(0, ddof=1)
    se_rho = sd_k / np.sqrt(K)
    print(f"\ntheta (middle-stage interp weight) = {th:.4f}")
    print(f"rho_g = mean_k r_{{k,g}}   : mean {rho.mean():+.4f}  sd across genes {rho.std():.4f}")
    print(f"within-gene sd across clusters: median {np.median(sd_k):.4f}  (noise on one cluster)")
    print(f"SE(rho_g)                     : median {np.median(se_rho):.4f}")
    print(f"signal/noise  |rho|/SE        : median {np.median(np.abs(rho)/np.maximum(se_rho,1e-9)):.2f}"
          f"   frac genes |rho|>2SE: {np.mean(np.abs(rho) > 2*se_rho):.3f}")
    # split-half reproducibility of rho (honest: disjoint cluster replicates)
    h1, h2 = r[: K // 2].mean(0), r[K // 2:].mean(0)
    print(f"split-half corr(rho) = {np.corrcoef(h1, h2)[0,1]:.3f}   (K1={K//2}, K2={K-K//2})")
    loo = np.array([np.delete(r, k, 0).mean(0) for k in range(K)])
    print(f"leave-one-cluster-out corr(rho) min = {min(np.corrcoef(loo[k], rho)[0,1] for k in range(K)):.3f}")

    # ---- full fit, curvature gamma and gauge-invariant prediction at t_target
    gam = np.zeros(G); a = np.zeros((K, G)); b = np.zeros((K, G)); rss = np.zeros(G); rss0 = np.zeros(G)
    for g in range(G):
        beta, resid, D = fit_gamma(Y[:, :, g], ts, W[:, :, g])
        a[:, g] = beta[: K]; b[:, g] = beta[K: 2 * K]; gam[g] = beta[-1]
        rss[g] = (resid ** 2 * W[:, :, g].ravel()).sum() / W[:, :, g].ravel().sum()
        D0 = D[:, : 2 * K]
        Wv = np.sqrt(W[:, :, g].ravel())[:, None]
        beta0, *_ = np.linalg.lstsq(D0 * Wv, Y[:, :, g].ravel() * Wv.ravel(), rcond=None)
        r0 = Y[:, :, g].ravel() - D0 @ beta0
        rss0[g] = (r0 ** 2 * W[:, :, g].ravel()).sum() / W[:, :, g].ravel().sum()
    print(f"\ngamma_g (curvature)   : sd {gam.std():.4f}   |gamma| median {np.median(np.abs(gam)):.4f}")
    print(f"weighted residual MSE: no-curvature {rss0.mean():.5f} -> with curvature {rss.mean():.5f}"
          f"  ({100*(1-rss.mean()/rss0.mean()):.1f}% reduction, df {K-1}->{K-2})")
    quad_t = (t_target - tA) * (t_target - tB)
    delta_star = gam * quad_t
    print(f"delta_hat(t*={t_target}) = gamma*(t*-tA)(t*-tB), (t*-tA)(t*-tB)={quad_t:+.4f}")
    print(f"  sd across genes {delta_star.std():.4f}  |median| {np.median(np.abs(delta_star)):.4f}")

    # scale reference: how big is the clock trend between the outer stages?
    trend = (b * (tB - tA))
    print(f"reference scales: sd_g(log pb_B - log pb_A) = {np.std(Y[:,2,:].mean(0)-Y[:,0,:].mean(0)):.4f}"
          f" | sd_g cluster clock trend b*(tB-tA) = {trend.std():.4f}")
    print(f"  -> |delta(t*)| / |clock trend| = {delta_star.std()/max(trend.std(),1e-12):.3f}")

    # predicted cluster means at t*, then pb with a composition estimate
    logm_star = a + b * t_target + delta_star[None, :]
    m_star = np.exp(logm_star) - E
    return dict(panel=panel, stages=stages, shared=shared, rho=rho, gam=gam, delta_star=delta_star,
                m_star=m_star, a=a, b=b, ts=ts, Y=Y, r=r, se_rho=se_rho)


if __name__ == "__main__":
    from t2.clusters import cluster_order
    res = analyse(
        "heart (A=E8.25, M=E8.75, B=E9.5; board target 8.5)",
        {"E8.25": "data/E8.25_late.h5ad", "E8.75": "data/E8.75.h5ad", "E9.5": "data/E9.5.h5ad"},
        {"E8.25": 8.25, "E8.75": 8.75, "E9.5": 9.5}, "heart",
        "panels/T2__heart__val_interp.genes.txt", 8.5, cluster_order("heart"))
