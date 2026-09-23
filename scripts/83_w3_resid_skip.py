#!/usr/bin/env python
"""Identity-skip residual Δ(t): X' = keepz(expm1(log1p(X)+δ)). Heart W3.

Cell is never decoded. Last layer of δ is zero-init so native t ≈ identity.
Train E8.25→E8.75 OT hop (one-way) + native identity. No E9.5, no PCA, no VAE.

  .venv/bin/python scripts/83_w3_resid_skip.py --train
  .venv/bin/python scripts/83_w3_resid_skip.py --eval-w3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from common.core_metrics import de_direction, de_score, mmd_unbiased, variogram_score  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402
from t2.clusters import cluster_id_map, labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.expression import cluster_mean_delta  # noqa: E402
from t2.flow import cov_frobenius_loss, get_device  # noqa: E402
from t2.cond_flow import variogram_loss  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.ot import minibatch_ot_pairs  # noqa: E402
from t2.paths import load_config  # noqa: E402

KEEPERS = frozenset({"CM_IFT", "CM_OFT", "peri", "pam", "branch", "st"})
ALPHA_234 = 0.95
PROJ_EPS = 0.20
PROJ = frozenset({"ncc"})
MAE_CAP = 0.04
RECON_MAE = 0.03
CKPT_DIR = ROOT / "outputs/t2/heart/gate95_residskip"


def _load_78():
    spec = importlib.util.spec_from_file_location(
        "s78_extrap", ROOT / "scripts/78_extrap_g0_clust_delta.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


class ResidSkip(nn.Module):
    """δ(log1p X, t, cluster, xyz); X is added, never reconstructed from z."""

    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        hidden: int = 512,
        z_dim: int = 64,
        t_anchor: float = 8.25,
        t_scale: float = 4.0,
    ):
        super().__init__()
        self.t_anchor = float(t_anchor)
        self.t_scale = float(t_scale)
        self.emb = nn.Embedding(n_clusters, 32)
        self.xyz_net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.enc = nn.Sequential(nn.Linear(n_genes, hidden), nn.SiLU(), nn.Linear(hidden, z_dim))
        in_dim = z_dim + 1 + 32 + 64
        self.delta = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def _tn(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return (t - self.t_anchor) / max(self.t_scale, 1e-6)

    def delta_log(self, x, t, cid, xyz):
        y = torch.log1p(x.clamp_min(0.0))
        z = self.enc(y)
        inp = torch.cat([z, self._tn(t), self.emb(cid), self.xyz_net(xyz)], dim=-1)
        return self.delta(inp)

    def forward(self, x, t, cid, xyz):
        y = torch.log1p(x.clamp_min(0.0))
        xg = torch.expm1(y + self.delta_log(x, t, cid, xyz)).clamp_min(0.0)
        return xg.masked_fill(x <= 0, 0.0)


def _sc(z):
    return z["score"] if isinstance(z, dict) else z


def keepz(Xn, X0):
    out = np.clip(np.asarray(Xn, dtype=np.float64), 0.0, None)
    out[np.asarray(X0) <= 0] = 0.0
    return out


def scores(X, C, XT, CT, XA):
    return {
        "de": _sc(de_score(X, XT, XA)),
        "dir": float(de_direction(X, XT, XA)),
        "mmd": _sc(mmd_unbiased(X, XT)),
        "var": _sc(variogram_score(X, XT)),
        "nfs": _sc(neighborhood_mmd(X, C, XT, CT)),
    }


def _t32(a, device):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)


def apply_np(model, X, names, coords, t, ids, usable, meta, device, batch=512):
    rms = float(meta["rms"])
    X = np.asarray(X, dtype=np.float32)
    xyz = np.asarray(scale_cloud(np.asarray(coords, dtype=np.float32), rms), dtype=np.float32) / rms
    cid = np.array([int(ids[str(n)]) if str(n) in ids else 0 for n in names], dtype=np.int32)
    out = np.array(X, copy=True)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch):
            sl = slice(start, min(start + batch, len(X)))
            use = np.array([str(nm) in usable and str(nm) in ids for nm in names[sl]])
            if not use.any():
                continue
            x = _t32(X[sl][use], device)
            c = torch.from_numpy(np.ascontiguousarray(cid[sl][use], dtype=np.int32)).to(device).long()
            zx = _t32(xyz[sl][use], device)
            tb = torch.full((int(use.sum()),), float(t), device=device, dtype=torch.float32)
            xg = model(x, tb, c, zx).cpu().numpy()
            dest = np.array(out[sl], copy=True)
            dest[use] = xg
            out[sl] = dest
    return keepz(out, X)


def train(args) -> int:
    cfg = load_config()
    spec = target_spec("heart", 9.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    left, right = stages["E8.25"], stages["E8.75"]
    t0, t1, t2 = 8.25, 8.75, 9.5
    usable = set(shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], "heart"))
    if args.keepers_only:
        usable &= {"CM_IFT", "peri", "pam"}
    ids = cluster_id_map("heart")
    rng = np.random.default_rng(args.seed)
    rms = float(args.rms)

    stage_list = [("E8.25", left, t0), ("E8.75", right, t1)]
    if args.three_time:
        stage_list.append(("E9.5", stages["E9.5"], t2))
    packed = {}
    for tag, ad, t in stage_list:
        X = _align(to_dense(ad.X), ad.var_names, panel)
        cl = np.asarray(labels_to_clusters(ad.obs["celltype"], "heart")).astype(str)
        xyz = np.asarray(scale_cloud(spatial_xyz(ad), rms), dtype=np.float32) / rms
        keep = np.array([c in usable and c in ids for c in cl])
        idx = np.flatnonzero(keep)
        if args.max_per_stage > 0 and len(idx) > args.max_per_stage:
            idx = rng.choice(idx, args.max_per_stage, replace=False)
        packed[tag] = {
            "X": X[idx],
            "cl": cl[idx],
            "xyz": xyz[idx],
            "cid": np.array([ids[str(c)] for c in cl[idx]], dtype=np.int64),
            "t": t,
        }
        print(f"  {tag} n={len(idx)}")

    hop_specs = [("E8.25", "E8.75", t1)]
    if args.three_time:
        hop_specs.append(("E8.75", "E9.5", t2))
    hop_by = []
    for src, dst, tt in hop_specs:
        entries = {}
        for name in sorted(usable):
            m0 = packed[src]["cl"] == name
            m1 = packed[dst]["cl"] == name
            if m0.sum() < 8 or m1.sum() < 8:
                continue
            entries[name] = {
                "x0": packed[src]["X"][m0],
                "x1": packed[dst]["X"][m1],
                "z0": packed[src]["xyz"][m0],
                "cid": int(ids[name]),
                "t": tt,
            }
        hop_by.append(entries)
        print(f"hop {src}->{dst} t={tt} clusters {sorted(entries)}")
    names = sorted(hop_by[0])

    Xn = np.concatenate([packed[k]["X"] for k, _, _ in stage_list], 0)
    Tn = np.concatenate(
        [np.full(len(packed[k]["X"]), t, np.float32) for k, _, t in stage_list]
    )
    Cn = np.concatenate([packed[k]["cid"] for k, _, _ in stage_list], 0)
    Zn = np.concatenate([packed[k]["xyz"] for k, _, _ in stage_list], 0)

    device = torch.device(args.device) if args.device else get_device()
    model = ResidSkip(Xn.shape[1], len(ids), hidden=args.hidden, z_dim=args.z_dim, t_anchor=t0, t_scale=args.t_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(Xn)
    model.train()
    for epoch in range(1, args.epochs + 1):
        logs = []
        order = rng.permutation(n)
        for start in range(0, n, args.batch):
            sl = order[start : start + args.batch]
            x = torch.as_tensor(Xn[sl], device=device)
            t = torch.as_tensor(Tn[sl], device=device)
            c = torch.as_tensor(Cn[sl], device=device)
            z = torch.as_tensor(Zn[sl], device=device)
            xg = model(x, t, c, z)
            dlog = model.delta_log(x, t, c, z)
            loss = ((xg - x) ** 2).mean() + 4.0 * (dlog.pow(2).mean())
            stats = {"id": float(((xg - x) ** 2).mean().detach()), "d": float(dlog.pow(2).mean().detach())}

            key = names[int(rng.integers(0, len(names)))]
            hb = hop_by[int(rng.integers(0, len(hop_by)))]
            if key not in hb:
                key = sorted(hb)[int(rng.integers(0, len(hb)))]
            x0c, x1c = hb[key]["x0"], hb[key]["x1"]
            z0c = hb[key]["z0"]
            hop_t = float(hb[key]["t"])
            m = int(min(args.batch, 128, len(x0c), len(x1c)))
            i = rng.choice(len(x0c), size=m, replace=len(x0c) < m)
            j = rng.choice(len(x1c), size=m, replace=len(x1c) < m)
            a, b = minibatch_ot_pairs(x0c[i], x1c[j], m, rng)
            za = torch.from_numpy(np.ascontiguousarray(z0c[i][: len(a)], dtype=np.float32)).to(device)
            xa_np = np.asarray(a, dtype=np.float32)
            if args.after_mean:
                dmu = (np.asarray(x1c.mean(0), np.float32) - np.asarray(x0c.mean(0), np.float32))
                xa_np = np.clip(xa_np + dmu, 0.0, None)
                xa_np[np.asarray(a) <= 0] = 0.0
            xa = torch.from_numpy(np.ascontiguousarray(xa_np)).to(device)
            xb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to(device)
            ca = torch.full((len(a),), hb[key]["cid"], device=device, dtype=torch.long)
            t1t = xa.new_full((len(a),), hop_t)
            xh = model(xa, t1t, ca, za)
            hop = ((xh - xb) ** 2).mean()
            loss = loss + hop
            stats["hop"] = float(hop.detach())
            if args.var_weight > 0:
                lv = variogram_loss(xh, xb)
                loss = loss + float(args.var_weight) * lv
                stats["var"] = float(lv.detach())
            if args.cov_weight > 0:
                lc = cov_frobenius_loss(xh, xb)
                loss = loss + float(args.cov_weight) * lc
                stats["cov"] = float(lc.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stats["loss"] = float(loss.detach())
            logs.append(stats)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            keys = sorted({k for s in logs for k in s})
            msg = " ".join(f"{k}={np.mean([s.get(k, 0) for s in logs]):.4f}" for k in keys)
            print(f"epoch {epoch:3d}/{args.epochs}  {msg}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "n_genes": int(Xn.shape[1]),
        "n_clusters": len(ids),
        "hidden": int(args.hidden),
        "z_dim": int(args.z_dim),
        "t_anchor": t0,
        "t_scale": float(args.t_scale),
        "rms": rms,
        "panel": panel,
        "cluster_to_id": ids,
        "usable": sorted(usable),
        "arch": "resid_skip",
        "three_time": bool(args.three_time),
    }
    path = args.out_dir / "ckpt" / "resid_skip.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)
    (args.out_dir / "meta.json").write_text(json.dumps({k: extra[k] for k in extra if k != "panel"}, indent=2, default=str))
    print(f"saved {path}")
    return 0


def load_skip(ckpt: Path, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    extra = {k: v for k, v in blob.items() if k != "state_dict"}
    model = ResidSkip(
        n_genes=int(extra["n_genes"]),
        n_clusters=int(extra["n_clusters"]),
        hidden=int(extra.get("hidden", 512)),
        z_dim=int(extra.get("z_dim", 64)),
        t_anchor=float(extra["t_anchor"]),
        t_scale=float(extra["t_scale"]),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, extra


def eval_w3(args) -> int:
    s78 = _load_78()
    device = torch.device(args.device) if args.device else get_device()
    model, meta = load_skip(args.out_dir / "ckpt" / "resid_skip.pt", device)
    ids = dict(meta["cluster_to_id"])
    usable = set(meta["usable"])
    cfg = load_config()
    spec = target_spec("heart", 9.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    left, right, truth = stages["E8.25"], stages["E8.75"], stages["E9.5"]
    XA = s78._align(to_dense(left.X), left.var_names, panel)
    XB = s78._align(to_dense(right.X), right.var_names, panel)
    XT = s78._align(to_dense(truth.X), truth.var_names, panel)
    CB = np.asarray(spatial_xyz(right), float)
    CT = np.asarray(spatial_xyz(truth), float)
    clB = np.asarray(labels_to_clusters(right.obs["celltype"], "heart")).astype(str)
    gdel, by_k = cluster_mean_delta(left, right, "heart")
    by_k = s78._birth_continuation_delta(left, right, by_k)
    usable_k = s78._usable(left, right, include_birth=True, only=set(KEEPERS))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(XB), min(args.w3_n, len(XB)), replace=False)
    X0, C0, names = XB[pick], CB[pick], clB[pick]
    Y234, _, _ = s78._apply(
        X0, names, by_k, gdel, usable_k, ALPHA_234,
        None, 0.0, None, 0.0, PROJ_EPS, set(PROJ), 0, C0,
    )
    Y234 = keepz(Y234, X0)

    rec = apply_np(model, X0, names, C0, 8.75, ids, usable, meta, device)
    mae_rec = float(np.abs(rec - X0).mean())
    rec_ok = mae_rec <= RECON_MAE
    print(f"RECON native t=8.75 mae={mae_rec:.4f} ({'PASS' if rec_ok else 'FAIL'} ≤{RECON_MAE})")
    clT = np.asarray(labels_to_clusters(truth.obs["celltype"], "heart")).astype(str)
    pickT = rng.choice(len(XT), min(args.w3_n, len(XT)), replace=False)
    rec95 = apply_np(model, XT[pickT].astype(np.float32), clT[pickT], CT[pickT], 9.5, ids, usable, meta, device)
    mae_rec95 = float(np.abs(rec95 - XT[pickT]).mean())
    rec95_ok = mae_rec95 <= RECON_MAE
    print(f"RECON native t=9.5  mae={mae_rec95:.4f} ({'PASS' if rec95_ok else 'FAIL'} ≤{RECON_MAE})")
    if not rec_ok:
        print("FAIL — stop, no W3 board")
        return 0
    if bool(meta.get("three_time")) and not rec95_ok:
        print("FAIL — three-time native t=9.5 recon")
        return 0

    fut = apply_np(model, X0, names, C0, 9.5, ids, usable, meta, device)
    on234 = apply_np(model, Y234.astype(np.float32), names, C0, 9.5, ids, usable, meta, device)
    t105 = apply_np(model, Y234.astype(np.float32), names, C0, 10.5, ids, usable, meta, device)
    m_k = np.isin(names, list(usable_k))
    skip_k = np.array(Y234, copy=True)
    skip_k[m_k] = fut[m_k]
    on234_k = np.array(Y234, copy=True)
    on234_k[m_k] = on234[m_k]
    t105_k = np.array(Y234, copy=True)
    t105_k[m_k] = t105[m_k]
    catalog = [
        ("copy", X0),
        ("234", Y234),
        ("skip_t95", fut),
        ("on234", on234),
        ("skip_k", keepz(skip_k, X0)),
        ("on234_k", keepz(on234_k, X0)),
        ("t105", keepz(t105, X0)),
        ("t105_k", keepz(t105_k, X0)),
    ]
    for a in (0.10, 0.25):
        catalog.append((f"stack_{a:g}", keepz(Y234 + a * (t105 - Y234), X0)))
    print(
        f"keepers n={int(m_k.sum())}/{len(names)}  "
        f"mae t105 vs 234={float(np.abs(t105 - Y234).mean()):.4f}  "
        f"mae_k t105={float(np.abs(t105[m_k] - Y234[m_k]).mean()) if m_k.any() else 0:.4f}"
    )
    print("=== W3 resid-skip  vs truth E9.5 (t105 = 234 @ t=10.5, hop95 leaked) ===")
    print(f"{'tag':>12} {'de':>7} {'dir':>7} {'mmd':>8} {'var':>8} {'nfs':>8} {'mae234':>7} {'mae0':>7}")
    rows, by_tag, copy_var = {}, dict(catalog), None
    for tag, X in catalog:
        row = scores(X, C0, XT, CT, XA)
        rows[tag] = row
        if tag == "copy":
            copy_var = row["var"]
        print(
            f"{tag:>12} {row['de']:7.3f} {row['dir']:7.3f} {row['mmd']:8.4f} "
            f"{row['var']:8.4f} {row['nfs']:8.4f} {float(np.abs(X - Y234).mean()):7.4f} "
            f"{float(np.abs(X - X0).mean()):7.4f}"
        )
    base = rows["234"]
    print("\nAND vs 234 + chimera/mae/polluter veto:")
    for tag, row in rows.items():
        if tag in {"copy", "234"}:
            continue
        and_ok = row["de"] >= base["de"] - 1e-6 and row["var"] <= base["var"] * 1.10 and row["mmd"] <= base["mmd"] * 1.10
        chimera = row["var"] > copy_var * 1.50
        mae234 = float(np.abs(by_tag[tag] - Y234).mean())
        mae_ok = mae234 <= MAE_CAP
        polluter_only = (
            tag in {"skip_t95", "on234", "stack_0.1", "stack_0.25"}
            and rows.get("t105_k", rows.get("on234_k", base))["de"] <= base["de"] + 1e-6
            and row["de"] > base["de"] + 1e-6
        )
        improve = and_ok and (row["de"] > base["de"] + 1e-6)
        leaked = tag in {"skip_t95", "on234", "skip_k", "on234_k"}
        board = rec_ok and and_ok and (not chimera) and mae_ok and improve and (not polluter_only) and (not leaked)
        if bool(meta.get("three_time")):
            board = board and rec95_ok
        if tag in {"skip_k", "on234_k", "t105_k"} and row["de"] <= base["de"] + 1e-6:
            board = False
        if leaked:
            board = False
        print(
            f"  {tag:>12}  AND={'PASS' if and_ok else 'FAIL'}  chimera={'YES' if chimera else 'no'}"
            f"  mae234={mae234:.4f}{'  polluter-de' if polluter_only else ''}"
            f"  de {row['de'] - base['de']:+.3f}  var {row['var'] - base['var']:+.4f}"
            f"  => {'BOARD' if board else 'no-board'}"
        )
    return 0


def write_board(args) -> int:
    import anndata as ad

    s78 = _load_78()
    pred = ROOT / "outputs/t2/heart/pred_E10.5_full_nodelta_n25179_rms438.h5ad"
    out = Path(args.out)
    tmp = out.with_name(out.stem + ".__234__.h5ad")
    s78.write_board(
        pred,
        tmp,
        alpha=ALPHA_234,
        knn=15,
        include_birth=True,
        only=set(KEEPERS),
        rest=None,
        alpha_rest=0.0,
        curv_beta=0.0,
        keeper_boost=1.0,
        seed=0,
        proj_rest=PROJ_EPS,
        proj_clusters=set(PROJ),
        proj_knn=0,
    )
    device = torch.device(args.device) if args.device else get_device()
    model, meta = load_skip(args.out_dir / "ckpt" / "resid_skip.pt", device)
    ids = dict(meta["cluster_to_id"])
    usable = set(meta["usable"])
    cfg = load_config()
    spec = target_spec("heart", 10.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    right = stages["E9.5"]
    g0 = ad.read_h5ad(pred)
    Xp = s78._align(to_dense(g0.X), g0.var_names, panel)
    ad234 = ad.read_h5ad(tmp)
    Y234 = s78._align(to_dense(ad234.X), ad234.var_names, panel)
    Xref = s78._align(to_dense(right.X), right.var_names, panel)
    cref = np.asarray(labels_to_clusters(right.obs["celltype"], "heart")).astype(str)
    names = s78._assign_clusters(Xp, Xref, cref, 15)
    C0 = np.asarray(ad234.obsm["spatial_3D"], dtype=np.float32)
    t105 = apply_np(model, Y234.astype(np.float32), names, C0, 10.5, ids, usable, meta, device)
    a = float(args.blend)
    Y = keepz(Y234 + a * (t105 - Y234), Xp)
    mae234 = float(np.abs(Y - Y234).mean())
    if mae234 > MAE_CAP:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"mae vs 234 {mae234:.4f} > {MAE_CAP}; refuse board")
    fill = int(((Xp <= 0) & (Y > 0)).sum())
    assert fill == 0, fill
    if list(ad234.var_names) == panel:
        ad234.X = Y.astype(np.float32)
    else:
        live = np.asarray(to_dense(ad234.X), dtype=np.float32)
        idxg = {g: i for i, g in enumerate(panel)}
        for j, g in enumerate(ad234.var_names):
            live[:, j] = Y[:, idxg[g]]
        ad234.X = live
    ad234.obsm["spatial_3D"] = C0
    out.parent.mkdir(parents=True, exist_ok=True)
    ad234.write_h5ad(out, compression="gzip")
    tmp.unlink(missing_ok=True)
    print(
        f"board skip t105 blend={a:g} mae_g0={float(np.abs(Y - Xp).mean()):.4f} "
        f"mae_234={mae234:.4f} fill0=0 "
        f"rms={float(np.sqrt((C0.astype(np.float64) ** 2).sum(1).mean())):.2f}  {out}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-w3", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--t-scale", type=float, default=4.0)
    ap.add_argument("--rms", type=float, default=216.9)
    ap.add_argument("--max-per-stage", type=int, default=12000)
    ap.add_argument("--var-weight", type=float, default=2.0)
    ap.add_argument("--cov-weight", type=float, default=0.5)
    ap.add_argument("--w3-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keepers-only", action="store_true")
    ap.add_argument("--after-mean", action="store_true")
    ap.add_argument("--three-time", action="store_true")
    ap.add_argument("--write-board", action="store_true")
    ap.add_argument("--blend", type=float, default=0.25)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/t2/queue/2026-09-21/239_extrap_234_skip3t_b025_onG0_n25179.h5ad",
    )
    args = ap.parse_args()
    if args.write_board:
        return write_board(args)
    if args.train:
        train(args)
    if args.eval_w3 or not args.train:
        if not args.train and not args.eval_w3:
            args.eval_w3 = True
        if not (args.out_dir / "ckpt" / "resid_skip.pt").exists():
            raise SystemExit("no ckpt; run --train first")
        eval_w3(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
