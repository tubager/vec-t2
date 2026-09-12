import sys
from pathlib import Path
ROOT=Path("/Users/eric/Downloads/Embryo")
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/".venv/lib/python3.13/site-packages")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))  # _reuse shim for T2.metrics
import numpy as np, anndata as ad
from common.core_metrics import de_genes
panel=[l.strip() for l in open(ROOT/"panels/T2__heart__val_interp.genes.txt") if l.strip()]
def dense(a):
    X=a.X; return np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
def pb(p, rna=False):
    a=ad.read_h5ad(p); vs=list(a.var_names); idx=[vs.index(g) for g in panel if g in set(vs)]
    X=dense(a)[:,idx]
    return (X.mean(0), X)
p825,X825=pb("data/E8.25_late.h5ad"); p875,X875=pb("data/E8.75.h5ad"); p95,X95=pb("data/E9.5.h5ad")
r85,_=pb("data/E8.5_RNA.h5ad",True); r95,_=pb("data/E9.5_RNA.h5ad",True)
pt=p875; ref=p825
rel=lambda v: np.linalg.norm(v-pt)/np.linalg.norm(pt)
drna=(r95-r85)            # RNA delta over 1.0 day (E8.5 -> E9.5)
# held-out estimators of pb_SP(E8.75): step BACKWARD from the spatial E9.5 anchor along the RNA delta
for s in (0.5,0.75,1.0,1.25):
    est=p95-0.25*s*drna
    print(f"RNA-backward s={s:4.2f}  rel_err={rel(est):.4f}  cos(delta,truth delta)={np.corrcoef(est-ref,pt-ref)[0,1]:.4f}")
est_clock=p825+0.4*(p95-p825)
print(f"clock-linear (W2 anchors)   rel_err={rel(est_clock):.4f}  cos={np.corrcoef(est_clock-ref,pt-ref)[0,1]:.4f}")
est_sp_back=p95-0.75*(p95-p875)*0  # placeholder
est_sp=p95-(p95-p825)*(0.75/1.25)
print(f"spatial-only backward       rel_err={rel(est_sp):.4f}  cos={np.corrcoef(est_sp-ref,pt-ref)[0,1]:.4f}")
print(f"ORACLE                      rel_err={rel(pt):.4f}")
# DE-ranking quality (what de_score actually rewards), truth = E8.75 vs ref = E8.25
up_t,dn_t,_=de_genes(X875,X825,0.05,de_genes.__defaults__[1] if de_genes.__defaults__ else 0.25)
n_up,n_dn=len(up_t),len(dn_t); G=len(panel)
print(f"\ntruth DE sets: n_up={n_up} n_dn={n_dn} of G={G}")
chance=np.argsort(ref)                     # the metric's own null: rank by ref mean
def ov(v):
    lfc=np.log2((v+1e-3)/(ref+1e-3))
    up=set(np.argsort(lfc)[-n_up:]); dn=set(np.argsort(-lfc)[-n_dn:])
    raw=(len(up&set(up_t))+len(dn&set(dn_t)))/(n_up+n_dn)
    lfc_c=np.log2((ref+1e-3)/(ref+1e-3))+ref*0+ref
    upc=set(np.argsort(ref)[-n_up:]); dnc=set(np.argsort(-ref)[-n_dn:])
    ch=(len(upc&set(up_t))+len(dnc&set(dn_t)))/(n_up+n_dn)
    return raw,(raw-ch)/(1-ch)
for nm,v in [("clock-linear",est_clock),("spatial backward",est_sp),
             ("RNA-backward s=0.75",p95-0.25*0.75*drna),("RNA-backward s=1.0",p95-0.25*drna),
             ("RNA-backward s=1.25",p95-0.25*1.25*drna),("ORACLE",pt)]:
    raw,sk=ov(v); print(f"  {nm:22s} overlap={raw:.4f}  de_score={(sk if np.isfinite(sk) else float('nan')):.4f}")
