import sys
from pathlib import Path
ROOT=Path("/Users/eric/Downloads/Embryo")
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/".venv/lib/python3.13/site-packages")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))  # _reuse shim for T2.metrics
import numpy as np, anndata as ad
from common.core_metrics import mmd_unbiased, variogram_score, de_score, de_direction, pb_rel_err
from T2.metrics import neighborhood_mmd
def dense(a):
    X=a.X; return np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
A=ad.read_h5ad("data/E8.25_late.h5ad"); B=ad.read_h5ad("data/E9.5.h5ad"); T=ad.read_h5ad("data/E8.75.h5ad")
XA,XB,XT=dense(A),dense(B),dense(T); tC=np.asarray(T.obsm['spatial_3D'],float)
P=ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad"); X=dense(P); C=np.asarray(P.obsm['spatial_3D'],float)
pa,pb,pt,px=XA.mean(0),XB.mean(0),XT.mean(0),X.mean(0)
w=0.4; e=1e-3
tgt={"geo":np.exp((1-w)*np.log(pa+e)+w*np.log(pb+e)),
     "lin":pa+w*(pb-pa),
     "geo_e6":np.exp((1-w)*np.log(pa+1e-6)+w*np.log(pb+1e-6))}
rel=lambda v: np.linalg.norm(v-pt)/np.linalg.norm(pt)
print("rel_err: ours %.4f | geo-tgt %.4f | lin-tgt %.4f"%(rel(px),rel(tgt['geo']),rel(tgt['lin'])))
print(f"{'target':6s} {'lam':>5s} {'rel_err':>8s} {'de_sc':>7s} {'de_dir':>7s} {'mmd':>8s} {'var':>8s} {'NFS':>8s}")
best=[]
for nm,t in tgt.items():
    lr=np.log(t+e)-np.log(px+e)
    for lam in (0.0,0.15,0.3,0.5,0.75,1.0):
        Xl=np.exp(np.log(X+e)+lam*lr)-e          # per-gene multiplicative in log space, keeps zeros
        Xl=np.maximum(Xl,0.0)
        r=pb_rel_err(Xl,XT)
        row=(nm,lam,r,de_score(Xl,XT,XA)['score'],de_direction(Xl,XT,XA),mmd_unbiased(Xl,XT,seed=0),
             variogram_score(Xl,XT,seed=0),neighborhood_mmd(Xl,C,XT,tC))
        print(f"{nm:6s} {lam:5.2f} {row[2]:8.4f} {row[3]:7.4f} {row[4]:7.4f} {row[5]:8.5f} {row[6]:8.5f} {row[7]:8.5f}")
        best.append(row)
