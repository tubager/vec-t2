import sys
from pathlib import Path
ROOT=Path("/Users/eric/Downloads/Embryo")
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/".venv/lib/python3.13/site-packages")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))  # _reuse shim for T2.metrics
import numpy as np, anndata as ad
from T2.metrics import neighborhood_mmd
def dense(a):
    X=a.X; return np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
T=ad.read_h5ad("data/E8.75.h5ad"); tX=dense(T); tC=np.asarray(T.obsm['spatial_3D'],float)
P=ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad"); X=dense(P); C=np.asarray(P.obsm['spatial_3D'],float)
n=len(X); rng=np.random.default_rng(0); i=rng.choice(len(tX),n,replace=False)
tXi,tCi=tX[i],tC[i]
print(f"{'variant':52s} NFS")
def s(k,v): print(f"{k:52s} {v:.5f}")
s("ours: our X + our coords (as submitted)", neighborhood_mmd(X,C,tX,tC))
s("our X + TRUTH coords, OT-paired", None) if False else None
from t2.shape import ot_assign_xyz
Cot=np.asarray(ot_assign_xyz(C, tCi),dtype=float)
s("our X + TRUTH coords (OT paired to our X)", neighborhood_mmd(X,Cot,tX,tC))
s("our X + TRUTH coords (random pairing)", neighborhood_mmd(X,tCi[rng.permutation(n)],tX,tC))
s("TRUTH X + our coords (OT paired)", neighborhood_mmd(tXi,ot_assign_xyz(tCi,C),tX,tC))
s("TRUTH cells (ceiling)", neighborhood_mmd(tXi,tCi,tX,tC))
