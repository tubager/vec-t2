import sys
from pathlib import Path
ROOT = Path("/Users/eric/Downloads/Embryo")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/".venv/lib/python3.13/site-packages")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))  # _reuse shim for T2.metrics
import numpy as np, anndata as ad
from t2.expression import CompositionT2, labels_to_clusters
from common.core_metrics import mmd_unbiased, variogram_score, de_score, de_direction
from common.shape_metrics import scale_log_ratio
from T2.metrics import neighborhood_mmd

def dense(a):
    X=a.X; return np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
def rms(C): 
    Z=C-C.mean(0); return float(np.sqrt((Z**2).sum(1).mean()))
def to_rms(C, target):
    Z=C-C.mean(0); return Z*(target/rms(C))

A=ad.read_h5ad("data/E8.25_late.h5ad"); B=ad.read_h5ad("data/E9.5.h5ad"); T=ad.read_h5ad("data/E8.75.h5ad")
XA,CA=dense(A),np.asarray(A.obsm['spatial_3D'],float)
XB,CB=dense(B),np.asarray(B.obsm['spatial_3D'],float)
tX,tC=dense(T),np.asarray(T.obsm['spatial_3D'],float)
RT=rms(tC); print("# truth E8.75 rms=%.1f n=%d"%(RT,len(tX)))
clA=np.array(labels_to_clusters(A.obs['celltype'],'heart')); clB=np.array(labels_to_clusters(B.obs['celltype'],'heart'))
comp=CompositionT2("heart",{"A":8.25,"B":9.5}); comp.fit({"A":A.obs['celltype'],"B":B.obs['celltype']},{"A":8.25,"B":9.5})
pi=comp.probs(8.75)
N=5000; w=0.4
rng=np.random.default_rng(0)
keys=[k for k in comp.order if pi[k]>0]; p=np.array([pi[k] for k in keys]); p/=p.sum()
counts=dict(zip(keys,rng.multinomial(N,p)))

def build(src, coord="own", target_rms=RT):
    Xs=[]; As=[]; Bs=[]
    for k,c in counts.items():
        if src=="mix":
            nb=int(round(c*w)); na=c-nb
        elif src=="A": na,nb=c,0
        else: na,nb=0,c
        for side,m in (("A",na),("B",nb)):
            if m<=0: continue
            X_,C_,cl_=(XA,CA,clA) if side=="A" else (XB,CB,clB)
            idx=np.flatnonzero(cl_==k)
            if len(idx)==0: continue
            sel=idx[rng.choice(len(idx),m,replace=len(idx)<m)]
            Xs.append(X_[sel])
            Z=C_[sel]-C_[sel].mean(0)
            r=rms(C_[sel]) if len(sel)>1 else 1.0
            # per-side rescale so BOTH anchors sit at the truth's overall scale (density-matched)
            As.append(Z*(target_rms/rms(C_ if False else C_[sel])) if False else Z*(target_rms/r))
            Bs.append(np.full(len(sel), side))
    X=np.vstack(Xs); C=np.vstack(As); side=np.concatenate(Bs)
    if coord=="shuffle":
        C=C[np.random.default_rng(7).permutation(len(C))]
    elif coord=="side_shuffle":   # keep each side's internal structure, but swap the two blocks' positions
        pass
    return X, C, side

print(f"{'variant':46s} {'NFS':>9s} {'mmd_u':>9s} {'vario':>9s} {'de_sc':>7s} {'de_dir':>7s}")
def sc(name,X,C):
    print(f"{name:46s} {neighborhood_mmd(X,C,tX,tC):9.5f} {mmd_unbiased(X,tX,seed=0):9.5f} {variogram_score(X,tX,seed=0):9.5f} {de_score(X,tX,XA)['score']:7.4f} {de_direction(X,tX,XA):7.4f}")
i=np.random.default_rng(0).choice(len(tX),N,replace=False)
sc("CEILING truth real 5000 @truth rms", tX[i], tC[i])
sc("CEILING truth real 5000 shuffled coords", tX[i], tC[i][np.random.default_rng(7).permutation(N)])
for src in ("mix","A","B"):
    X,C,s=build(src,"own"); sc(f"real cells src={src}, OWN coords @{RT:.0f}", X, C)
X,C,s=build("mix","shuffle"); sc("real cells src=mix, SHUFFLED coords", X, C)
# production pipeline on the same window
import subprocess
out="outputs/t2/heart/w2_pipeline.h5ad"
subprocess.run([str(ROOT/'.venv/bin/python'),"scripts/13_predict_t2.py","--setting","heart","--target","8.75",
  "--method","full","--shape","anisotropic","--ot-interp","--ot-x","pick","--ot-xyz","left","--ot-w","0.4",
  "--n","5000","--out",out],cwd=ROOT,capture_output=True)
if Path(out).exists():
    P=ad.read_h5ad(out); pX=dense(P); pC=np.asarray(P.obsm['spatial_3D'],float)
    sc(f"PRODUCTION pipeline (rms {rms(pC):.0f} native)", pX, pC)
    sc(f"PRODUCTION pipeline rescaled to {RT:.0f}", pX, to_rms(pC,RT))
    print("   pipeline scale_log_ratio vs truth:", round(scale_log_ratio(pC,tC),4))
