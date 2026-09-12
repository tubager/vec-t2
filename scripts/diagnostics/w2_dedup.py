import sys
from pathlib import Path
ROOT=Path("/Users/eric/Downloads/Embryo")
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/".venv/lib/python3.13/site-packages")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))  # _reuse shim for T2.metrics
import numpy as np, anndata as ad
from t2.expression import labels_to_clusters
from common.core_metrics import mmd_unbiased, variogram_score, de_score, de_direction
from T2.metrics import neighborhood_mmd
def dense(a):
    X=a.X; return np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
A=ad.read_h5ad("data/E8.25_late.h5ad"); B=ad.read_h5ad("data/E9.5.h5ad"); T=ad.read_h5ad("data/E8.75.h5ad")
XA,XB,XT=dense(A),dense(B),dense(T); tC=np.asarray(T.obsm['spatial_3D'],float)
cA=np.array(labels_to_clusters(A.obs['celltype'],'heart')); cB=np.array(labels_to_clusters(B.obs['celltype'],'heart'))
P=ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad"); X=dense(P).copy(); C=np.asarray(P.obsm['spatial_3D'],float)
# identify the source (anchor, row) of every submitted row by exact match
def build_index(Xs):
    d={}
    for i,r in enumerate(Xs): d.setdefault(r.tobytes(),[]).append(i)
    return d
iA,iB=build_index(XA),build_index(XB)
src=[]; used=set()
for r in X:
    k=r.tobytes()
    if k in iA: src.append(("A",iA[k]))
    elif k in iB: src.append(("B",iB[k]))
    else: src.append((None,None))
print("rows matched to an anchor:",sum(1 for s,_ in src if s),"of",len(src))
# duplicate groups: for each repeated row, replace extra copies with unused cells of the SAME cluster from the SAME anchor
from collections import defaultdict
groups=defaultdict(list)
for i,(s,idxs) in enumerate(src):
    if s is None: continue
    groups[(s,idxs[0])].append(i)
rng=np.random.default_rng(3)
X2=X.copy(); nrep=0
pool_used={("A",i) for i in range(0)}  # placeholder
usedA=set(); usedB=set()
for (s,idxs) in list(groups):
    pass
# mark which anchor rows are already used by the submission
for (s,idxs) in src:
    if s is None: continue
for key,rows in groups.items():
    if len(rows)<2: continue
    s,first=key
    # the anchor row actually used for this submitted row (find which of idxs matches by cluster availability)
    cl = (cA if s=="A" else cB)[first]
    Xs = XA if s=="A" else XB
    cs = cA if s=="A" else cB
    used_idx = {idxs[0]}
    cand=[i for i in np.flatnonzero(cs==cl) if i not in used_idx]
    rng.shuffle(cand)
    for r_i in rows[1:]:
        if not cand: break
        j=cand.pop(); X2[r_i]=Xs[j]; nrep+=1
print(f"replaced {nrep} duplicate rows with fresh same-cluster real cells")
u1=len({r.tobytes() for r in X}); u2=len({r.tobytes() for r in X2})
print(f"unique rows: before {u1}  after {u2}")
def sc(nm,Xv):
    print(f"{nm:34s} de_sc={de_score(Xv,XT,XA)['score']:.4f} de_dir={de_direction(Xv,XT,XA):.4f} mmd={mmd_unbiased(Xv,XT,seed=0):.5f} var={variogram_score(Xv,XT,seed=0):.5f} NFS={neighborhood_mmd(Xv,C,XT,tC):.5f}")
sc("pipeline as-is (24% dups)", X)
sc("dup rows -> fresh real cells", X2)
# control: same number of random same-cluster swaps (to check the gain is not just 'more diversity')
X3=X.copy(); rng2=np.random.default_rng(9); n=0
for key,rows in groups.items():
    s,first=key
    if len(rows)<2: continue
    cl=(cA if s=="A" else cB)[first]; Xs=XA if s=="A" else XB; cs=cA if s=="A" else cB
    cand=np.flatnonzero(cs==cl); 
    for r_i in rows[1:]:
        if n>=nrep: break
        X3[r_i]=Xs[cand[rng2.integers(len(cand))]]; n+=1
sc(f"control: {n} random same-cluster swaps", X3)
