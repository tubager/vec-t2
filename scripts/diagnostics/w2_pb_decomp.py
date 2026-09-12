import sys
from pathlib import Path
ROOT=Path("/Users/eric/Downloads/Embryo")
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/".venv/lib/python3.13/site-packages"))
import numpy as np, anndata as ad
from t2.expression import labels_to_clusters, cluster_order, CompositionT2
def dense(a):
    X=a.X; return np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
A=ad.read_h5ad("data/E8.25_late.h5ad"); B=ad.read_h5ad("data/E9.5.h5ad"); T=ad.read_h5ad("data/E8.75.h5ad")
XA,XB,XT=dense(A),dense(B),dense(T)
cA=np.array(labels_to_clusters(A.obs['celltype'],'heart')); cB=np.array(labels_to_clusters(B.obs['celltype'],'heart'))
cT=np.array(labels_to_clusters(T.obs['celltype'],'heart'))
order=cluster_order('heart'); w=0.4
pt=XT.mean(0)
def cmeans(X,c):
    return {k:(X[c==k].mean(0) if (c==k).sum() else None) for k in order}
mA,mB,mT=cmeans(XA,cA),cmeans(XB,cB),cmeans(XT,cT)
piA={k:(cA==k).mean() for k in order}; piB={k:(cB==k).mean() for k in order}; piT={k:(cT==k).mean() for k in order}
comp=CompositionT2("heart",{"A":8.25,"B":9.5}); comp.fit({"A":A.obs['celltype'],"B":B.obs['celltype']},{"A":8.25,"B":9.5})
pi_est=comp.probs(8.75)
def mix(pi):
    out=np.zeros(XA.shape[1]); tot=0.0
    for k in order:
        p=pi.get(k,0.0)
        if p<=0: continue
        a,b=mA[k],mB[k]
        if a is None and b is None: continue
        if a is None: m=b
        elif b is None: m=a
        else: m=(1-w)*a+w*b
        out+=p*m; tot+=p
    return out/tot
rel=lambda v: np.linalg.norm(v-pt)/np.linalg.norm(pt)
print("pb_rel_err decompositions (W2, truth E8.75):")
print("  oracle-composition + anchor cluster means : %.4f"%rel(mix(piT)))
print("  estimated composition + anchor cluster means: %.4f"%rel(mix(pi_est)))
print("  oracle composition + ORACLE cluster means : %.4f"%rel(sum(piT[k]*mT[k] for k in order if piT[k]>0 and mT[k] is not None)/sum(piT[k] for k in order if piT[k]>0 and mT[k] is not None)))
print("  anchor A pb alone                         : %.4f"%rel(XA.mean(0)))
print("  anchor B pb alone                         : %.4f"%rel(XB.mean(0)))
print("  linear pb interpolation (clock)           : %.4f"%rel(XA.mean(0)+w*(XB.mean(0)-XA.mean(0))))
P=ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad"); px=dense(P).mean(0)
print("  OUR pipeline submission                   : %.4f"%rel(px))
s=float(px@pt/(px@px)); print("  OURS after optimal GLOBAL scale (%.4f)     : %.4f"%(s,rel(s*px)))
sc=float(mix(pi_est)@pt/(mix(pi_est)@mix(pi_est))); print("  est-comp mix after optimal global scale   : %.4f"%rel(sc*mix(pi_est)))
# composition L1 errors
l1=lambda p,q: sum(abs(p[k]-q[k]) for k in order)/2
print("composition L1/2: est vs truth %.4f | A vs truth %.4f | B vs truth %.4f"%(l1(pi_est,piT),l1(piA,piT),l1(piB,piT)))
print("cluster means availability in B (missing):",[k for k in order if mB[k] is None])
# per-cluster level bias between anchors and truth (dataset effect?), for clusters present in all
common=[k for k in order if mA[k] is not None and mB[k] is not None and mT[k] is not None]
print("clusters present in all three:",len(common))
R=[]
for k in common:
    for nm,m in (("A",mA[k]),("B",mB[k])):
        R.append(np.log((m+1e-3)/(mT[k]+1e-3)))
R=np.array(R)
print("log-ratio to truth per cluster/dataset: mean %.3f sd %.3f | corr across datasets of same cluster %s"%(
    R.mean(), R.std(), ""))
# is the A-vs-truth log ratio gene-wise consistent across clusters (dataset bias) ?
ra=np.array([np.log((mA[k]+1e-3)/(mT[k]+1e-3)) for k in common])
rb=np.array([np.log((mB[k]+1e-3)/(mT[k]+1e-3)) for k in common])
print("A-vs-truth log-ratio: cross-cluster corr (mean pairwise) %.3f"%np.mean([np.corrcoef(ra[i],ra[j])[0,1] for i in range(len(ra)) for j in range(i+1,len(ra))]))
print("B-vs-truth log-ratio: cross-cluster corr (mean pairwise) %.3f"%np.mean([np.corrcoef(rb[i],rb[j])[0,1] for i in range(len(rb)) for j in range(i+1,len(rb))]))
print("A-vs-B log-ratio:     cross-cluster corr (mean pairwise) %.3f"%np.mean([np.corrcoef(ra[i]-rb[i],ra[j]-rb[j])[0,1] for i in range(len(ra)) for j in range(i+1,len(ra))]))
