"""CompositionT2.probs() interpolates cell-type proportions ARITHMETICALLY.
Cell populations grow/shrink multiplicatively, so test geometric (log-linear) and other estimators
against the REAL composition at held-out stages."""
import sys, numpy as np, anndata as ad
from collections import Counter
import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[2]) + '/'
def pi(stage):
    a=ad.read_h5ad(ROOT+'data/%s.h5ad'%stage)
    lab=np.asarray(a.obs['celltype'].values)
    c=Counter(lab); n=len(lab)
    return {k:v/n for k,v in c.items()}, n
S={s:pi(s) for s in ['E6.75','E7.25','E8.0','E8.25_late','E8.75','E9.5']}
def vec(d, keys): return np.array([d.get(k,0.0) for k in keys])
def l1(a,b): return float(np.abs(a-b).sum())
def jsd(p,q):
    p=np.clip(p,1e-12,None); q=np.clip(q,1e-12,None); m=0.5*(p+q)
    kl=lambda x,y: float((x*np.log(x/y)).sum())
    return 0.5*kl(p,m)+0.5*kl(q,m)
def renorm(v): 
    v=np.clip(v,0,None); s=v.sum(); return v/s if s>0 else v
def estimators(pl,pr,w,keys,extra=None,t_extra=None,t_l=None,t_r=None,t_tgt=None):
    vl,vr=vec(pl,keys),vec(pr,keys)
    out={}
    out['arith (CURRENT)']=(1-w)*vl+w*vr
    eps=1e-9
    out['geometric loglin']=renorm(np.exp((1-w)*np.log(vl+eps)+w*np.log(vr+eps)))
    out['sqrt-geometric']=renorm(np.sqrt(np.clip(vl,0,None)*np.clip(vr,0,None)))
    # power/geometric with temperature
    for tau in (0.5,2.0):
        out[f'geometric tau={tau}']=renorm(np.exp((1-w)*np.log(vl+eps)+w*np.log(vr+eps))**tau)
    # Hellinger / chordal interpolation (unit-sphere)
    a=np.sqrt(np.clip(vl,0,None)); b=np.sqrt(np.clip(vr,0,None))
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    out['hellinger chord']=renorm(((1-w)*a+w*b)**2)
    if extra is not None:
        # quadratic in log-space through left, right and one extra stage
        ve=vec(extra,keys)
        te=t_extra; tl,tr,tt=t_l,t_r,t_tgt
        def lag(y_l,y_r,y_e,t):
            return (y_l*(t-tr)*(t-te)/((tl-tr)*(tl-te)) + y_r*(t-tl)*(t-te)/((tr-tl)*(tr-te))
                    + y_e*(t-tl)*(t-tr)/((te-tl)*(te-tr)))
        out['log-quadratic(+extra)']=renorm(np.exp(lag(np.log(vl+eps),np.log(vr+eps),np.log(ve+eps),tt)))
        out['arith-quadratic(+extra)']=renorm(lag(vl,vr,ve,tt))
    return out
cases=[
 ('EMBRYO proxy A  tgt E7.25  anchors E6.75/E8.0  w=0.400','E7.25','E6.75','E8.0',0.4,None,None),
 ('EMBRYO narrow   tgt E8.25  anchors E8.0/E8.75  w=0.333','E8.25_late','E8.0','E8.75',1/3,None,None),
 ('EMBRYO wide     tgt E8.0   anchors E7.25/E8.75 w=0.167','E8.0','E7.25','E8.75',0.16667,None,None),
 ('HEART proxy     tgt E8.75  anchors E8.25/E9.5  w=0.400','E8.75','E8.25_late','E9.5',0.4,'E8.0',8.0),
 ('HEART narrow    tgt E8.25  anchors E8.0/E8.75  w=0.333','E8.25_late','E8.0','E8.75',1/3,'E9.5',9.5),
]
tl_map={'E6.75':6.75,'E7.25':7.25,'E8.0':8.0,'E8.25_late':8.25,'E8.75':8.75,'E9.5':9.5}
for tag,tgt,l,r,w,extra,te in cases:
    pt,nt=S[tgt]; pl,_=S[l]; pr,_=S[r]
    keys=sorted(set(pl)|set(pr)|set(pt))
    vt=vec(pt,keys)
    print(f"\n===== {tag}   n_true={nt}  n_types={len(keys)}")
    rows=estimators(pl,pr,w,keys, S[extra][0] if extra else None, te, tl_map[l], tl_map[r], tl_map[tgt])
    if extra: rows['copy left (floor)']=vec(pl,keys); rows['copy right']=vec(pr,keys)
    res=[(k,l1(renorm(v),vt),jsd(renorm(v),vt)) for k,v in rows.items()]
    base=[x for x in res if x[0]=='arith (CURRENT)'][0]
    for k,a,j in sorted(res,key=lambda x:x[1]):
        print(f"  {k:26s} L1={a:.4f} ({a-base[1]:+.4f} vs current)  JSD={j:.5f}")
