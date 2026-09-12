"""Does a multi-point per-gene trajectory beat the 2-point linear delta, on the OFFICIAL expr metrics?
expr group (25% of total) depends ONLY on pb(predX)-pb(refX) ranks -> we can score estimators exactly."""
import pathlib
import sys, numpy as np, anndata as ad
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'.venv/lib/python3.13/site-packages'))
from common.core_metrics import de_score, de_direction, mmd_unbiased, variogram_score, pseudobulk
ROOT = str(pathlib.Path(__file__).resolve().parents[2]) + '/'
def panel(k): return open(ROOT+'panels/'+k).read().split()
def load(stage, genes):
    a=ad.read_h5ad(ROOT+'data/%s.h5ad'%stage); a=a[:,[g for g in genes if g in set(a.var_names)]].copy()
    X=a.X; X=np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
    return X.astype(np.float64), np.asarray(a.obs['celltype'].values)
EMB=panel('T2__embryo__val_interp.genes.txt'); HRT=panel('T2__heart__val_interp.genes.txt')
C={}
def cells(stage, genes, tag):
    k=(stage,tag)
    if k not in C: C[k]=load(stage,genes)
    return C[k]
def lagrange(ts, ps, t):
    """vector-valued Lagrange interpolation/extrapolation at t over times ts (list), ps (list of vectors)"""
    out=np.zeros_like(ps[0]); 
    for i,ti in enumerate(ts):
        w=1.0
        for j,tj in enumerate(ts):
            if i!=j: w*= (t-tj)/(ti-tj)
        out=out+w*ps[i]
    return out
def loclin(ts, ps, t, h):
    """Gaussian-weighted local linear fit in time, per gene."""
    w=np.array([np.exp(-0.5*((ti-t)/h)**2) for ti in ts])
    T=np.array(ts); num0=(w).sum(); num1=(w*(T-t)).sum(); num2=(w*(T-t)**2).sum()
    s0=(w*ps).sum(0) if False else sum(wi*p for wi,p in zip(w,ps))
    s1=sum(wi*(ti-t)*p for wi,ti,p in zip(w,T,ps))
    det=num0*num2-num1*num1
    a=(num2*s0-num1*s1)/det
    return a
def score_est(name, p_target, trueX, refX, tag_de):
    dp=p_target-pseudobulk(refX); dt=pseudobulk(trueX)-pseudobulk(refX)
    from scipy.stats import spearmanr, rankdata
    rho=spearmanr(dp,dt).statistic
    M=np.tile(p_target,(64,1))
    de=de_score(M,trueX,refX); dd=de_direction(M,trueX,refX)
    print(f"  {name:26s} de={de['score']:+.4f} raw={de['raw']:.4f} ddir={dd:+.4f} rho={rho:+.4f} |d|={np.linalg.norm(dp):7.2f} cos={float(dp@dt/np.linalg.norm(dp)/np.linalg.norm(dt)):+.4f}")
    return de['score'], dd

def run(tag, genes, ref, right, target, extras, w=None, tt=None):
    print(f"\n===== {tag}: ref={ref} right={right} target={target} extras={extras} panel={len(genes)}")
    Xr,_=cells(ref,genes,'x'); Xt,_=cells(target,genes,'x')
    times=[]; pbs=[]
    for s in [ref]+extras+[right]:
        X,_=cells(s,genes,'x'); times.append(tt[s]); pbs.append(pseudobulk(X))
    order=np.argsort(times); times=[times[i] for i in order]; pbs=[pbs[i] for i in order]
    t=tt[target]; pref=pseudobulk(Xr)
    print("  times:",[round(x,3) for x in times], " (truth at %.3f)"%t)
    res={}
    res['copy_ref']=score_est('copy pb(ref)', pref, Xt, Xr, tag)
    i_l=times.index(tt[ref]); i_r=times.index(tt[right])
    wl=(t-times[i_l])/(times[i_r]-times[i_l])
    p2=pref+wl*(pbs[i_r]-pref)
    res['lin2']=score_est(f'lin2 bracket w={wl:.3f}', p2, Xt, Xr, tag)
    # every 3-point quadratic that includes both bracket ends
    for k,tk in enumerate(times):
        if k in (i_l,i_r): continue
        p3=lagrange([times[i_l],times[i_r],tk],[pbs[i_l],pbs[i_r],pbs[k]],t)
        res['quad_%s'%tk]=score_est(f'quad3 +{tk}', p3, Xt, Xr, tag)
        for lam in (0.3,0.5,0.7):
            res['bl_%s_%s'%(tk,lam)]=score_est(f'  blend lin2/quad+{tk} l={lam}', (1-lam)*p2+lam*p3, Xt, Xr, tag)
    if len(times)>=4:
        p4=lagrange(times,pbs,t); res['poly_all']=score_est('polyall(%d pts)'%len(times), p4, Xt, Xr, tag)
    for h in (0.3,0.5,0.8,1.2):
        res['loclin_%s'%h]=score_est(f'locLin h={h}', loclin(times,pbs,t,h), Xt, Xr, tag)
    # one-step-extrap style (for the extrap board): last observed MERFISH slope
    if len(times)>=2:
        step=times[-1]-times[-2]; sl=pbs[-1]-pbs[-2]
        for a in (0.5,1.0,1.5):
            res['step_%s'%a]=score_est(f'lastStep x{a} ({times[-2]}->{times[-1]})', pbs[-1]+a*(t-times[-1])/step*sl, Xt, Xr, tag)
    # remove the expression-magnitude component from the delta (de_score's null direction)
    for c in (0.05,0.1,0.2,0.4):
        d=p2-pref; d=d-c*(pref-pref.mean())*np.linalg.norm(d)/max(np.linalg.norm(pref-pref.mean()),1e-9)
        res['demag_%s'%c]=score_est(f'lin2 -{c}*pb_ref-dir', pref+d, Xt, Xr, tag)
    best=max(res.items(), key=lambda kv: kv[1][0]); print(f"  >>> BEST by de_score: {best[0]}  de={best[1][0]:+.4f}")
    best2=max(res.items(), key=lambda kv: (kv[1][0]+kv[1][1])); print(f"  >>> BEST by de+ddir : {best2[0]}  de={best2[1][0]:+.4f} ddir={best2[1][1]:+.4f}")
    return res

if __name__=='__main__':
    tt={'E6.75':6.75,'E7.25':7.25,'E8.0':8.0,'E8.25_late':8.25,'E8.75':8.75,'E9.5':9.5}
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    if which in ('all','emb'):
        run('EMBRYO-proxyA (existing proxy)', EMB,'E6.75','E8.0','E7.25',['E8.25_late'],tt=tt)
        run('EMBRYO-proxyB (mirror of real)', EMB,'E7.25','E8.25_late','E8.0',['E6.75','E8.75'],tt=tt)
    if which in ('all','hrt'):
        run('HEART-proxy (target E8.75)', HRT,'E8.25_late','E9.5','E8.75',['E8.0'],tt=tt)
    if which in ('all','ext'):
        run('EXTRAP-proxy (target E9.5)', HRT,'E8.75','E9.5','E9.5',['E8.0','E8.25_late'],tt=tt)
