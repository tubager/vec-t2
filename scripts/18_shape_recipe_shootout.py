"""Shape-recipe shootout on bracketed proxies (official metrics, full group totals).
Expression construction held FIXED; only the coordinate recipe varies.
heart proxy:   L=E8.0  R=E8.75 T=E8.25_late
embryo proxy:  L=E6.75 R=E8.0  T=E7.25
"""
import sys, numpy as np; sys.path.insert(0,'/tmp/t2d')
sys.path.insert(0,'/Users/eric/Downloads/Embryo/src')
from harness import *
from common.core_metrics import split_half
from t2.geometry import canonicalize, scale_cloud, axis_stds, rms_radius
from t2.shape import (transform_cloud, match_axis_flips, occupancy_interpolate_cloud,
                      ot_interpolate_clouds, greedy_assign_xyz)
from t2.growth import axis_interp

NP = 5000
def run_proxy(tag, lp, rp, tp, w_clock, alpha):
    XL,aL=L(lp); XR,aR=L(rp); XT,aT=L(tp)
    CL,CR,CT=coords(aL),coords(aR),coords(aT)
    gl=list(aL.var_names); gr=list(aR.var_names); gt=list(aT.var_names)
    common=[g for g in gl if g in set(gr) and g in set(gt)]
    if len(common)<len(gl) or len(common)<len(gr) or len(common)<len(gt):
        il=[gl.index(g) for g in common]; ir=[gr.index(g) for g in common]; it=[gt.index(g) for g in common]
        XL=XL[:,il]; XR=XR[:,ir]; XT=XT[:,it]
        print(f"  gene panels aligned to {len(common)} common genes")
    rT=rms_radius(CT)
    print(f"\n===== {tag}: L={lp} R={rp} T={tp}  w_clock={w_clock}  n={NP} alpha={alpha} =====")
    print(f"  sizes L/R/T = {len(XL)}/{len(XR)}/{len(XT)}   rms L/R/T = {rms_radius(CL):.1f}/{rms_radius(CR):.1f}/{rT:.1f}")
    rng=np.random.default_rng(0)
    sub=rng.choice(len(XL),NP,replace=False)
    h1,h2=split_half(len(XT),0)
    step=pb(XR)-pb(XL)
    Xfix=np.clip(Xfix0(XL[sub],alpha,step),0,None) if False else np.clip(XL[sub]+alpha*step,0,None)
    FLc=scale_cloud(CL[sub],rT)
    LIBd=dict(de_score=False,de_direction=False,mmd_u=True,variogram=True,d2_shape=True,
              occupancy_dice=False,scale_log_ratio=True,neighborhood_mmd=True)
    from common.shape_metrics import d2_distance, scale_log_ratio, occupancy_dice
    def raw(X,C,TX,TC,RX):
        de=de_score(X,TX,RX); d,vox=occupancy_dice(C,TC)
        return dict(de_score=de['score'],de_direction=de_direction(X,TX,RX),
            mmd_u=mmd_unbiased(X,TX),variogram=variogram_score(X,TX),
            d2_shape=d2_distance(C,TC),occupancy_dice=d,scale_log_ratio=abs(scale_log_ratio(C,TC)),
            neighborhood_mmd=neighborhood_mmd(X,C,TX,TC))
    FL=raw(XL[sub],FLc,XT,CT,XL)
    c1=raw(XT[h1],CT[h1],XT,CT,XL); c2=raw(XT[h2],CT[h2],XT,CT,XL)
    CE={k:(c1[k]+c2[k])/2 for k in c1}
    from common.core_metrics import skill as _sk
    GROUPS=[('expr',['de_score','de_direction']),('state',['mmd_u','variogram']),
            ('shape',['d2_shape','occupancy_dice','scale_log_ratio']),('local',['neighborhood_mmd'])]
    def tot(row):
        g={n:float(np.mean([_sk(row[m],FL[m],CE[m],LIBd[m])*100 for m in ms])) for n,ms in GROUPS}
        return float(np.mean(list(g.values()))),g
    Zi,ri,_=canonicalize(CL); Zr,rr,_=canonicalize(CR)
    Zr=match_axis_flips(Zi,Zr,rng,n_pair=800)
    stdL=axis_stds(CL); stdR=axis_stds(Zr*rr)
    recs={}
    for w in [0.0,0.25,0.33,0.5,0.67,1.0]:
        at=axis_interp(0.0,1.0,w,stdL,stdR)
        Y=transform_cloud(CL,rT,mode='anisotropic',axis_std_target=at)
        idx=rng.choice(len(Y),NP,replace=len(Y)<NP)
        recs[f'S2_aniso_w{w:g}']=Y[idx]
    Zu=(Zi/max(ri,1e-8)).astype(np.float32); Zru=(Zr/max(rr,1e-8)).astype(np.float32)
    for w in [0.33,0.5,0.67]:
        for nb in [24,48]:
            P=occupancy_interpolate_cloud(Zu,Zru,w,NP,rng,n_bins=nb)
            recs[f'S3_occ_w{w:g}_b{nb}']=scale_cloud(np.asarray(P,float),rT)
    m=min(10000,len(CL),len(CR))
    il=rng.choice(len(CL),m,replace=False); ir=rng.choice(len(CR),m,replace=False)
    for w in [0.33,0.5,0.67]:
        Y=ot_interpolate_clouds(CL[il],CR[ir],w,rT,n_pair=800,rng=rng)
        idx=rng.choice(len(Y),NP,replace=len(Y)<NP)
        recs[f'S4_otinterp_w{w:g}']=Y[idx]
    recs['copyR']=scale_cloud(CR[rng.choice(len(CR),NP,replace=len(CR)<NP)],rT)
    t,g=tot(raw(Xfix,FLc,XT,CT,XL))
    print(f"  {'FLOOR copyL(alpha-shifted X)':30s} TOTAL {t:6.2f}  "+" ".join(f"{k}={v:.1f}" for k,v in g.items()))
    print(f"  {'recipe':30s} {'d2':>8s} {'ODS':>7s} {'NFS':>8s} | {'expr':>6s} {'state':>6s} {'shape':>6s} {'local':>6s} {'TOTAL':>7s}")
    out=[]
    for name,P in recs.items():
        C=greedy_assign_xyz(CL[sub],np.asarray(P,np.float64))
        r=raw(Xfix,C,XT,CT,XL); t,g=tot(r)
        print(f"  {name:30s} {r['d2_shape']:8.5f} {r['occupancy_dice']:7.4f} {r['neighborhood_mmd']:8.5f} | "
              f"{g['expr']:6.1f} {g['state']:6.1f} {g['shape']:6.1f} {g['local']:6.1f} {t:7.2f}")
        out.append((t,name,r,g))
    out.sort(key=lambda z:-z[0])
    print(f"  WINNER {tag}: {out[0][1]}  total {out[0][0]:.2f}")
    return out

res={}
res['heart']=run_proxy('HEART','data/E8.0.h5ad','data/E8.75.h5ad','data/E8.25_late.h5ad',0.5,0.348)
res['embryo']=run_proxy('EMBRYO','data/E6.75.h5ad','data/E8.0.h5ad','data/E7.25.h5ad',0.4,0.4)
print("\n===== cross-proxy summary (top6 by TOTAL) =====")
for k in ['heart','embryo']:
    print(f"-- {k}: "+" | ".join(f"{n}:{t:.1f}" for t,n,_,_ in res[k][:6]))
