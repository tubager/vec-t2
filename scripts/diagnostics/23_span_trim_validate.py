"""Validate the `auto` (target-nocc) trim against REAL TRUTH, including the case where it must self-disable."""
import pathlib
import sys, numpy as np, anndata as ad, importlib.util
from pathlib import Path
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'.venv/lib/python3.13/site-packages'))
import score_h5ad as _sh
_t2m,_=_sh._load_task_metrics("T2")
from common.shape_metrics import occupancy_dice, d2_distance
nfs=_t2m.neighborhood_mmd
spec=importlib.util.spec_from_file_location('st',str(pathlib.Path(__file__).resolve().parents[1]/'20_span_trim.py'))
st=importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
ROOT = pathlib.Path(__file__).resolve().parents[2]
def coords(p):
    a=ad.read_h5ad(p); z=a.obsm['spatial_3D']
    return np.asarray(z.todense() if hasattr(z,'todense') else z,float)
def X(p,g=None):
    a=ad.read_h5ad(p)
    if g: a=a[:,g].copy()
    M=a.X; return np.asarray(M.todense() if hasattr(M,'todense') else M,float)
def pipeline(pred_C, anchors, w, extrap, rms, n, seed=0, tag=''):
    Z0,rms0,Vt,mu=st.canonicalise(pred_C)
    out={'0 no-trim':pred_C}
    n_anchor=n
    tgt,counts=st.target_nocc([Path(a) for a in anchors],n_anchor,w,seed,extrap)
    U=st.allowed_set([Path(a) for a in anchors],'union',seed,n_anchor)
    I=st.allowed_set([Path(a) for a in anchors],'intersect',seed,n_anchor)
    nocc0=len(st.occupied(Z0))
    if nocc0<=tgt:
        Z1,mv=Z0.copy(),0
    else:
        Z1,mv=st.trim_to_nocc(Z0.copy(),U,I,tgt,0.02,seed)
    C1=(Z1*rms0)@Vt+mu; c=C1.mean(0); C1=c+(C1-c)*(rms/st.rms_radius(C1))
    out[f'1 auto(target {tgt:.0f}, nocc {nocc0})']=C1
    for lbl,allow in [('2 union-all',U),('3 intersect-all',I)]:
        Z2,mv2=st.relocate(Z0.copy(),allow,0.02,seed,4)
        C2=(Z2*rms0)@Vt+mu; c=C2.mean(0); C2=c+(C2-c)*(rms/st.rms_radius(C2))
        out[f'{lbl}(mv {mv2})']=C2
    return out,counts,tgt,nocc0
def evaluate(lbl,variants,Xp,Ct,seeds=(0,1,2)):
    print(f"{'variant':40s}{'ODS':>22s}{'d2':>22s}{'NFS':>14s}")
    base=None
    for k,C in variants.items():
        od=[occupancy_dice(C,Ct,seed=s)[0] for s in seeds]
        d2=[d2_distance(C,Ct,seed=s) for s in seeds]
        nf=nfs(Xp,C,Xp[:,:0].reshape(0,0) if False else Xt_g,Ct) if False else None
        print(f"{k:40s}{np.mean(od):11.4f}+-{np.std(od):.4f}{np.mean(d2):11.5f}+-{np.std(d2):.5f}")
        if base is None: base=(np.mean(od),np.mean(d2))
        else: print(f"{'':40s}{np.mean(od)-base[0]:+11.4f}{'':11s}{np.mean(d2)-base[1]:+11.5f}")
print("########## HEART proxy: recipe output for target E8.75, truth E8.75, anchors E8.25/E9.5, w=0.4")
p='/tmp/t2e/proxy/hrt_p875.h5ad'  # scripts/13_predict_t2.py --setting heart --target 8.75 ...
Cp=coords(p); Xp=X(p,open(ROOT/'panels/T2__heart__val_interp.genes.txt').read().split())
Ct_all=coords(ROOT/'data/E8.75.h5ad'); n=5000
Ct=Ct_all[np.random.default_rng(7).choice(len(Ct_all),n,replace=False)]
v,counts,tgt,nocc0=pipeline(Cp,[ROOT/'data/E8.25_late.h5ad',ROOT/'data/E9.5.h5ad'],0.4,False,st.rms_radius(Cp),n)
print(f"  anchor nocc={counts} target={tgt:.1f} submitted nocc={nocc0} truth nocc={len(st.occupied(st.canonicalise(Ct)[0]))}")
evaluate('',v,Xp,Ct)
print("\n########## EMBRYO proxy: recipe output for target E7.25, truth E7.25, anchors E6.75/E8.0, w=0.4")
p='/tmp/t2e/proxy/emb_p725.h5ad'  # scripts/13_predict_t2.py --setting embryo --target 7.25 ...
Cp=coords(p); Xp=X(p,open(ROOT/'panels/T2__embryo__val_interp.genes.txt').read().split())
Ct_all=coords(ROOT/'data/E7.25.h5ad'); Ct=Ct_all[np.random.default_rng(7).choice(len(Ct_all),n,replace=False)]
v,counts,tgt,nocc0=pipeline(Cp,[ROOT/'data/E6.75.h5ad',ROOT/'data/E8.0.h5ad'],0.4,False,st.rms_radius(Cp),n)
print(f"  anchor nocc={counts} target={tgt:.1f} submitted nocc={nocc0} truth nocc={len(st.occupied(st.canonicalise(Ct)[0]))}")
evaluate('',v,Xp,Ct)
print("\n########## EMBRYO-analogue NARROW proxy: OT-interp cloud, truth E8.25, anchors E8.0/E8.75, w=1/3")
from scipy.spatial import cKDTree
rng=np.random.default_rng(0)
CL=coords(ROOT/'data/E8.0.h5ad'); CR=coords(ROOT/'data/E8.75.h5ad'); CT=coords(ROOT/'data/E8.25_late.h5ad')
w=1/3
L=CL[rng.choice(len(CL),n,replace=False)]
Rs=CR[rng.choice(len(CR),8000,replace=False)]
R=Rs[cKDTree(Rs).query(L)[1]]
P=(1-w)*L+w*R; r=np.exp((1-w)*np.log(st.rms_radius(L))+w*np.log(st.rms_radius(CR)))
c=P.mean(0); P=c+(P-c)*(r/st.rms_radius(P))
Ct=CT[np.random.default_rng(7).choice(len(CT),n,replace=False)]
v,counts,tgt,nocc0=pipeline(P,[ROOT/'data/E8.0.h5ad',ROOT/'data/E8.75.h5ad'],w,False,r,n)
print(f"  anchor nocc={counts} target={tgt:.1f} submitted nocc={nocc0} truth nocc={len(st.occupied(st.canonicalise(Ct)[0]))}")
evaluate('',v,None,Ct)
