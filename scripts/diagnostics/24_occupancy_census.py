"""occupancy_dice = 2|A&B|/(|A|+|B|) on a 16^3 canonical grid. It is an OCCUPIED-VOXEL-COUNT statistic.
Measure our submissions' occupied count vs the anchors', at the n the board actually scores."""
import pathlib
import sys, numpy as np, anndata as ad
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'.venv/lib/python3.13/site-packages'))
from common.shape_metrics import _canonicalise, occupancy_dice, d2_distance
ROOT = str(pathlib.Path(__file__).resolve().parents[2]) + '/'
def coords(p):
    a=ad.read_h5ad(p); z=a.obsm['spatial_3D']
    return np.asarray(z.todense() if hasattr(z,'todense') else z, float)
def occset(Z,n_grid=16,extent=3.0):
    v=np.floor((np.clip(Z,-extent,extent-1e-9)+extent)/(2*extent)*n_grid).astype(int)
    return set(map(tuple,v))
def report(name,C,n_sub=None,seed=0):
    if n_sub and len(C)>n_sub:
        C=C[np.random.default_rng(seed).choice(len(C),n_sub,replace=False)]
    Z,rms=_canonicalise(C); o=occset(Z)
    # radial profile of canonical coords
    r=np.linalg.norm(Z,axis=1)
    return dict(name=name,n=len(C),nocc=len(o),rms=rms,q50=np.percentile(r,50),q90=np.percentile(r,90),
                q99=np.percentile(r,99),rmax=r.max(),Z=Z,o=o,r=r)
S={}
files={
 'SUB heart_interp':ROOT+'outputs/t2/submit/T2_heart_val_interp.h5ad',
 'SUB embryo_interp':ROOT+'outputs/t2/submit/T2_embryo_val_interp.h5ad',
 'SUB heart_extrap':ROOT+'outputs/t2/submit/T2_heart_val_extrap.h5ad',
 'E8.25_late':ROOT+'data/E8.25_late.h5ad','E8.75':ROOT+'data/E8.75.h5ad','E9.5':ROOT+'data/E9.5.h5ad',
 'E6.75':ROOT+'data/E6.75.h5ad','E7.25':ROOT+'data/E7.25.h5ad','E8.0':ROOT+'data/E8.0.h5ad'}
print("NOTE: every cloud is subsampled to n=5000 so the counts are comparable;\n      the extrap board is actually scored at n=25179 (there: submitted 251 vs anchors 246/257 => already on target).")
print(f"{'cloud':20s}{'n':>7s}{'nocc':>6s}{'q50':>7s}{'q90':>7s}{'q99':>7s}{'rmax':>7s}")
for n5000 in (True,):
    for k,p in files.items():
        r=report(k,coords(p), 5000); S[k]=r
        print(f"{k:20s}{r['n']:7d}{r['nocc']:6d}{r['q50']:7.3f}{r['q90']:7.3f}{r['q99']:7.3f}{r['rmax']:7.3f}")
print("\n== dice at n=5000 among real clouds (a proxy for what truth-vs-anchor looks like) ==")
def dice(a,b):
    oa,ob=S[a]['o'],S[b]['o']; return 2*len(oa&ob)/(len(oa)+len(ob))
keys=list(files)
print('      '+''.join(f"{k[:11]:>13s}" for k in keys))
for a in keys:
    print(f"{a[:13]:14s}"+''.join(f"{dice(a,b):13.3f}" for b in keys))
print("\n== d2 at n=5000 ==")
for a in ['E8.25_late','E8.75','E9.5']:
    for b in ['SUB heart_interp']:
        print(f"  d2({a},{b})={d2_distance(S[a]['Z'][:0].reshape(0,3) if False else S[a]['Z'],S[b]['Z']):.5f}")
