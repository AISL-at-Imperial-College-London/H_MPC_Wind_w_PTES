"""Test the precise Algorithm 2 choice and postselection redistribution."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'forecast_release/source_releases/horse_creek_forecasting_release'))
from original_backward_reduction import backward_reduce_original
from scipy.spatial.distance import cdist

def _bruteforce(w,p,keep):
    n=len(w);x=np.c_[ (w-w.mean())/max(w.std(),1e-10),(p-p.mean())/max(p.std(),1e-10) ]
    d=cdist(x,x);np.fill_diagonal(d,np.inf)
    active=list(range(n));deleted=[]
    while len(active)>keep:
        costs=[]
        for l in active:
            rem=[j for j in active if j!=l]
            costs.append(sum(d[j,rem].min()/n for j in deleted+[l]))
        l=active.pop(int(np.argmin(costs)));deleted.append(l)
    mass=np.ones(len(active))/n
    for j in deleted:mass[np.argmin(d[j,active])]+=1/n
    return np.array(active),mass

def test_matches_original_pseudocode_on_small_random_cases():
    for seed in range(8):
        rng=np.random.default_rng(seed)
        w=rng.normal(size=(12,7));p=rng.normal(size=(12,7))
        ids,weights=backward_reduce_original(w,p,keep=4)
        expected,ew=_bruteforce(w,p,keep=4)
        assert np.array_equal(ids,expected)
        assert np.allclose(weights,ew)
        assert np.isclose(weights.sum(),1)

def test_reducer_preserves_selected_original_paths():
    w=np.array([[1,2],[3,4],[5,6],[7,8],[9,10]],float)
    p=np.array([[7,8],[5,6],[3,4],[1,2],[-1,0]],float)
    ids,weights=backward_reduce_original(w,p,2)
    assert len(set(ids))==2 and len(weights)==2
    assert np.isclose(weights.sum(),1)

def test_prefix_does_not_influence_hl_only_reduction():
    rng=np.random.default_rng(38)
    hlw=rng.normal(size=(40,84));hlp=rng.normal(size=(40,84))
    prefix_w=rng.normal(size=(40,15))*1e7
    prefix_p=rng.normal(size=(40,15))*1e9
    fullw=np.concatenate([prefix_w,hlw],axis=1)
    fullp=np.concatenate([prefix_p,hlp],axis=1)
    a,aw=backward_reduce_original(hlw,hlp,keep=8)
    b,bw=backward_reduce_original(fullw[:,15:],fullp[:,15:],keep=8)
    assert np.array_equal(a,b) and np.array_equal(aw,bw)
