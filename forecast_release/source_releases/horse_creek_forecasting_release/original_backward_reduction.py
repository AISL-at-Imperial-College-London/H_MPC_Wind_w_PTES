"""Faithful finite-scenario backward deletion (Algorithm 2, submitted paper).

Distance: Euclidean norm of concatenated standardised wind and price over
ONLY the 84 HL-relevant hours. Selection explicitly includes ALL deleted paths.
Masses are redistributed only after the final 50 representatives are chosen.
This is a greedy Wasserstein-1 heuristic; not a global discrete optimum.
"""
import numpy as np
from scipy.spatial.distance import cdist

def backward_reduce_original(wind_hl, price_hl, keep=50):
    w=np.asarray(wind_hl,dtype='float64');p=np.asarray(price_hl,dtype='float64')
    if w.ndim!=2 or w.shape!=p.shape or not np.isfinite(w).all() or not np.isfinite(p).all():
        raise ValueError('Finite paired [raw scenario, HL hour] matrices required')
    n,h=w.shape
    if n<2 or not 1<=keep<=n:raise ValueError('Invalid keep value')
    # Exactly two physical variables, each standardised prior to concatenation.
    # Standard deviations use ONLY the generated scenarios (no realised future).
    ws=max(float(np.std(w)),1e-10);ps=max(float(np.std(p)),1e-10)
    feat=np.concatenate(((w-np.mean(w))/ws,(p-np.mean(p))/ps),axis=1)
    d=cdist(feat,feat,metric='euclidean')
    np.fill_diagonal(d,np.inf)
    # Original raw samples all have probability 1/n for both selection and mass transfer.
    survivor=np.ones(n,dtype=bool)
    while int(survivor.sum())>keep:
        active=np.flatnonzero(survivor)
        deleted=np.flatnonzero(~survivor)
        between=d[np.ix_(active,active)]
        # Direct cost of deleting each active scenario l: pi_l * nearest survivor excluding l.
        cost=between.min(axis=1)/n
        if len(deleted):
            # Previously deleted originals still have their original probabilities;
            # account for their increased nearest-survivor cost if l is removed.
            to_active=d[np.ix_(deleted,active)]
            chosen=to_active.argmin(axis=1)
            first=to_active[np.arange(len(deleted)),chosen]
            second=np.partition(to_active,1,axis=1)[:,1]
            np.add.at(cost,chosen,(second-first)/n)
        loser=active[int(np.argmin(cost))]
        survivor[loser]=False
    kept=np.flatnonzero(survivor)
    deleted=np.flatnonzero(~survivor)
    prob=np.full(len(kept),1./n)
    if len(deleted):
        nearest=d[np.ix_(deleted,kept)].argmin(axis=1)
        np.add.at(prob,nearest,1./n)
    if not np.isclose(prob.sum(),1.0,atol=1e-12):raise RuntimeError('Mass did not sum to unity')
    return kept, prob
