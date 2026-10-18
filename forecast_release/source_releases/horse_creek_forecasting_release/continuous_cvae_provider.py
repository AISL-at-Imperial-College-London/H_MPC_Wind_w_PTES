"""Continuous calibrated joint wind/DA price CVAE v3.

Never reads 2026 holdout outcomes for model training, sampling or calibration.
Every raw trajectory starts immediately after the last completed input hour.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from continuous_cvae import checkpoint_load, make_raw, calendar_features, invert

class ContinuousCalibratedCVAE:
    def __init__(self,checkpoint,calibration=None):
        torch.set_num_threads(min(4,torch.get_num_threads()))
        self.model,self.cp=checkpoint_load(checkpoint)
        self.scaling=self.cp['scaling']
        if self.cp.get('gap_h')!=0 or self.cp.get('horizon_h')!=120:
            raise ValueError('Checkpoint must represent continuous 120-hour forecast')
        self.calibration=Path(calibration) if calibration is not None else None
        if self.calibration is None:self.resid=None
        else:
            with np.load(self.calibration,allow_pickle=False) as dat:
                res=dat['residuals'].astype('float32')
            if res.ndim!=3 or res.shape[1:]!=(120,2) or len(res)<50 or not np.isfinite(res).all():
                raise ValueError('Invalid paired 120-hour calibration trajectories')
            # Existing v2 calibration policy: centre wind residual mean only,
            # retain paired residual serial and price dependence.
            res[:,:,0]-=res[:,:,0].mean(axis=0,keepdims=True)
            self.resid=res

    def generate_raw(self,history,first_unobserved,*,n=500,seed=3714):
        h=np.asarray(history,dtype=float)
        if h.shape!=(96,2) or not np.isfinite(h).all():
            raise ValueError('96 completed hourly 2-channel observations required')
        first=pd.Timestamp(first_unobserved)
        if first.tzinfo is None:raise ValueError('UTC aware first hour required')
        cal=calendar_features([first])[0]
        base=make_raw(self.model,self.scaling,h,cal,n=n,seed=seed).astype('float64')
        if self.resid is not None:
            rng=np.random.default_rng(seed+8731)
            ids=rng.integers(0,len(self.resid),size=n)
            center=base.mean(axis=0)
            base=center[None,:,:]+self.resid[ids]+.15*(base-center)
        out=invert(base,self.scaling)
        out[...,1]=np.clip(out[...,1],-1000,10000)
        return out[...,0],out[...,1]
