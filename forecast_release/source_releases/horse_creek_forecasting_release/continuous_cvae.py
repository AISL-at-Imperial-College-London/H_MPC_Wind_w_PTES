"""Horse Creek continuous conditional VAE (CVAE v3).

Inputs: last 96 completed hours at DA bid issue time, calendar at delivery.
Output: 120 hourly joint wind and DA-price values STARTING at the first
        unobserved hour after the latest fully completed hourly interval.
Training: 2018-2021 + 2024. Early stopping: Jan-Feb 2025.
Calibration must use separate Mar-Jun 2025. NEVER fit 2026 observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

HISTORY=96
GAP_H=0
HORIZON=120
RATING=230.0
FEATURES=10

def calendar_features(timestamps):
    local=pd.DatetimeIndex(timestamps).tz_convert('America/Chicago')
    doy=np.asarray(local.dayofyear,dtype=float)
    hour=np.asarray(local.hour,dtype=float)
    dow=np.asarray(local.dayofweek,dtype=float)
    year=np.asarray(local.year,dtype=float)
    return np.stack([np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25),
                     np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                     np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),
                     np.asarray(local.month,dtype=float)/12,
                     np.asarray(local.day,dtype=float)/31,
                     np.asarray(local.weekday<5,dtype=float),
                     np.ones(len(local))],axis=-1).astype('float32')

def fit_scaling(x):
    z=x.astype('float64').copy();z[:,1]=np.arcsinh(z[:,1]/40)
    return {'mean':z.mean(axis=0).tolist(),'std':z.std(axis=0).tolist(),
            'price_transform':'asinh_price_div_40','wind_rating_mw':230,
            'training_years':'2018-2021,2024'}

def transform(x,sc):
    a=np.asarray(x,dtype=np.float64).copy()
    a[...,1]=np.arcsinh(a[...,1]/40)
    return ((a-np.asarray(sc['mean']))/np.asarray(sc['std'])).astype('float32')

def invert(x,sc):
    a=np.asarray(x,dtype=np.float64)*np.asarray(sc['std'])+np.asarray(sc['mean'])
    a[...,1]=40*np.sinh(np.clip(a[...,1],-12,12))
    a[...,0]=np.clip(a[...,0],0,RATING)
    return a

def invert_price_transformed(x,sc):
    a=np.asarray(x,dtype=np.float64)*np.asarray(sc['std'])+np.asarray(sc['mean'])
    a[...,0]=np.clip(a[...,0],0,RATING)
    return a

class ContinuousCVAE(nn.Module):
    def __init__(self,hidden=128,latent=16):
        super().__init__();self.hidden=hidden;self.latent=latent
        self.cond=nn.Sequential(nn.Linear(HISTORY*2+FEATURES,hidden),nn.LayerNorm(hidden),nn.GELU(),
                                nn.Linear(hidden,hidden),nn.GELU())
        self.prior=nn.Linear(hidden,2*latent)
        self.post=nn.Sequential(nn.Linear(hidden+HORIZON*2,hidden),nn.GELU(),nn.Linear(hidden,2*latent))
        self.decode=nn.Sequential(nn.Linear(hidden+latent,hidden*2),nn.GELU(),
                                  nn.Linear(hidden*2,hidden*2),nn.GELU(),nn.Linear(hidden*2,HORIZON*2))
        self.noise=nn.Linear(hidden,HORIZON*2)
    def context(self,h,calendar):return self.cond(torch.cat((h.flatten(1),calendar),dim=1))
    def posterior(self,ctx,y):return self.post(torch.cat((ctx,y.flatten(1)),dim=1)).chunk(2,dim=-1)
    def prior_params(self,ctx):return self.prior(ctx).chunk(2,dim=-1)
    def decoder(self,ctx,z):
        base=self.decode(torch.cat((ctx,z),dim=-1)).reshape(-1,HORIZON,2)
        logscale=self.noise(ctx).reshape(-1,HORIZON,2).clamp(-3.5,1.0)
        return base,logscale
    def forward(self,h,cal,y,beta=.08):
        ctx=self.context(h,cal)
        pm,pl=self.prior_params(ctx);qm,ql=self.posterior(ctx,y)
        pl=pl.clamp(-6,3);ql=ql.clamp(-6,3)
        z=qm+torch.randn_like(qm)*torch.exp(.5*ql)
        mu,logscale=self.decoder(ctx,z)
        # Conditional likelihood rather than deterministic MSE. Channel scales
        # are transformed on TRAINING data only, preventing price spikes domination.
        nll=(.5*((y-mu)*torch.exp(-logscale))**2+logscale).mean()
        kl=(.5*(pl-ql+(ql.exp()+(qm-pm)**2)/pl.exp()-1)).sum(dim=-1).mean()
        loss=nll+beta*kl/self.latent
        return loss,nll.detach(),kl.detach()
    @torch.no_grad()
    def sample(self,h,calendar,n=500,seed=3714):
        self.eval();ctx=self.context(h,calendar)
        if ctx.shape[0]!=1:raise ValueError('single issue time only')
        pm,pl=self.prior_params(ctx)
        gen=torch.Generator(device=h.device).manual_seed(int(seed))
        z=pm+torch.randn((n,self.latent),generator=gen,device=h.device)*torch.exp(.5*pl.clamp(-6,3))
        mu,logsigma=self.decoder(ctx.expand(n,-1),z)
        # Smooth, correlated short-time innovations plus independent component.
        noise=torch.randn((n,HORIZON,2),generator=gen,device=h.device)
        noise_sm=torch.nn.functional.avg_pool1d(noise.transpose(1,2),7,1,3).transpose(1,2)*np.sqrt(7)
        innovation=.85*noise_sm+.5268*noise
        return (mu+innovation*torch.exp(logsigma)).cpu().numpy()

def assemble_arrays(df, sc, years, *, stride=3):
    """Only complete 96h history and continuous 120h target, same year split."""
    ts=pd.DatetimeIndex(df.index).as_unit('ns'); vals=df[['HorseCreek_MW','DA_price_usd_mwh']].to_numpy(float)
    ok=np.isfinite(vals).all(axis=1)
    hourdiff=np.diff(ts.asi8)==3600*10**9
    # O(1) validity for complete ranges and contiguous calendar
    bad=np.concatenate(([0],np.cumsum((~hourdiff).astype('int32'))))
    invalid=np.concatenate(([0],np.cumsum((~ok).astype('int32'))))
    origins=[]
    for i in range(HISTORY,len(vals)-GAP_H-HORIZON+1,stride):
        a=i-HISTORY;b=i+GAP_H+HORIZON
        if ts[i].year not in years or ts[a].year not in years or ts[b-1].year not in years:continue
        if invalid[b]-invalid[a] or bad[b-1]-bad[a]:continue
        origins.append(i)
    idx=np.asarray(origins,dtype=int)
    if not len(idx):raise ValueError('No windows')
    scaled=transform(vals,sc)
    past=np.stack([scaled[i-HISTORY:i] for i in idx])
    fut=np.stack([scaled[i+GAP_H:i+GAP_H+HORIZON] for i in idx])
    targetts=ts[idx+GAP_H]
    c=calendar_features(targetts)
    return past,c,fut,idx

def checkpoint_load(path):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    m=ContinuousCVAE(hidden=cp['hidden'],latent=cp['latent'])
    m.load_state_dict(cp['state_dict'],strict=True);m.eval()
    return m,cp

@torch.no_grad()
def make_raw(model,sc,past,cal,*,seed=3714,n=500):
    h=torch.from_numpy(transform(past,sc)[None,:,:]);c=torch.from_numpy(np.asarray(cal,dtype='float32').reshape(1,-1))
    return model.sample(h,c,n=n,seed=seed)
