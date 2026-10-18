#!/usr/bin/env python3
"""Thesis-style temporal CVAE for joint Horse Creek hourly wind and ERCOT DA price.

- Loads hourly wind + DA price directly from Horse Creek training ZIP, without extracting.
- Rejects 2026 scored holdouts; uses strict contiguous hourly windows.
- 96h historical + 96h future, ALL eligible hourly origins for training.
- CNN/LSTM historical + future encoders; 4D VAE latent; calendar embeddings;
  LSTM + transposed convolution decoder, matching Simon Gatt MSc thesis Section 3.2.4.
- Time-weighted MSE plus beta * KL; beta=.001 first 30 epochs, ramps to .35
  at epoch 180, remains through 250. Adam with ReduceLROnPlateau.

This is a fresh model trained on Horse Creek data, NOT the frozen probabilistic
AR low-rank model used in the earlier MPC simulations. It requires independent
validation before MPC integration. No 2026 deployment file is read.
"""
from __future__ import annotations
import argparse, csv, io, json, os, random, sys, time, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HISTORY=96
FUTURE=96
RATING=230.0
TZ='America/Chicago'


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_zip_hourly(zip_path):
    """Select only joint-hourly development files, no holdout/5-min/15-min data."""
    source=Path(zip_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f'ZIP not found: {source}. Edit ZIP_PATH in RUN_IN_SPYDER.py')
    frames=[]; used=[]
    with zipfile.ZipFile(source) as z:
        allfiles=z.namelist()
        # Priority: a prepared integrated development master with 2018-21, 2024, 2025.
        masters=[n for n in allfiles if n.lower().endswith('.csv') and
                 'development_master_hourly' in Path(n).name.lower() and
                 'deployment' not in n.lower() and 'holdout' not in n.lower()]
        if masters:
            candidates=[min(masters,key=len)]
        else:
            candidates=[n for n in allfiles if n.lower().endswith('.csv') and
                        'tcvae_hourly_wind_da' in Path(n).name.lower() and
                        'deployment' not in n.lower() and 'holdout' not in n.lower()]
        if not candidates:
            raise ValueError('No development_master_hourly or TCVAE_hourly_wind_DA CSV found in ZIP. '
                             'Provide the Horse Creek forecasting or training bundle.')
        for name in candidates:
            df=pd.read_csv(io.BytesIO(z.read(name)))
            needed={'timestamp_utc','HorseCreek_MW','DA_price_usd_mwh'}
            if not needed.issubset(df.columns):
                raise ValueError(f'{name} is missing columns {needed-set(df.columns)}')
            dt=pd.to_datetime(df['timestamp_utc'],utc=True,errors='coerce')
            frame=pd.DataFrame({'timestamp_utc':dt,
                                'HorseCreek_MW':pd.to_numeric(df['HorseCreek_MW'],errors='coerce'),
                                'DA_price_usd_mwh':pd.to_numeric(df['DA_price_usd_mwh'],errors='coerce')})
            if 'model_usable' in df.columns:
                good=df['model_usable'].astype(str).str.lower().isin(('true','1','yes'))
                frame=frame.loc[good]
            frame=frame.dropna(subset=['timestamp_utc','HorseCreek_MW','DA_price_usd_mwh'])
            # EXPLICIT holdout protection: no 2026 data, even if developer ZIP includes it.
            frame=frame[frame.timestamp_utc.dt.year.between(2018,2025)]
            frame=frame[frame.HorseCreek_MW.between(0,RATING)]
            frames.append(frame); used.append(name)
    df=pd.concat(frames,ignore_index=True).drop_duplicates('timestamp_utc',keep='first')
    df=df.sort_values('timestamp_utc').reset_index(drop=True)
    if df.empty: raise ValueError('No eligible hourly development records')
    # Check clock positions and reject data not on exact hourly UTC boundaries.
    if not ((df.timestamp_utc.dt.minute==0)&(df.timestamp_utc.dt.second==0)).all():
        raise ValueError('The selected dataset is not aligned to UTC hourly starts')
    print('Joint development files loaded:')
    for path in used: print('  ',path)
    print('Hourly samples by year:',df.timestamp_utc.dt.year.value_counts().sort_index().to_dict())
    return df, used


def prepare_splits(df):
    """Make nonoverlapping year/period splits, then windows strictly INSIDE each split."""
    years=set(df.timestamp_utc.dt.year.unique().tolist())
    if not ({2018,2019,2020,2021}&years):
        raise ValueError('No training years 2018-2021 found')
    # Keep training data pre-2022, independent later-year validation and test.
    if 2024 in years:
        policy='train 2018-2021; validate 2024; independent development test 2025'
        def split(t):
            if t.year<=2021:return 'train'
            if t.year==2024:return 'validation'
            if t.year==2025:return 'test'
            return 'unused'
    elif 2025 in years:
        policy='train 2018-2021; validate 2025 Jan-Jun; development test 2025 Jul-Dec (no 2024 in ZIP)'
        def split(t):
            if t.year<=2021:return 'train'
            if t.year==2025:return 'validation' if t.month<=6 else 'test'
            return 'unused'
    else:
        raise ValueError('No 2024 or 2025 validation data. Use a training bundle including at least one.')
    df=df.copy()
    df['split']=[split(t) for t in df.timestamp_utc]
    df=df[df.split!='unused'].reset_index(drop=True)
    return df,policy


def eligible_starts(df):
    """Return every hourly start with 96 complete preceding and following hourly observations.

    The full 192-hour window must share the same split, so that train/val/test
    cannot leak into one another and no missing calendar periods are bridged.
    """
    # IMPORTANT (pandas 2.x/3.x compatibility): pandas 3 may store parsed
    # timestamps at microsecond resolution. `.array.asi8` returns integers
    # in the *array's own* unit, not necessarily nanoseconds. The original
    # implementation compared these against 3600 * 10**9, which silently
    # rejected every valid hourly window under pandas 3. Normalize to ns.
    times=pd.DatetimeIndex(df.timestamp_utc).as_unit('ns').asi8
    grp=df.split.to_numpy()
    starts=[]
    n=HISTORY+FUTURE
    for i in range(HISTORY, len(df)-FUTURE+1):
        first=i-HISTORY
        last=i+FUTURE-1
        if grp[first]!=grp[last]:continue
        if times[last]-times[first] != (n-1)*3600*10**9:continue
        starts.append(i)
    starts=np.array(starts,dtype=np.int64)
    if len(starts)==0: raise ValueError('No full 96h/96h windows; inspect gaps and your ZIP')
    counts={s:int(np.sum(grp[starts]==s)) for s in ('train','validation','test')}
    print('Overlapping HOURLY origin counts:',counts)
    if counts['train']==0 or counts['validation']==0:
        raise ValueError('Missing train or validation 192-hour windows')
    return starts,counts


def fit_transform(values, training_mask, price_transform='none'):
    x=values.astype(np.float64).copy()
    # `none` reproduces thesis' direct standardization. Optional asinh for 2021 tails.
    if price_transform=='asinh': x[:,1]=np.arcsinh(x[:,1]/50.0)
    mu=x[training_mask].mean(0)
    sigma=x[training_mask].std(0)
    if (sigma<=0).any(): raise ValueError('One channel has zero training standard deviation')
    return ((x-mu)/sigma).astype(np.float32),{'mean':mu.tolist(),'std':sigma.tolist(),
            'price_transform':price_transform,'fitted_on':'2018-2021 training years only'}


def inverse_transform(arr, scaling):
    x=np.asarray(arr,dtype=np.float64)*np.array(scaling['std'])+np.array(scaling['mean'])
    if scaling['price_transform']=='asinh': x[...,1]=50.0*np.sinh(np.clip(x[...,1],-9,9))
    x[...,0]=np.clip(x[...,0],0,RATING)
    return x


class WindowDataset(Dataset):
    def __init__(self, normalized, timestamps, starts):
        self.data=torch.from_numpy(normalized)
        self.starts=np.asarray(starts,dtype=np.int64)
        # the weekday/time refer to the first target hour in America/Chicago
        starts_local=pd.DatetimeIndex(timestamps[self.starts]).tz_convert(TZ)
        self.dow=torch.as_tensor(starts_local.dayofweek.to_numpy(),dtype=torch.long)
        self.hour=torch.as_tensor(starts_local.hour.to_numpy(),dtype=torch.long)
    def __len__(self):return len(self.starts)
    def __getitem__(self,idx):
        i=int(self.starts[idx]); return (self.data[i-HISTORY:i],self.data[i:i+FUTURE],
                                           self.dow[idx],self.hour[idx])


class TemporalEncoder(nn.Module):
    def __init__(self,hidden=36,ctxdim=4):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv1d(2,hidden,5,padding=2),nn.GELU(),
                               nn.MaxPool1d(2),nn.Conv1d(hidden,hidden,3,padding=1),nn.GELU())
        self.rnn=nn.LSTM(hidden,hidden,batch_first=True)
        self.fc=nn.Linear(hidden,ctxdim)
    def forward(self,x):
        h=self.cnn(x.transpose(1,2)).transpose(1,2)
        _,(state,_) = self.rnn(h)
        return self.fc(state[-1])


class ThesisTCVAE(nn.Module):
    def __init__(self,hidden=36,latent_dim=4):
        super().__init__()
        self.latent_dim=latent_dim
        self.history_enc=TemporalEncoder(hidden,ctxdim=4)
        self.future_enc=TemporalEncoder(hidden,ctxdim=hidden)
        self.dow_emb=nn.Embedding(7,8)
        self.hour_emb=nn.Embedding(24,24)
        cond_dim=4+8+24
        self.posterior=nn.Sequential(nn.Linear(hidden+cond_dim,64),nn.GELU())
        self.mu=nn.Linear(64,latent_dim)
        self.logvar=nn.Linear(64,latent_dim)
        self.to_decoder=nn.Sequential(nn.Linear(latent_dim+cond_dim,48*hidden),nn.GELU())
        self.decoder_rnn=nn.LSTM(hidden,hidden,batch_first=True)
        self.upsample=nn.ConvTranspose1d(hidden,hidden,4,stride=2,padding=1)
        self.out_conv=nn.Conv1d(hidden,2,3,padding=1)
    def context(self,past,dow,hour):
        return torch.cat([self.history_enc(past),self.dow_emb(dow),self.hour_emb(hour)],dim=-1)
    def decode(self,z,ctx):
        h=self.to_decoder(torch.cat([z,ctx],dim=1)).reshape(-1,48,self.decoder_rnn.hidden_size)
        h,_=self.decoder_rnn(h)
        h=self.upsample(h.transpose(1,2))
        return self.out_conv(torch.relu(h)).transpose(1,2)
    def forward(self,past,future,dow,hour):
        ctx=self.context(past,dow,hour)
        enc=self.posterior(torch.cat([self.future_enc(future),ctx],dim=1))
        mu=self.mu(enc)
        logvar=self.logvar(enc).clamp(-12,8)
        z=mu+torch.exp(logvar/2)*torch.randn_like(mu)
        return self.decode(z,ctx),mu,logvar
    @torch.no_grad()
    def sample(self,past,dow,hour,n=50,seed=2026):
        self.eval()
        ctx=self.context(past,dow,hour)
        if past.shape[0]!=1: raise ValueError('Sample supports one conditioning history at a time')
        gen=torch.Generator(device=past.device).manual_seed(seed)
        z=torch.randn(n,self.latent_dim,device=past.device,generator=gen)
        return self.decode(z,ctx.expand(n,-1)).cpu().numpy()


def beta_at(epoch):
    if epoch<=30:return .001
    return .001+(.35-.001)*min(1.0,(epoch-30)/(180-30))


def train_loss(recon,target,mu,logvar,beta):
    # Thesis equation 3.61: SUM over 96 horizon steps AND both channels;
    # do not replace by mean over T/C as that changes the relative KL weight.
    w=torch.linspace(1,0.2,FUTURE,device=target.device)
    mse=(((recon-target)**2)*w[None,:,None]).sum(dim=(1,2)).mean()
    kl=(-0.5*(1+logvar-mu.square()-logvar.exp()).sum(dim=1)).mean()
    return mse+beta*kl, mse.detach(), kl.detach()


def evaluate(model,loader,device,seed):
    model.eval()
    losses=np.zeros(3); count=0
    torch.manual_seed(seed)
    with torch.no_grad():
        for a,b,c,d in loader:
            a,b,c,d=(v.to(device) for v in (a,b,c,d))
            rec,mu,logvar=model(a,b,c,d)
            loss,mse,kl=train_loss(rec,b,mu,logvar,0)
            size=len(a); losses+=np.array([mse.item(),kl.item(),0])*size;count+=size
    return losses[:2]/count


def train(args):
    t0=time.time(); seed_everything(args.seed)
    torch.set_num_threads(max(1,min(args.threads,os.cpu_count() or 1)))
    df,used=load_zip_hourly(args.zip)
    df,policy=prepare_splits(df)
    starts,counts=eligible_starts(df)
    values=df[['HorseCreek_MW','DA_price_usd_mwh']].to_numpy(np.float32)
    training_rows=(df.split.to_numpy()=='train')
    norm,scaling=fit_transform(values,training_rows,args.price_transform)
    split=df.split.to_numpy()
    dsets={name:WindowDataset(norm,df.timestamp_utc,starts[split[starts]==name])
           for name in ['train','validation','test']}
    if args.max_examples is not None:
        # Development-only diagnostic option, not for publication-quality training.
        rng=np.random.default_rng(args.seed)
        for name in dsets:
            ds=dsets[name]
            if len(ds)>args.max_examples:
                ds.starts=ds.starts[np.sort(rng.choice(len(ds),args.max_examples,replace=False))]
                # Reconstruct matching weekday/hour labels for subsampled indices.
                starts_local=pd.DatetimeIndex(df.timestamp_utc.iloc[ds.starts]).tz_convert(TZ)
                ds.dow=torch.as_tensor(starts_local.dayofweek.to_numpy(),dtype=torch.long)
                ds.hour=torch.as_tensor(starts_local.hour.to_numpy(),dtype=torch.long)
    train_loader=DataLoader(dsets['train'],batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(dsets['validation'],batch_size=args.batch*2,shuffle=False,num_workers=0)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    print('Device:',device, 'parameters:',sum(p.numel() for p in ThesisTCVAE().parameters()))
    model=ThesisTCVAE(hidden=args.hidden,latent_dim=4).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5)
    out=Path(args.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    metadata={'source_zip':str(Path(args.zip).expanduser()),'loaded_files':used,'split_policy':policy,
              'n_eligible_windows':counts,'n_used_windows':{k:len(ds) for k,ds in dsets.items()},
              'architecture':'Thesis T-CVAE: CNN LSTM, 96-96, 4D context/latent, calendar embeddings',
              'epochs_requested':args.epochs,'batch':args.batch,'seed':args.seed,
              'loss':'sum over future time and channels of wt*(prediction-target)^2 + beta*KL',
              'beta':'0.001 epochs 1-30; linear ramp to 0.35 at epoch 180',
              'scaling':scaling,'no_2026_training_or_validation':True,
              'warning':'Training is NOT the earlier frozen probabilistic AR T-CVAE. Retraining requires MPC forecasts to be regenerated.'}
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2))
    stats=[];best=float('inf');best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train();sums=np.zeros(3);count=0;beta=beta_at(epoch)
        for a,b,c,d in train_loader:
            a,b,c,d=(v.to(device) for v in (a,b,c,d))
            opt.zero_grad(set_to_none=True)
            rec,mu,logvar=model(a,b,c,d)
            loss,mse,kl=train_loss(rec,b,mu,logvar,beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            size=len(a);count+=size
            sums+=np.array([float(loss.detach()),float(mse),float(kl)])*size
        tr=sums/count
        val_mse,val_kl=evaluate(model,val_loader,device,args.seed+epoch)
        # Thesis scheduler observes validation loss; this score includes active beta.
        val_total=val_mse+beta*val_kl
        scheduler.step(val_total)
        rec={'epoch':epoch,'beta':beta,'lr':opt.param_groups[0]['lr'],
             'train_loss':float(tr[0]),'train_mse':float(tr[1]),'train_kl':float(tr[2]),
             'val_loss':float(val_total),'val_mse':float(val_mse),'val_kl':float(val_kl)}
        stats.append(rec)
        pd.DataFrame(stats).to_csv(out/'training_history.csv',index=False)
        if val_total<best:
            best=float(val_total);best_epoch=epoch
            ck={'model':model.state_dict(),'hidden':args.hidden,'latent_dim':4,
                'scaling':scaling,'best_epoch':epoch,'score':best,'seed':args.seed,
                'history_h':HISTORY,'future_h':FUTURE, 'run_config':metadata}
            torch.save(ck,out/'best_checkpoint.pt')
        # Persist latest even when validation gets worse, to keep full 250 epoch outcome.
        if epoch==args.epochs or epoch%5==0:
            torch.save({'model':model.state_dict(),'hidden':args.hidden,'latent_dim':4,
                       'scaling':scaling,'epoch':epoch,'seed':args.seed,
                       'history_h':HISTORY,'future_h':FUTURE,'run_config':metadata},
                       out/'last_checkpoint.pt')
        print(f"epoch {epoch:3d}/{args.epochs} beta={beta:.3f} trainMSE={tr[1]:.3f} "
              f"valMSE={val_mse:.3f} valKL={val_kl:.3f} lr={opt.param_groups[0]['lr']:.2g} "
              f"best={best_epoch} runtime={time.time()-t0:.0f}s",flush=True)
    print('Training completed, full checkpoint:',out/'last_checkpoint.pt')
    print('Best validation checkpoint:',out/'best_checkpoint.pt')
    print('Check conditioning response using check_conditioning.py before MPC use.')
    return out


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip',required=True,help='Horse Creek training ZIP, Windows path permitted')
    ap.add_argument('--output',default='TCVAE_TRAINING_RESULTS')
    ap.add_argument('--epochs',type=int,default=250)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--hidden',type=int,default=36)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20261017)
    ap.add_argument('--price-transform',choices=['none','asinh'],default='none',
                    help='none = direct standardization as in thesis; asinh optional sensitivity')
    ap.add_argument('--max-examples',type=int,default=None,
                    help='SMOKE TEST ONLY: cap training/validation sample counts')
    return ap.parse_args()

if __name__=='__main__':train(parse_args())
