"""Reproducible fast training of continuous probabilistic CVAE v3.
Saved model selection is a Jan-Feb 2025 validation NLL; do not select using protected 2026 holdouts.
"""
from pathlib import Path
import sys,json,argparse,time,random
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from continuous_cvae import *

def train(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    df=pd.read_csv(args.input);df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
    df=df.set_index('timestamp_utc').sort_index()
    trainmask=df.index.year.isin([2018,2019,2020,2021,2024])
    sc=fit_scaling(df.loc[trainmask,['HorseCreek_MW','DA_price_usd_mwh']].dropna().to_numpy())
    tr=assemble_arrays(df,sc,[2018,2019,2020,2021,2024],stride=args.stride)
    va=assemble_arrays(df,sc,[2025],stride=12)
    # Evaluate only first two months; after that keep independent for calibration.
    va_idxs=pd.DatetimeIndex(df.index[va[3]+GAP_H]); mask=(va_idxs+pd.Timedelta(hours=HORIZON)<=pd.Timestamp('2025-03-01',tz='UTC'))
    va=tuple(np.asarray(a)[mask] for a in va)
    def dl(arr,bs,shuffle=False):
        return DataLoader(TensorDataset(*[torch.tensor(x,dtype=torch.float32) for x in arr[:3]]),batch_size=bs,shuffle=shuffle,num_workers=0)
    dtrain=dl(tr,args.batch,True);dval=dl(va,args.batch*2)
    model=ContinuousCVAE(args.hidden,args.latent);optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=1e10; bestep=None;rows=[];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    for ep in range(1,args.epochs+1):
        model.train();scores=[];beta=args.beta*min(1,ep/args.kl_warmup)
        for h,c,y in dtrain:
            optim.zero_grad(set_to_none=True);loss,nll,kl=model(h,c,y,beta)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);optim.step()
            scores.append([loss.item(),nll.item(),kl.item()])
        model.eval();vals=[]
        with torch.no_grad():
            for h,c,y in dval:
                torch.manual_seed(100)
                l,nll,kl=model(h,c,y,beta)
                vals.append([l.item(),nll.item(),kl.item()])
        stat={'epoch':ep,'train_loss':float(np.mean(scores,axis=0)[0]),'val_loss':float(np.mean(vals,axis=0)[0]),
              'val_nll':float(np.mean(vals,axis=0)[1]),'train_kl':float(np.mean(scores,axis=0)[2]),'beta':beta}
        rows.append(stat);score=stat['val_loss']
        if score<best:
            best=score;bestep=ep
            torch.save({'state_dict':model.state_dict(),'hidden':args.hidden,'latent':args.latent,
                        'scaling':sc,'best_epoch':ep,'val_loss':best,'history_h':96,
                        'gap_h':0,'horizon_h':120,'train_years':'2018-2021,2024',
                        'validation':'2025 Jan-Feb','random_seed':args.seed},out/'continuous_cvae_v3_checkpoint.pt')
        if ep%5==0 or ep==1:
            print('epoch',ep,'train',round(stat['train_loss'],3),'val',round(score,3),
                  'best',bestep,'elapsed',round(time.time()-t0,1),flush=True)
        if bestep is not None and ep-bestep>=args.patience:break
    pd.DataFrame(rows).to_csv(out/'training_history.csv',index=False)
    (out/'model_manifest.json').write_text(json.dumps({'training_samples':len(tr[0]),'earlystop_val_samples':len(va[0]),
                 'selected_epoch':bestep,'price_scaling':sc,'history':96,'gap_h':0,'horizon':120,
                 'training_years':'2018-2021,2024','earlystop':'2025 Jan-Feb',
                 'calibration_reserved':'2025 Mar-Jun','development_test_reserved':'2025 Jul-Dec',
                 'protected_holdouts':'2026 Feb,May,Jul','runtime_seconds':time.time()-t0},indent=2))
    print('Done',out,'bestep',bestep,'runtime',round(time.time()-t0),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--out',required=True)
    a.add_argument('--seed',type=int,default=2600);a.add_argument('--epochs',type=int,default=80)
    a.add_argument('--batch',type=int,default=256);a.add_argument('--stride',type=int,default=3)
    a.add_argument('--hidden',type=int,default=128);a.add_argument('--latent',type=int,default=16)
    a.add_argument('--beta',type=float,default=.05);a.add_argument('--kl-warmup',type=int,default=12)
    a.add_argument('--patience',type=int,default=12);a.add_argument('--threads',type=int,default=4)
    a.add_argument('--lr',type=float,default=0.0008)
    train(a.parse_args())
