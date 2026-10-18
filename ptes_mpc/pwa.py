"""Exact graph of five-segment PWA approximations via segment selection binaries."""
from .physics import pwa_breakpoints

def add_mapping(m,p,thermal,mode_binary,mode,prm,prefix):
    xs,ys=pwa_breakpoints(mode,prm)
    seg=[]; sub=[]
    for i in range(len(xs)-1):
        z=m.v(f'{prefix}_{mode}_seg{i}',0,1,integer=True)
        u=m.v(f'{prefix}_{mode}_u{i}',0,float(xs[i+1]-xs[i]))
        m.c({u:1,z:-float(xs[i+1]-xs[i])},hi=0)
        seg.append(z);sub.append(u)
    m.c({**{j:1 for j in seg}, mode_binary:-1},lo=0,hi=0)
    m.c({p:1,**{j:-float(xs[i]) for i,j in enumerate(seg)}, **{j:-1 for j in sub}},lo=0,hi=0)
    m.c({thermal:1,**{j:-float(ys[i]) for i,j in enumerate(seg)},
         **{j:-float((ys[i+1]-ys[i])/(xs[i+1]-xs[i])) for i,j in enumerate(sub)}},lo=0,hi=0)
