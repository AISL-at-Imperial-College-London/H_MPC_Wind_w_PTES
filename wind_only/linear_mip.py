"""Sparse linear MIP builder with SciPy/HiGHS MILP and optional exact Gurobi MIQP."""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import milp, Bounds, LinearConstraint

def _read_gurobi_result(model, x, GRB, *, n, nrows, has_integer_variables, requested_gap):
    """Interpret both continuous LP/QP and mixed-integer Gurobi solutions.

    Gurobi provides MIPGap and branch-and-bound statistics for mixed-integer
    solves, but these are not valid attributes for an ordinary continuous LP.
    An LP is accepted only with certified OPTIMAL termination.
    """
    status = int(model.Status)
    nsol = int(model.SolCount)
    runtime = float(model.Runtime)
    if has_integer_variables:
        gap = float(model.MIPGap) if nsol else None
        certified = bool(nsol) and (
            status == GRB.OPTIMAL or
            (status == GRB.TIME_LIMIT and gap is not None and
             np.isfinite(gap) and gap <= requested_gap + 1e-9)
        )
        node_count = float(model.NodeCount)
    else:
        gap = None  # MIPGap is not defined for a continuous LP.
        certified = bool(nsol) and status == GRB.OPTIMAL
        node_count = 0.0
    message = (f'Gurobi {"MIP" if has_integer_variables else "LP/QP"} '
               f'status={status}; solutions={nsol}; gap={gap}; '
               f'certified_to_requested_gap={certified}')
    if not nsol:
        return SimpleNamespace(x=None, fun=np.inf, success=False,
            status=status, message=message, mip_gap=gap,
            solver_backend='gurobi', runtime_seconds=runtime,
            node_count=node_count, best_bound=None,
            model_nvars=n, model_nrows=nrows)
    obj = float(model.ObjVal)
    # For an optimal LP, the primal objective is the certified bound.
    # Do not query ObjBound for LP models: Gurobi may not expose it.
    best_bound = (float(model.ObjBound) if has_integer_variables else
                  obj if certified else None)
    return SimpleNamespace(x=np.asarray(x.X, dtype=float).reshape(-1),
        fun=obj, success=certified, status=status, message=message,
        mip_gap=gap, solver_backend='gurobi', runtime_seconds=runtime,
        node_count=node_count, best_bound=best_bound,
        model_nvars=n, model_nrows=nrows)


class MIP:
    def __init__(self):
        self.names=[];self.lb=[];self.ub=[];self.integer=[];self.obj=[]
        self.rows=[];self.row_lb=[];self.row_ub=[]
    def v(self,name,lo=0.,hi=np.inf,integer=False,obj=0.):
        n=len(self.names);self.names.append(name);self.lb.append(lo);self.ub.append(hi)
        self.integer.append(int(integer));self.obj.append(obj)
        return n
    def c(self,coefs,lo=-np.inf,hi=np.inf):
        self.rows.append({int(k):float(v) for k,v in coefs.items() if abs(v)>1e-12})
        self.row_lb.append(lo);self.row_ub.append(hi)
    def solve(self,time_limit_s=20.,mip_rel_gap=.02):
        ri=[];ci=[];data=[]
        for r,row in enumerate(self.rows):
            for c,v in row.items():ri.append(r);ci.append(c);data.append(v)
        a=coo_matrix((data,(ri,ci)),shape=(len(self.rows),len(self.names))).tocsr()
        return milp(np.array(self.obj),integrality=np.array(self.integer),
                    bounds=Bounds(self.lb,self.ub),
                    constraints=LinearConstraint(a,self.row_lb,self.row_ub),
                    options={'time_limit':time_limit_s,'mip_rel_gap':mip_rel_gap})
    def solve_gurobi(self, time_limit_s=30., mip_rel_gap=.01,
                     quadratic_terms=(), threads=0, seed=1,
                     log_file=None, gurobi_output=False):
        """Solve the *same* sparse MILP; optional exact convex quadratic terms.

        Build variables and linear constraints using Gurobi's matrix API to avoid
        Python-per-row overhead on the 50-scenario, 84-hour extensive form.
        A non-OPTIMAL time-limit incumbent is accepted for MIPs only if it
        has a finite, certified MIPGap no larger than the requested tolerance.
        Continuous LPs are accepted only after OPTIMAL termination.
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError('gurobipy is not installed in the active Python environment') from exc
        n=len(self.names)
        if n==0:raise ValueError('cannot optimise an empty model')
        if not (0<mip_rel_gap<1):raise ValueError('MIPGap must be in (0,1)')
        from scipy.sparse import coo_matrix
        rid=[];cid=[];values=[]
        for i,coefs in enumerate(self.rows):
            for j,a in coefs.items():
                rid.append(i);cid.append(j);values.append(a)
        A=coo_matrix((values,(rid,cid)),shape=(len(self.rows),n)).tocsr()
        row_lo=np.asarray(self.row_lb,float)
        row_hi=np.asarray(self.row_ub,float)
        equal=np.isfinite(row_lo)&np.isfinite(row_hi)&(abs(row_lo-row_hi)<=1e-9)
        lower=np.isfinite(row_lo)&~equal
        upper=np.isfinite(row_hi)&~equal
        model=gp.Model('HorseCreek_Wind_Only_HL_LL')
        model.Params.OutputFlag=int(bool(gurobi_output))
        model.Params.TimeLimit=float(time_limit_s)
        model.Params.MIPGap=float(mip_rel_gap)
        model.Params.Threads=int(threads)    # 0 = Gurobi automatic selection
        model.Params.Seed=int(seed)
        model.Params.NumericFocus=1         # a mild numerical safeguard
        model.Params.NodefileStart=0.5      # spill nodes after 0.5 GB
        if log_file:
            model.Params.LogFile=str(log_file)
        vtypes=np.asarray([GRB.BINARY if it and abs(self.lb[i])<1e-10 and abs(self.ub[i]-1)<1e-10
                           else GRB.INTEGER if it else GRB.CONTINUOUS
                           for i,it in enumerate(self.integer)],dtype='<U1')
        safe_lb=np.where(np.isfinite(self.lb),np.asarray(self.lb,float),-GRB.INFINITY)
        safe_ub=np.where(np.isfinite(self.ub),np.asarray(self.ub,float),GRB.INFINITY)
        x=model.addMVar(n,lb=safe_lb,ub=safe_ub,
                        vtype=vtypes,name='decision')
        if equal.any():model.addMConstr(A[equal],x,'=',row_lo[equal])
        if lower.any():model.addMConstr(A[lower],x,'>',row_lo[lower])
        if upper.any():model.addMConstr(A[upper],x,'<',row_hi[upper])
        obj=np.asarray(self.obj,float)@x
        for j,target,beta in quadratic_terms:
            if beta<0:raise ValueError('nonconvex negative quadratic weight')
            # The quadratic target is exact; NO epigraph approximation.
            obj=obj+float(beta)*(x[int(j)]-float(target))*(x[int(j)]-float(target))
        model.setObjective(obj,GRB.MINIMIZE)
        model.optimize()
        return _read_gurobi_result(
            model, x, GRB, n=n, nrows=A.shape[0],
            has_integer_variables=any(self.integer),
            requested_gap=mip_rel_gap)

    def solve_exact_miqp(self,quadratic_terms,time_limit_s=20.,mip_rel_gap=.02):
        """Exact LL MIQP via the common, sparse, Gurobi solver path."""
        return self.solve_gurobi(time_limit_s,mip_rel_gap,quadratic_terms)

