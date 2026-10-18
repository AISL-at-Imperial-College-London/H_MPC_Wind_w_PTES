"""Sparse linear MIP builder with SciPy/HiGHS MILP and optional exact Gurobi MIQP."""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import milp, Bounds, LinearConstraint

def assess_gurobi_incumbent(*, status, solution_count, gap, requested_gap,
                            maximum_time_limit_gap=None, max_violation=0.0):
    """Return (accepted, certified-to-target, accepted-under-relaxed-cap).

    Gurobi GRB.OPTIMAL=2, GRB.TIME_LIMIT=9. The optional cap applies only
    when the low-level caller passes it, never implicitly to the HL model.
    """
    import math
    has_solution = int(solution_count) > 0
    finite_gap = gap is not None and math.isfinite(float(gap))
    certified = bool(has_solution and (int(status) == 2 or
                 (int(status) == 9 and finite_gap and float(gap) <= float(requested_gap)+1e-9)))
    policy_allowed = (maximum_time_limit_gap is not None and
                      float(requested_gap) <= float(maximum_time_limit_gap) < 1.)
    relaxed = bool(not certified and policy_allowed and int(status) == 9 and
                   has_solution and finite_gap and
                   float(gap) <= float(maximum_time_limit_gap)+1e-9 and
                   math.isfinite(float(max_violation)) and float(max_violation) <= 1e-4)
    return certified or relaxed, certified, relaxed

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
                     log_file=None, gurobi_output=False,
                     maximum_time_limit_gap=None, mip_start=None):
        """Solve the *same* sparse MILP; optional exact convex quadratic terms.

        Build variables and linear constraints using Gurobi's matrix API to avoid
        Python-per-row overhead on the 50-scenario, 84-hour extensive form.
        A time-limited LL incumbent above the requested MIPGap may be used
        only under a caller-supplied bound and a small primal violation check.
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
        model=gp.Model('HorseCreek_PTES_MPC')
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
        # Partial Gurobi MIP start: initialize selected integer values, leaving
        # all other variables as GRB.UNDEFINED (not zero!). This is an initial
        # suggestion, never a bound or constraint. The solver repairs/discards
        # a candidate that no longer fits the updated plant/forecast.
        start_count=0
        if mip_start:
            candidate=np.full(n,GRB.UNDEFINED,dtype=float)
            for j,value in mip_start.items():
                j=int(j);value=float(value)
                if not 0<=j<n or not self.integer[j]:
                    raise ValueError('MIP start may initialize only existing integer variables')
                if not np.isfinite(value) or abs(value-round(value))>1e-6:
                    raise ValueError('MIP start must specify finite integral values')
                if value<self.lb[j]-1e-6 or value>self.ub[j]+1e-6:
                    raise ValueError('MIP start violates variable bound')
                candidate[j]=value
                start_count+=1
            if start_count:
                model.update()
                x.Start=candidate
        model.optimize()
        status=int(model.Status)
        nsol=int(model.SolCount)
        gap=(float(model.MIPGap) if nsol else None)
        # Solver status 9 is a time limit, not necessarily an infeasible model.
        # A time-limited incumbent is accepted ONLY when the caller explicitly
        # permits a bounded exception to the target MIPGap.
        max_vio = (max(float(model.ConstrVio),float(model.BoundVio),
                       float(model.IntVio)) if nsol else float('inf'))
        accepted, certified, relaxed = assess_gurobi_incumbent(
            status=status,solution_count=nsol,gap=gap,requested_gap=mip_rel_gap,
            maximum_time_limit_gap=maximum_time_limit_gap,max_violation=max_vio)
        message=f'Gurobi status={status}; solutions={nsol}; gap={gap}; '
        message+=(f'certified_to_requested_gap={certified}; '
                  f'accepted_time_limit_incumbent={relaxed}; '
                  f'max_constraint_or_integrality_violation={max_vio:.3g}')
        from types import SimpleNamespace
        if not nsol:
            return SimpleNamespace(x=None,fun=np.inf,success=False,status=status,
                 message=message,mip_gap=gap,solver_backend='gurobi',
                 runtime_seconds=float(model.Runtime),node_count=float(model.NodeCount),
                 best_bound=None,model_nvars=n,model_nrows=A.shape[0],
                 certified_to_requested_gap=False,accepted_time_limit_incumbent=False,
                 max_solution_violation=max_vio,
                 mip_start_variables=start_count)
        return SimpleNamespace(x=np.asarray(x.X,dtype=float).reshape(-1),
             fun=float(model.ObjVal),success=accepted,status=status,
             message=message,mip_gap=gap,solver_backend='gurobi',
             runtime_seconds=float(model.Runtime),node_count=float(model.NodeCount),
             best_bound=float(model.ObjBound),model_nvars=n,model_nrows=A.shape[0],
             certified_to_requested_gap=certified,
             accepted_time_limit_incumbent=relaxed,max_solution_violation=max_vio,
             mip_start_variables=start_count)

    def solve_exact_miqp(self,quadratic_terms,time_limit_s=20.,mip_rel_gap=.02,
                         maximum_time_limit_gap=None,mip_start=None):
        """Exact LL MIQP via the common, sparse, Gurobi solver path."""
        return self.solve_gurobi(time_limit_s,mip_rel_gap,quadratic_terms,
                                 maximum_time_limit_gap=maximum_time_limit_gap,
                                 mip_start=mip_start)

def add_quadratic_epigraph(m,energy_variable,target_mwhth,beta_usd_per_mwhth2,
                            capacity_mwhth,n_tangents=401):
    """MILP outer approximation for beta*(E-target)^2, no new terminal objective.

    Tangents over x=(E-target)/capacity in [-1,1]. The maximum epigraph
    underestimate (for x in [-1,1]) is <= beta*capacity**2/(n_tangents-1)**2.
    The approximation is purely computational: the intended penalty is quadratic.
    """
    if n_tangents<3 or n_tangents%2!=1:raise ValueError('n_tangents must be odd >=3')
    if beta_usd_per_mwhth2<0:raise ValueError('negative quadratic penalty')
    if beta_usd_per_mwhth2==0:return 0.
    weight=beta_usd_per_mwhth2*capacity_mwhth**2
    epi=m.v('terminal_soc_squared_gap_epigraph',0,np.inf,obj=weight)
    for a in np.linspace(-1.,1.,n_tangents):
        # epi >= 2*a*((E-target)/capacity) - a*a
        m.c({epi:1.,energy_variable:-2*a/capacity_mwhth},
            lo=-2*a*target_mwhth/capacity_mwhth-a*a)
    return float(weight/(n_tangents-1)**2)
