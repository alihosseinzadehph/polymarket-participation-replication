"""Frozen BiCM fit, seeded marginal null simulations and Holm correction."""
import numpy as np
from scipy.special import expit
from scipy.optimize import least_squares
from scipy.stats import beta

def fit_bicm(counts,columns):
    counts=np.asarray(counts,np.int64);columns=np.asarray(columns,np.int64)
    p=np.zeros((len(counts),len(columns)));r=counts>0;c=np.ones(len(columns),bool)
    d=np.arange(len(counts),dtype=float);target=columns.astype(float).copy()
    while True:
        old=(int(r.sum()),int(c.sum()))
        zero=r&(d==0);r[zero]=False
        full=r&(d==c.sum())
        if full.any():
            p[np.ix_(full,c)]=1;target[c]-=counts[full].sum();r[full]=False
        total=counts[r].sum()
        if not r.any() or not c.any():break
        zero=c&(target==0);c[zero]=False
        full=c&(target==total)
        if full.any():p[np.ix_(r,full)]=1;d[r]-=int(full.sum());c[full]=False
        if old==(int(r.sum()),int(c.sum())):break
    fitinfo={'iterations':0,'interior_row_classes':int(r.sum()),'interior_columns':int(c.sum())}
    if r.any() and c.any():
        nr,nc=r.sum(),c.sum();n=counts[r].astype(float);N=n.sum();want=d[r];col=target[c]
        def probabilities(theta):return expit(theta[:nr,None]+np.r_[theta[nr:],0][None,:])
        def fun(theta):
            q=probabilities(theta)
            return np.r_[q.sum(axis=1)-want,((n[:,None]*q).sum(axis=0)-col)[:-1]/N]
        def jac(theta):
            q=probabilities(theta);v=q*(1-q);j=np.zeros((nr+nc-1,nr+nc-1))
            j[:nr,:nr]=np.diag(v.sum(axis=1));j[:nr,nr:]=v[:,:-1]
            j[nr:,:nr]=(n[:,None]*v[:,:-1]).T/N
            j[nr:,nr:]=np.diag((n[:,None]*v).sum(axis=0)[:-1]/N)
            return j
        initial=np.r_[np.log(want/(nc-want)),np.zeros(nc-1)]
        fit=least_squares(fun,initial,jac=jac,xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=3000,x_scale='jac')
        assert fit.success,fit.message
        p[np.ix_(r,c)]=probabilities(fit.x)
        fitinfo.update(iterations=int(fit.nfev),solver_message=fit.message)
    row_error=float(np.max(abs(p.sum(axis=1)[counts>0]-np.arange(len(counts))[counts>0])))
    col_error=float(np.max(abs((counts[:,None]*p).sum(axis=0)-columns)))
    assert row_error<1e-7 and col_error<.02,(row_error,col_error)
    fitinfo.update(max_row_degree_error=row_error,max_column_degree_error=col_error)
    return p,fitinfo


def holm(p):
    p=np.asarray(p);order=np.argsort(p,kind='stable');out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def analyze(data,label,B=0):
    counts=np.asarray(data['subject_degree_counts'],np.int64);shared=np.asarray(data['shared'],np.int64)
    p,info=fit_bicm(counts,shared.diagonal());rows=[]
    for a in range(shared.shape[0]):
        for b in range(a+1,shared.shape[0]):
            prob=p[:,a]*p[:,b];mean=float(counts@prob);var=float(counts@(prob*(1-prob)));observed=int(shared[a,b])
            row={'a':a,'b':b,'observed_shared_wallets':observed,'null_expected':mean,'null_sd':float(np.sqrt(var)),
                 'observed_expected_ratio':observed/mean if mean>0 else None,
                 'z':float((observed-mean)/np.sqrt(var)) if var>0 else None}
            if B:
                rng=np.random.default_rng(np.random.SeedSequence([20260916,a,b]))
                samples=np.zeros(B,np.int64)
                for k in np.flatnonzero(counts):samples+=rng.binomial(int(counts[k]),prob[k],size=B)
                exceed=int((samples>=observed).sum())
                row.update(upper_tail_exceedances=exceed,replicates=B,p_upper=(exceed+1)/(B+1),
                    p_upper_mc_low=float(beta.ppf(.025,exceed,B-exceed+1)) if exceed else 0.,
                    p_upper_mc_high=float(beta.ppf(.975,exceed+1,B-exceed)) if exceed<B else 1.,
                    simulated_mean=float(samples.mean()),simulated_sd=float(samples.std(ddof=1)))
            rows.append(row)
    if B:
        adj=holm([r['p_upper'] for r in rows]);lo=holm([r['p_upper_mc_low'] for r in rows]);hi=holm([r['p_upper_mc_high'] for r in rows])
        for row,padj,l,h in zip(rows,adj,lo,hi):
            row.update(p_holm=float(padj),p_holm_mc_lower=float(l),p_holm_mc_upper=float(h),enriched_holm_005=bool(padj<=.05))
    return {'period':label,'fit':info,'rows':rows,'probabilities_by_subject_degree':p.tolist(),
            'subject_degree_counts':counts.tolist(),'canonical_constraints':'row subject breadth and subject popularity in expectation'}
