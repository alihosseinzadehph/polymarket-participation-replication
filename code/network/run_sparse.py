"""Run the all-observed-node sparse optimizer with explicit resource budgets.

This produces descriptive optimized partitions, not null-test p-values. It never
downloads, filters participants, projects the graph, or modifies the manuscript.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
from sparse_storage import digest, guard, load_storage, memory, save_json
from sparse_optimizer import optimize_bipartite, modularity


@njit(cache=True)
def internal_components(wp,we,wlabels,elabels):
    nw=len(wlabels);n=nw+len(elabels)
    parent=np.arange(n,dtype=np.int32);size=np.ones(n,dtype=np.int32)
    for u in range(nw):
        for j in range(wp[u],wp[u+1]):
            v=we[j]
            if wlabels[u]!=elabels[v]:continue
            a=u;b=nw+v
            while parent[a]!=a:
                parent[a]=parent[parent[a]];a=parent[a]
            while parent[b]!=b:
                parent[b]=parent[parent[b]];b=parent[b]
            if a!=b:
                if size[a]<size[b]:a,b=b,a
                parent[b]=a;size[a]+=size[b]
    roots=np.zeros(n,dtype=np.int64)
    for i in range(n):
        if parent[i]==i:
            c=wlabels[i] if i<nw else elabels[i-nw]
            roots[c]+=1
    return roots


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--mode',choices=['amount','wallet_share'],required=True)
    parser.add_argument('--max-sweeps',type=int,default=100)
    parser.add_argument('--max-merge-rounds',type=int,default=100)
    parser.add_argument('--max-seconds',type=float,default=900)
    parser.add_argument('--reserve-gib',type=float,default=3.)
    parser.add_argument('--gamma',type=float,default=1.)
    parser.add_argument('--verify-storage-hashes',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args();start=time.perf_counter()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Existing run preserved; use empty output')
    out.mkdir(parents=True,exist_ok=True)
    arrays,storage=load_storage(args.storage,args.verify_storage_hashes)
    before=guard(args.reserve_gib)
    snapshot=out/'code';snapshot.mkdir()
    for filename in ['run_sparse.py','sparse_optimizer.py','sparse_storage.py','connectivity.py','METHOD.md']:
        shutil.copyfile(Path(__file__).with_name(filename),snapshot/filename)
    initial={'status':'running','seed':args.seed,'weight_mode':args.mode,'gamma':args.gamma,
             'scope':storage['scope'],'storage_manifest_sha256':digest(args.storage/'storage.json'),
             'optimizer_sha256':digest(Path(__file__).with_name('sparse_optimizer.py')),
             'connectivity_sha256':digest(Path(__file__).with_name('connectivity.py')),
             'runner_sha256':digest(__file__),'memory_before':before,
             'max_sweeps':args.max_sweeps,'max_merge_rounds':args.max_merge_rounds,
             'max_seconds':args.max_seconds,'reserve_gib':args.reserve_gib,
             'n_wallets':storage['n_wallets'],'n_events':storage['n_events'],'n_links':storage['n_links'],
             'nodes_excluded':0,'links_excluded':0,'community_count_fixed':False,
             'inferential_p_value':None,'no_new_download':True}
    save_json(out/'run_status.json',initial)
    last_print=[0.]
    def callback(item):
        record={**item,'process_memory':guard(args.reserve_gib)}
        if item['stage'] in ('strengths','initial','finished') or time.perf_counter()-last_print[0]>20:
            print(json.dumps(record,allow_nan=False),flush=True)
            last_print[0]=time.perf_counter()
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
    resume={}
    if args.resume_from:
        previous=json.loads((args.resume_from/'result.json').read_text(encoding='utf-8'))
        if previous['storage_manifest_sha256']!=initial['storage_manifest_sha256']:
            raise ValueError('Resume population differs')
        if previous['weight_mode']!=args.mode or previous['gamma']!=args.gamma:
            raise ValueError('Resume objective differs')
        for kind in ['wallet','event']:
            p=args.resume_from/(kind+'_memberships.parquet')
            oldspec=next(s for s in previous['memberships'] if s['file']==p.name)
            if digest(p)!=oldspec['sha256']:raise ValueError('Resume membership hash changed')
            table=pq.read_table(p,use_threads=False)
            if not np.array_equal(table['original_dimension_id'].to_numpy(),arrays[kind+'_original_ids']):
                raise ValueError('Resume ID order differs')
            resume['initial_'+kind+'_labels']=table['community'].to_numpy()
        initial['resume_result_sha256']=digest(args.resume_from/'result.json')
        initial['resume_note']='Same objective and population; seeded continuation, not exact uninterrupted RNG restart'
    result=optimize_bipartite(*(arrays[k] for k in ['wallet_indptr','wallet_events','wallet_weights',
           'event_indptr','event_wallets','event_weights']),seed=args.seed,gamma=args.gamma,
           weight_mode=args.mode,max_sweeps=args.max_sweeps,max_merge_rounds=args.max_merge_rounds,
           max_seconds=args.max_seconds,callback=callback,**resume)
    w=np.asarray(result.pop('wallet_labels'));e=np.asarray(result.pop('event_labels'))
    if len(w)!=storage['n_wallets'] or len(e)!=storage['n_events']:raise AssertionError('Node count changed')
    if np.any(w<0) or np.any(e<0):raise AssertionError('Unassigned node')
    # Common canonical labels, independent of optimizer's internal label IDs.
    unique=np.unique(np.concatenate((w,e)))
    remap=np.full(int(unique[-1])+1,-1,dtype=np.int32);remap[unique]=np.arange(len(unique),dtype=np.int32)
    w=remap[w];e=remap[e];n=len(unique)
    filenames=[]
    for name,ids,labels in [('wallet',arrays['wallet_original_ids'],w),('event',arrays['event_original_ids'],e)]:
        p=out/(name+'_memberships.parquet')
        pq.write_table(pa.table({'original_dimension_id':ids,'community':labels}),p,compression='zstd')
        filenames.append({'file':p.name,'sha256':digest(p),'rows':len(labels)})
    independent_q=modularity(arrays['wallet_indptr'],arrays['wallet_events'],arrays['wallet_weights'],
                              w,e,weight_mode=args.mode,gamma=args.gamma)
    if abs(independent_q-result['Q'])>2e-10:raise AssertionError('Final canonical-label objective mismatch')
    wc=np.bincount(w,minlength=n);ec=np.bincount(e,minlength=n)
    ws=np.bincount(w,weights=arrays['wallet_strength_raw'],minlength=n)
    es=np.bincount(e,weights=arrays['event_strength_raw'],minlength=n)
    roots=internal_components(arrays['wallet_indptr'],arrays['wallet_events'],w,e)[:n]
    pq.write_table(pa.table({'community':np.arange(n,dtype=np.int32),'wallets':wc,'events':ec,
         'wallet_raw_participation_usd':ws,'event_raw_participation_usd':es,
         'internal_weak_components':roots}),out/'community_summary.parquet',compression='zstd')
    report={**initial,'status':'finished_for_declared_budget_not_a_significance_test',
            'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'optimizer':result,
            'q_barber_checked':float(independent_q),'n_communities':n,
            'all_nodes_assigned':True,'memberships':filenames,
            'groups_with_multiple_internal_components':int(np.count_nonzero(roots>1)),
            'maximum_internal_components':int(roots.max()),'memory_after':memory(),
            'total_seconds':time.perf_counter()-start,
            'stopping_note':'Inspect optimizer stop reason; exhausted budget is not convergence. Optimized Q is not a null-test p-value.'}
    save_json(out/'result.json',report)
    save_json(out/'run_status.json',{'status':report['status'],'result':'result.json'})
    print(json.dumps({k:v for k,v in report.items() if k!='optimizer'},allow_nan=False),flush=True)
    gc.collect()


if __name__=='__main__':main()
