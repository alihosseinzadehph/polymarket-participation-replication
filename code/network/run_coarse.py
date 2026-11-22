"""Objective-preserving all-link quotient refinement; not final inference."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sparse_storage import digest, guard, load_storage, memory, save_json
from coarse_refine import refine_coarse_leiden
from run_sparse import internal_components


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage',type=Path,required=True)
    p.add_argument('--initial-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--max-iterations',type=int,default=30)
    p.add_argument('--max-seconds',type=float,default=180.)
    p.add_argument('--reserve-gib',type=float,default=3.)
    args=p.parse_args();start=time.perf_counter()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Existing output preserved')
    out.mkdir(parents=True,exist_ok=True)
    arrays,storage=load_storage(args.storage)
    prior=json.loads((args.initial_run/'result.json').read_text(encoding='utf-8'))
    sh=digest(args.storage/'storage.json')
    if prior['storage_manifest_sha256']!=sh:raise ValueError('Initial partition population mismatch')
    wl=[]
    for kind in ['wallet','event']:
        src=args.initial_run/(kind+'_memberships.parquet')
        spec=next(s for s in prior['memberships'] if s['file']==src.name)
        if digest(src)!=spec['sha256']:raise ValueError('Initial membership hash mismatch')
        table=pq.read_table(src,use_threads=False)
        if not np.array_equal(table['original_dimension_id'].to_numpy(),arrays[kind+'_original_ids']):
            raise ValueError('Original node IDs or order changed')
        wl.append(table['community'].to_numpy())
    before=guard(args.reserve_gib)
    snapshot=out/'code';snapshot.mkdir()
    for name in ['run_coarse.py','coarse_refine.py','run_sparse.py','sparse_optimizer.py',
                 'sparse_storage.py','connectivity.py','COARSE_REFINEMENT.md','METHOD.md']:
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    metadata={'status':'running','stage':'all_link_quotient_refinement',
        'scope':storage['scope'],'seed':args.seed,'weight_mode':prior['weight_mode'],
        'gamma':prior['gamma'],'storage_manifest_sha256':sh,
        'initial_result_sha256':digest(args.initial_run/'result.json'),
        'coarse_code_sha256':digest(Path(__file__).with_name('coarse_refine.py')),
        'runner_sha256':digest(__file__),'n_wallets':storage['n_wallets'],
        'n_events':storage['n_events'],'n_links':storage['n_links'],'nodes_excluded':0,
        'links_excluded':0,'community_count_fixed':False,'inferential_p_value':None,
        'memory_before':before,'no_new_download':True}
    save_json(out/'run_status.json',metadata)
    last=[0.]
    def callback(item):
        record={**item,'process_memory':guard(args.reserve_gib)}
        if item['stage']!='coarse_input' or time.perf_counter()-last[0]>20:
            print(json.dumps(record),flush=True);last[0]=time.perf_counter()
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
    result=refine_coarse_leiden(arrays['wallet_indptr'],arrays['wallet_events'],arrays['wallet_weights'],
           wl[0],wl[1],work_directory=out,seed=args.seed,weight_mode=prior['weight_mode'],
           gamma=prior['gamma'],max_iterations=args.max_iterations,max_seconds=args.max_seconds,
           callback=callback,return_quotient=True)
    w=result.pop('wallet_labels');e=result.pop('event_labels')
    original_group_labels=result.pop('original_group_labels',None)
    result.pop('wallet_groups',None);result.pop('event_groups',None)
    qe=result.pop('quotient_edges',None);qw=result.pop('quotient_weights',None)
    coarse_result=result.get('coarse_result',{})
    cl=coarse_result.pop('membership',None)
    if cl is not None:
        if original_group_labels is None:original_group_labels=np.unique(np.concatenate(wl))
        pq.write_table(pa.table({'original_community':original_group_labels,'coarse_community':cl}),
                       out/'coarse_group_mapping.parquet',compression='zstd')
    if qe is not None:
        pq.write_table(pa.table({'source_group_index':qe[:,0],'target_group_index':qe[:,1],
              'analysis_weight':qw}),out/'complete_quotient_edges.parquet',compression='zstd')
    unique=np.unique(np.concatenate((w,e)));n=len(unique)
    remap=np.full(int(unique[-1])+1,-1,dtype=np.int32);remap[unique]=np.arange(n,dtype=np.int32)
    w=remap[w];e=remap[e]
    specs=[]
    for kind,labels in [('wallet',w),('event',e)]:
        target=out/(kind+'_memberships.parquet')
        pq.write_table(pa.table({'original_dimension_id':arrays[kind+'_original_ids'],'community':labels}),
                       target,compression='zstd')
        specs.append({'file':target.name,'sha256':digest(target),'rows':len(labels)})
    roots=internal_components(arrays['wallet_indptr'],arrays['wallet_events'],w,e)[:n]
    pq.write_table(pa.table({'community':np.arange(n,dtype=np.int32),
        'wallets':np.bincount(w,minlength=n),'events':np.bincount(e,minlength=n),
        'wallet_raw_participation_usd':np.bincount(w,weights=arrays['wallet_strength_raw'],minlength=n),
        'event_raw_participation_usd':np.bincount(e,weights=arrays['event_strength_raw'],minlength=n),
        'internal_weak_components':roots}),out/'community_summary.parquet',compression='zstd')
    report={**metadata,'status':result['status'],'refinement':result,'memberships':specs,
       'q_barber_checked':result.get('Q_after',prior['q_barber_checked']),
       'n_communities':n,'all_nodes_assigned':True,
       'groups_with_multiple_internal_components':int(np.count_nonzero(roots>1)),
       'maximum_internal_components':int(roots.max()),'memory_after':memory(),
       'total_seconds':time.perf_counter()-start,
       'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
       'original_network_converged':False,'subsequent_sparse_refinement_required':True}
    save_json(out/'result.json',report)
    save_json(out/'run_status.json',{'status':report['status'],'result':'result.json'})
    print(json.dumps({k:v for k,v in report.items() if k!='refinement'}),flush=True)


if __name__=='__main__':main()
