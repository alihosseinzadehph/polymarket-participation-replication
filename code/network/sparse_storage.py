"""Lossless, bounded-batch CSR/CSC storage for the observed participation graph.

No raw-fill scan, sampling, projection, activity threshold or float32 amounts.
All original positive links survive; IDs are remapped with explicit inverse maps.
The global input is strictly sorted by (wallet_id,event_index). The explicit
--unsorted-input option handles quarter files by sorting individual sparse rows.
A failed build is never published as a completed storage manifest.
"""
from __future__ import annotations
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow.parquet as pq
from numba import njit

GIB = 2**30


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pending.replace(path)


def memory():
    class Status(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
            (name,ctypes.c_ulonglong) for name in ['total','available','totalpage',
            'availablepage','totalvirtual','availablevirtual','extended']]
    s = Status(); s.length = ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise OSError('Cannot inspect available physical memory')
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)] + [
            (name,ctypes.c_size_t) for name in ['peak','working','p1','p2','p3','p4','commit','peakcommit']]
    c = Counters(); c.cb = ctypes.sizeof(c)
    get = ctypes.windll.kernel32.GetCurrentProcess; get.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not query(get(),ctypes.byref(c),c.cb):
        raise OSError('Cannot inspect process memory')
    return {'physical_total_bytes':int(s.total),'available_bytes':int(s.available),
            'working_set_bytes':int(c.working),'peak_working_set_bytes':int(c.peak),
            'committed_bytes':int(c.commit),'peak_committed_bytes':int(c.peakcommit)}


def guard(reserve_gib=3.):
    m=memory()
    if m['available_bytes'] < reserve_gib*GIB:
        raise MemoryError('Available memory below reserve; stopped without filtering: '+str(m))
    return m


@njit(cache=True)
def count_batch(w,e,a,f,dw,de,sw,se,previous_w,previous_e,require_order=True):
    fills=0
    for k in range(len(w)):
        u=int(w[k]); v=int(e[k]); x=a[k]
        if u<0 or u>=len(dw) or v<0 or v>=len(de):
            raise ValueError('ID outside supplied dimensions')
        if not np.isfinite(x) or x<=0 or f[k]<=0:
            raise ValueError('Nonpositive or nonfinite input')
        if require_order and (u<previous_w or (u==previous_w and v<=previous_e)):
            raise ValueError('Pairs not strictly lexicographically sorted/unique')
        dw[u]+=1; de[v]+=1; sw[u]+=x; se[v]+=x; fills+=f[k]
        previous_w=u; previous_e=v
    return previous_w,previous_e,fills


@njit(cache=True)
def scatter(w,e,a,wmap,emap,we,ww,ew,ev,ecursor,wcursor):
    for k in range(len(w)):
        u=wmap[w[k]]; v=emap[e[k]]
        if u<0 or v<0: raise ValueError('Active endpoint missing from remapping')
        pos=wcursor[u];we[pos]=v;ww[pos]=a[k];wcursor[u]+=1
        pos=ecursor[v]; ew[pos]=u; ev[pos]=a[k]; ecursor[v]+=1


@njit(cache=True)
def sort_rows(ptr,neighbors,weights):
    for i in range(len(ptr)-1):
        lo=ptr[i];hi=ptr[i+1]
        if hi-lo<2: continue
        order=np.argsort(neighbors[lo:hi])
        ids=neighbors[lo:hi].copy();amounts=weights[lo:hi].copy()
        for k in range(hi-lo):
            neighbors[lo+k]=ids[order[k]];weights[lo+k]=amounts[order[k]]


@njit(cache=True)
def validate_transpose(wp,we,ww,ep,ew,ev):
    # In each CSC column, original-wallet order is strictly increasing. Cursors
    # find the exact transposed CSR link; amounts must be bit-for-bit identical.
    cursor=ep[:-1].copy()
    for u in range(len(wp)-1):
        previous=-1
        for k in range(wp[u],wp[u+1]):
            v=we[k]
            if v<=previous: raise ValueError('Nonunique or unsorted CSR row')
            previous=v
            j=cursor[v]
            if j>=ep[v+1] or ew[j]!=u or ev[j]!=ww[k]:
                raise ValueError('CSR/CSC exact transpose mismatch')
            cursor[v]+=1
    for v in range(len(ep)-1):
        if cursor[v]!=ep[v+1]: raise ValueError('Unused CSC link')
    return True


def batches(pf, columns):
    for batch in pf.iter_batches(batch_size=262144,columns=columns,use_threads=False):
        yield [batch.column(i).to_numpy(zero_copy_only=False) for i in range(len(columns))]


def load_storage(folder, verify_hashes=False):
    folder=Path(folder)
    m=json.loads((folder/'storage.json').read_text(encoding='utf-8'))
    if m['status']!='validated_complete': raise ValueError('Incomplete storage')
    arrays={}
    for key,spec in m['arrays'].items():
        p=folder/spec['file']
        if p.stat().st_size!=spec['bytes']: raise ValueError('Array size changed: '+key)
        if verify_hashes and digest(p)!=spec['sha256']: raise ValueError('Array hash changed: '+key)
        arrays[key]=np.load(p,mmap_mode='r',allow_pickle=False)
    return arrays,m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dimension-manifest',type=Path,required=True)
    parser.add_argument('--scope',required=True)
    parser.add_argument('--unsorted-input',action='store_true')
    parser.add_argument('--reserve-gib',type=float,default=3.)
    parser.add_argument('--disk-reserve-gib',type=float,default=15.)
    args=parser.parse_args(); began=time.perf_counter()
    source=args.input.resolve(); out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory; existing outputs are preserved')
    out.mkdir(parents=True,exist_ok=True)
    dims=json.loads(args.dimension_manifest.read_text(encoding='utf-8'))
    nw0=dims['files']['wallets']['rows']; ne0=dims['files']['events']['rows']
    if max(nw0,ne0)>=2**31: raise ValueError('int32 ID capacity exceeded')
    pf=pq.ParquetFile(source); L=pf.metadata.num_rows
    if L<=0: raise ValueError('Empty graph')
    required=24*L+24*(nw0+ne0+2)
    free=shutil.disk_usage(out).free
    if free<required+args.disk_reserve_gib*GIB: raise OSError('Insufficient free disk plus reserve')
    before=guard(args.reserve_gib)
    save_json(out/'build_status.json',{'status':'building','source':str(source),'rows':L,
              'scope':args.scope,'memory_before':before,'estimated_array_bytes':required})
    print(json.dumps({'stage':'count_and_validate','links':L,'memory':before}),flush=True)
    dw=np.zeros(nw0,dtype=np.int64);de=np.zeros(ne0,dtype=np.int64)
    sw=np.zeros(nw0,dtype=np.float64);se=np.zeros(ne0,dtype=np.float64)
    lastw=-1;laste=-1;fills=0;seen=0;lastlog=time.perf_counter()
    for w,e,a,f in batches(pf,['wallet_id','event_index','usd','participation_fills']):
        lastw,laste,nf=count_batch(w,e,a,f,dw,de,sw,se,lastw,laste,not args.unsorted_input)
        fills+=int(nf);seen+=len(w)
        if time.perf_counter()-lastlog>15:
            print(json.dumps({'stage':'count','rows':seen,'memory':guard(args.reserve_gib)}),flush=True)
            lastlog=time.perf_counter()
    if seen!=L or dw.sum()!=L or de.sum()!=L: raise AssertionError('Input row conservation')
    wallet_ids=np.flatnonzero(dw).astype(np.int32);event_ids=np.flatnonzero(de).astype(np.int32)
    nw=len(wallet_ids);ne=len(event_ids)
    wmap=np.full(nw0,-1,dtype=np.int32);emap=np.full(ne0,-1,dtype=np.int32)
    wmap[wallet_ids]=np.arange(nw,dtype=np.int32);emap[event_ids]=np.arange(ne,dtype=np.int32)
    values={
        'wallet_indptr':np.r_[np.int64(0),np.cumsum(dw[wallet_ids],dtype=np.int64)],
        'event_indptr':np.r_[np.int64(0),np.cumsum(de[event_ids],dtype=np.int64)],
        'wallet_original_ids':wallet_ids,'event_original_ids':event_ids,
        'wallet_strength_raw':sw[wallet_ids],'event_strength_raw':se[event_ids]}
    arrays={}
    for key,value in values.items():
        np.save(out/(key+'.npy'),value,allow_pickle=False)
        arrays[key]=np.load(out/(key+'.npy'),mmap_mode='r',allow_pickle=False)
    for key,dtype in [('wallet_events','int32'),('wallet_weights','float64'),
                      ('event_wallets','int32'),('event_weights','float64')]:
        arrays[key]=np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=dtype,shape=(L,))
    cursor=arrays['event_indptr'][:-1].copy();wcursor=arrays['wallet_indptr'][:-1].copy()
    seen=0;lastlog=time.perf_counter()
    print(json.dumps({'stage':'stream_scatter','wallets':nw,'events':ne,'links':L}),flush=True)
    for w,e,a in batches(pf,['wallet_id','event_index','usd']):
        scatter(w,e,a,wmap,emap,arrays['wallet_events'],arrays['wallet_weights'],
                arrays['event_wallets'],arrays['event_weights'],cursor,wcursor)
        seen+=len(w)
        if time.perf_counter()-lastlog>15:
            print(json.dumps({'stage':'scatter','rows':seen,'memory':guard(args.reserve_gib)}),flush=True)
            lastlog=time.perf_counter()
    if seen!=L or not np.array_equal(cursor,arrays['event_indptr'][1:]):
        raise AssertionError('Transposed row conservation')
    if not np.array_equal(wcursor,arrays['wallet_indptr'][1:]): raise AssertionError('CSR row conservation')
    if args.unsorted_input:
        sort_rows(arrays['wallet_indptr'],arrays['wallet_events'],arrays['wallet_weights'])
        sort_rows(arrays['event_indptr'],arrays['event_wallets'],arrays['event_weights'])
    for key in ['wallet_events','wallet_weights','event_wallets','event_weights']: arrays[key].flush()
    print(json.dumps({'stage':'exact_transpose_validation','memory':guard(args.reserve_gib)}),flush=True)
    validate_transpose(arrays['wallet_indptr'],arrays['wallet_events'],arrays['wallet_weights'],
                       arrays['event_indptr'],arrays['event_wallets'],arrays['event_weights'])
    specs={}
    for key,arr in arrays.items():
        p=out/(key+'.npy');specs[key]={'file':p.name,'dtype':str(arr.dtype),
            'shape':list(arr.shape),'bytes':p.stat().st_size,'sha256':digest(p)}
    m={'status':'validated_complete','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
       'scope':args.scope,'source_path':str(source),'source_sha256':digest(source),
       'dimension_manifest_sha256':digest(args.dimension_manifest),'builder_sha256':digest(__file__),
       'n_wallets':nw,'n_events':ne,'n_links':L,'participation_fills':fills,
       'raw_participation_usd':float(np.sum(values['wallet_strength_raw'],dtype=np.float64)),
       'arrays':specs,'storage_bytes':sum(s['bytes'] for s in specs.values()),
       'source_required_sorted':not args.unsorted_input,'strict_sorted_unique_stored_pairs':True,
       'exact_CSR_CSC_transpose_verified':True,
       'positive_observed_nodes_excluded':0,'links_excluded':0,'weight_precision':'float64 unchanged',
       'original_IDs_preserved':True,'projection':False,'no_download':True,
       'dimension_wallets_without_input_links':nw0-nw,'dimension_events_without_input_links':ne0-ne,
       'memory_before':before,'memory_after':memory(),'seconds':time.perf_counter()-began,
       'reserve_note':'Available RAM checked between batches/stages; not an OS-enforced process limit'}
    save_json(out/'storage.json',m)
    save_json(out/'build_status.json',{'status':'validated_complete','manifest':'storage.json'})
    print(json.dumps({k:v for k,v in m.items() if k!='arrays'}),flush=True)


if __name__=='__main__':main()
