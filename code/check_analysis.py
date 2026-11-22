"""Re-estimate numerical results from the released inputs; reads data without modifying it."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
for dirname in ['tails','network','election']:
    sys.path.insert(0,str(HERE/dirname))

def readcsv(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def readjson(p): return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE.parent/'data/Source_Data')
    parser.add_argument('--study-data',type=Path,default=HERE.parent/'data',help='Parent of study/ and leaderboard/ inputs')
    parser.add_argument('--tail-bootstrap',action='store_true',help='Replay all 2,500 goodness-of-fit and 1,000 CI draws per monetary sample')
    parser.add_argument('--election-bootstrap',action='store_true',help='Replay 1,999 eligible-wallet resamples per view')
    parser.add_argument('--bicm-simulations',action='store_true',help='Replay 19,999 marginal trials per subject pair')
    parser.add_argument('--output',type=Path,help='Optional new .json file for machine-readable results')
    args=parser.parse_args()
    if args.output and (args.output.exists() or any(args.output.resolve().is_relative_to(d.resolve()) for d in [args.source,args.study_data])):
        parser.error('Use a new output file outside the data directory.')
    from estimators import continuous_fit,_init_worker,_one_continuous
    from permutation_audit import load_vectors,observed_metric_jsd
    from subject_null import analyze
    import pyarrow.parquet as pq
    released=args.source
    taildir=released/'Fig2_powerlaw'
    points=readcsv(taildir/'three_tail_plot_points.csv')
    seeds={'top1000_selected':2026091902,'leaderboard_pnl':2026091905,'leaderboard_volume':2026091906}
    event_amount=np.sort(pq.read_table(args.study_data/'study/mapped_event_totals.parquet',columns=['usd_volume']).column(0).to_numpy())
    assert len(event_amount)==764378 and np.isfinite(event_amount).all() and event_amount.min()>0
    n=len(event_amount)
    gini=2*np.dot(np.arange(1,n+1,dtype=float),event_amount)/(n*event_amount.sum())-(n+1)/n
    top=float(event_amount[-int(np.ceil(n*.01)):].sum()/event_amount.sum())
    np.testing.assert_allclose([gini,top],[.932587391694129,.6400372219579931],atol=1e-8)
    wallet_amount=np.sort(pq.read_table(args.study_data/'study/wallet_amounts',columns=['participant_volume_usdc']).column(0).to_numpy())
    nw=len(wallet_amount)
    assert nw==3106517 and np.isfinite(wallet_amount).all() and wallet_amount.min()>=0
    gw=float(2*np.dot(np.arange(1,nw+1,dtype=float),wallet_amount)/(nw*wallet_amount.sum())-(nw+1)/nw)
    tw=float(wallet_amount[-int(np.ceil(nw*.01)):].sum()/wallet_amount.sum())
    wref=readcsv(released/'Fig1_concentration/wallet_concentration.csv')[0]
    np.testing.assert_allclose([gw,tw],[float(wref['gini']),float(wref['top_0.01_share'])],atol=1e-8)
    degree=readcsv(args.study_data/'study/bipartite_degree_counts.csv')
    for side,nodes in [('wallet',3106514),('event',764378)]:
        part=[r for r in degree if r['node_partition']==side]
        assert sum(int(r['nodes_at_degree']) for r in part)==nodes
        assert sum(int(r['nodes_at_degree'])*int(r['degree']) for r in part)==255704359
    board=readcsv(args.study_data/'leaderboard/overall_all_top1000.csv')
    source_values={'top1000_selected':event_amount[-1000:]}
    for k,order,col in [('leaderboard_pnl','PNL','reported_pnl'),('leaderboard_volume','VOL','reported_volume')]:
        subset=[r for r in board if r['order_by']==order]
        assert sorted(int(r['rank']) for r in subset)==list(range(1,1001))
        source_values[k]=np.sort([float(r[col]) for r in subset])
    out={'concentration':{'event_gini':float(gini),'top_one_percent_event_amount_share':top,
                          'wallet_gini':gw,'top_one_percent_wallet_amount_share':tw},
         'mapped_graph':{'wallets':3106514,'events':764378,'wallet_event_edges_from_both_degree_tables':255704359},
         'powerlaw':{},'election':{},'bicm':{}}
    gdir=released/'Fig4_complete_groups'
    g=pq.read_table(gdir/'amount_final_group_summary.parquet').to_pydict()
    q=pq.read_table(gdir/'amount_final_quotient.parquet').to_pydict()
    w=np.asarray(g['wallet_analysis_strength']);e=np.asarray(g['event_analysis_strength']);m=w.sum()
    np.testing.assert_allclose([e.sum(),sum(q['analysis_weight'])],[m,m],rtol=1e-11)
    modularity=float(sum(g['internal_analysis_weight'])/m-(w/m)@(e/m))
    edges=set(tuple(sorted((a,b))) for a,b in zip(q['wallet_community'],q['event_community']) if a!=b)
    assert (len(w),len(edges),sum(g['wallets']),sum(g['events']),sum(q['original_link_count']))==(149,1289,3106514,764378,255704359)
    np.testing.assert_allclose(modularity,.6729080437562294,atol=1e-10)
    out['mapped_graph'].update(amount_groups=len(w),off_diagonal_undirected_group_pairs=len(edges),amount_bipartite_modularity_at_gamma_one=modularity)
    for row in readcsv(taildir/'three_tail_fit_summary.csv'):
        key=row['key']
        x=np.sort([float(r['x']) for r in points if r['key']==key])
        assert len(x)==int(row['n'])==1000
        np.testing.assert_allclose(x,source_values[key],rtol=1e-12,atol=1e-8)
        fit=continuous_fit(x)
        for name in ['n_tail','xmin','ks','alpha']: np.testing.assert_allclose(fit[name],float(row[name]),rtol=1e-9)
        report={k:fit[k] for k in ['n','n_tail','xmin','ks','alpha']}
        if args.tail_bootstrap:
            _init_worker(x,fit,dict(seed=seeds[key],minimum=100,candidate_budget=500,exhaustive_limit=2000))
            draws=[_one_continuous(i,'gof') for i in range(2500)]
            assert all(d['valid'] for d in draws)
            p=float(np.mean([d['exceed_ge'] for d in draws]))
            np.testing.assert_allclose(p,float(row['p_csn_raw']))
            ci=np.quantile([_one_continuous(i,'ci')['alpha'] for i in range(1000)],[.025,.975])
            np.testing.assert_allclose(ci,[float(row['alpha_ci_low']),float(row['alpha_ci_high'])],rtol=1e-8)
            report.update(p_csn_raw=p,alpha_ci=ci.tolist())
        out['powerlaw'][key]=report
    election=released/'Fig6_8_election'
    _,groups=load_vectors(election/'election_october_wallet_vectors.csv')
    observed=readjson(election/'election_permutation_audit.json')['direct_bottom_minus_top_test']['metrics']
    for name,first,second in [('bottom','HB','TB'),('top','HT','TT')]:
        metrics,_=observed_metric_jsd(groups[first],groups[second])
        for metric,value in metrics.items():
            np.testing.assert_allclose(value,observed[metric][name+'_jsd_bits'],atol=1e-12)
        out['election'][name+'_jsd_bits']=metrics
    if args.election_bootstrap:
        from bootstrap import bootstrap_delta
        f=election/'election_october_eligible_wallet_vectors.csv'
        rows=readcsv(f); fields=list(rows[0])[4:]
        data={}
        for key,name in [('HB','Harris Bottom 5%'),('TB','Trump Bottom 5%'),('HT','Harris Top 5%'),('TT','Trump Top 5%')]:
            data[name]=np.array([[float(r[s]) for s in fields] for r in rows if r['cohort']==key])
        assert sum(map(len,data.values()))==1630
        boot=bootstrap_delta(data,np.random.default_rng(20269925))
        ref=readjson(election/'multiview.json')['bootstrap_convergence']
        for k in ref:
            for field in ref[k]:
                np.testing.assert_allclose(boot[k][field],ref[k][field],rtol=0,atol=1e-12)
        out['election']['eligible_bootstrap']=boot
    sd=released/'Fig5_subject_networks'
    full=analyze(readjson(sd/'subject_networks.json'),'full',19999 if args.bicm_simulations else 0)
    reference=readjson(sd/'subject_bicm_full.json')
    assert len(full['rows'])==len(reference['rows'])==210
    for computed,expected in zip(full['rows'],reference['rows']):
        for name in ['a','b','observed_shared_wallets','null_expected','null_sd']+(['p_holm','upper_tail_exceedances','enriched_holm_005'] if args.bicm_simulations else []):
            np.testing.assert_allclose(computed[name],expected[name],rtol=1e-8,atol=1e-8)
    out['bicm']={'pair_expectations_checked':210,'null_simulations_per_pair':19999 if args.bicm_simulations else 0}
    text=json.dumps(out,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f: f.write(text+'\n')
    print(text)

if __name__=='__main__': main()
