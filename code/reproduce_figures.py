"""Rebuild the six main figures from released numerical data in a disposable workspace."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
INPUTS = ['Fig1_concentration/concentration_rank_plot_data.csv',
          'Fig2_powerlaw/three_tail_plot_points.csv', 'Fig2_powerlaw/three_tail_fit_summary.csv',
          'Fig3_subject_shares/fullnetwork_statistics.json',
          'Fig4_complete_groups/atlas/island_data.json','Fig4_complete_groups/atlas/garden_xy.json',
          'Fig5_subject_networks/subject_networks.json','Fig5_subject_networks/subject_bicm_full.json',
          'Fig6_8_election/multiview.json','Fig6_8_election/analysis.json']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE.parent/'data/Source_Data')
    p.add_argument('--output', type=Path, required=True, help='Fresh directory for plots and diagnostics.')
    args=p.parse_args()
    out=args.output.resolve()
    if out.exists() or out.is_relative_to(args.source.resolve()):
        p.error('--output must be a new directory outside the released source data')
    for rel in INPUTS:
        if not (args.source/rel).is_file(): p.error('Missing source file: '+rel)
    with tempfile.TemporaryDirectory(prefix='polymarket-figures-') as tmp:
        work=Path(tmp)
        for rel in INPUTS:
            dest=work/'Source_Data'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.source/rel,dest)
        tools=work/'reproducibility'
        shutil.copytree(HERE/'figures',tools,ignore=shutil.ignore_patterns('__pycache__'))
        for cmd in (['build_reviewed_main_figures.py','--figures','1','2','4','6'],
                    ['draw_subject_shares.py'], ['build_subject_networks_journal.py']):
            subprocess.run([sys.executable,str(tools/cmd[0]),*cmd[1:]],cwd=work,check=True)
        files=list((work/'figures').glob('*.pdf'))
        if len(files)!=6: raise RuntimeError(f'Expected six PDFs; found {[f.name for f in files]}')
        shutil.copytree(work/'figures',out)
        shutil.copyfile(work/'Source_Data/FIGURE_REBUILD_CURRENT.json',out/'figure_1246_audit.json')
        shutil.copyfile(work/'SUBJECT_NETWORKS_JOURNAL_PROVENANCE_20261004.json',out/'figure_5_provenance.json')
        print(json.dumps({'output':str(out),'pdfs':sorted(f.name for f in files)}))

if __name__=='__main__': main()
