"""Figure 3: isolated category panel from the original full-network plotting script."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
stats=json.loads((root/'Source_Data/Fig3_subject_shares/fullnetwork_statistics.json').read_text(encoding='utf-8'))
assert stats['status']=='pass'
ordered=sorted(stats['categories'],key=lambda r:r['participation_usd'],reverse=True)
names=[r['category'].replace('—','-').replace('–','-') for r in ordered]
money=np.array([r['participation_usd']/stats['population']['mapped_participation_usd'] for r in ordered])*100
wallets=np.array([r['sum_wallet_fractions']/stats['population']['active_wallets'] for r in ordered])*100
assert len(names)==21 and np.allclose([money.sum(),wallets.sum()],100)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
y=np.arange(len(ordered))
fig,ax=plt.subplots(figsize=(7.0,5.4),layout='constrained')
ax.hlines(y,np.minimum(money,wallets),np.maximum(money,wallets),color='#b9bfc5',lw=1.1)
ax.scatter(money,y-.07,color='#a84325',s=25,label='Share of mapped amount',zorder=3)
ax.scatter(wallets,y+.07,color='#245d85',marker='s',s=22,label='Mean within-wallet share',zorder=3)
ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Category share (%) - logarithmic scale')
ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=8)
for ext,opts in [('pdf',{}),('png',{'dpi':170})]:
    fig.savefig(root/'figures'/('full_network_category_shares.'+ext),**opts)
plt.close(fig)
