"""Rebuild Figures 1, 2, 4, 6 and the archived former Figure 7 from compact data.

No model is fitted, no raw archive is scanned and no observation is downloaded.
Figure 1 is the standalone Lorenz plot; Figure 4 is the subject-guided atlas.
The 22/30-edge election displays are not the matrices used for inference.
Requires numpy, pypdf and reportlab; optional pdftoppm creates PNG previews.
"""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
AUDIT = ROOT / "Source_Data" / "FIGURE_REBUILD_CURRENT.json"
FONT, BOLD = "Helvetica", "Helvetica-Bold"
if Path("C:/Windows/Fonts/arial.ttf").exists():
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    FONT, BOLD = "Arial", "Arial-Bold"
INK, GRAY = "#17324D", "#536B7F"
BLUE, ORANGE = "#235789", "#AD5318"
FAMILY = {"Crypto":"#148C83", "Sports":"#4179B0", "Politics":"#C7832C",
          "Finance":"#8663A8", "Culture":"#BE607C", "Meta":"#6B7786",
          "Other":"#6B7786", "Weather":"#AA8B18"}
INPUTS, CHECKS, OUTPUTS = {}, {}, []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(relative):
    p = ROOT / relative
    INPUTS[relative] = sha(p)
    return json.loads(p.read_text(encoding="utf-8"))

def read_csv(relative):
    p = ROOT / relative
    INPUTS[relative] = sha(p)
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def text(c, x, y, s, size=16, color=INK, align="left", bold=False):
    c.setFont(BOLD if bold else FONT, size)
    c.setFillColor(HexColor(color))
    getattr(c, {"left":"drawString", "center":"drawCentredString", "right":"drawRightString"}[align])(x,y,str(s))

def line(c, x1,y1,x2,y2,color=GRAY,width=1,dash=None,alpha=1):
    c.saveState(); c.setStrokeColor(HexColor(color)); c.setLineWidth(width)
    c.setStrokeAlpha(alpha)
    if dash: c.setDash(dash)
    c.line(x1,y1,x2,y2); c.restoreState()

def circle(c,x,y,r,color,outline=None,linewidth=1):
    c.setFillColor(HexColor(color)); c.setStrokeColor(HexColor(outline or color)); c.setLineWidth(linewidth)
    c.circle(x,y,r,fill=1,stroke=int(outline is not None))

def polyline(c,xy,color,width=1.8,dash=None):
    c.saveState(); c.setStrokeColor(HexColor(color)); c.setLineWidth(width)
    if dash: c.setDash(dash)
    p=c.beginPath(); p.moveTo(*xy[0])
    for x,y in xy[1:]: p.lineTo(x,y)
    c.drawPath(p); c.restoreState()

def power(c,x,y,k,align="center",size=16):
    # Center the entire typographic power, not just its base.
    base_width=pdfmetrics.stringWidth("10", FONT, size)
    exp_width=pdfmetrics.stringWidth(str(k), FONT, size*.7)
    width=base_width+exp_width
    x-=width/2 if align=="center" else width if align=="right" else 0
    text(c,x,y,"10",size); text(c,x+base_width,y+size*.45,str(k),size*.7)

def save(c,name):
    c.showPage(); c.save(); p=OUT/(name+".pdf"); OUTPUTS.append(str(p.relative_to(ROOT)))
    poppler=shutil.which("pdftoppm")
    if poppler:
        subprocess.run([poppler,"-r","160","-png","-singlefile",str(p),str(OUT/name)],check=True,capture_output=True)
        OUTPUTS.append(str((OUT/(name+".png")).relative_to(ROOT)))

def new(name,w,h):
    c=canvas.Canvas(str(OUT/(name+".pdf")),pagesize=(w,h),pageCompression=1,invariant=1)
    c.setTitle(name.replace("_"," ")); return c

def axes(c,box,xlim,ylim,xlog=False,ylog=False):
    left,bottom,w,h=box
    tx=(lambda v:math.log10(v)) if xlog else float
    ty=(lambda v:math.log10(v)) if ylog else float
    def p(x,y):
        return (left+w*(tx(x)-tx(xlim[0]))/(tx(xlim[1])-tx(xlim[0])),
                bottom+h*(ty(y)-ty(ylim[0]))/(ty(ylim[1])-ty(ylim[0])))
    line(c,left,bottom,left+w,bottom,INK,1)
    line(c,left,bottom,left,bottom+h,INK,1)
    return p

def ylabel(c,x,y,s,size=17):
    c.saveState(); c.translate(x,y); c.rotate(90);text(c,0,0,s,size,align="center");c.restoreState()

def redraw(script,number,stem,inputs,description):
    for relative in inputs:
        INPUTS[relative]=sha(ROOT/relative)
    import sys
    subprocess.run([sys.executable,str(ROOT/'reproducibility'/script)],check=True)
    for ext in ('.pdf','.png'):
        output=OUT/(stem+ext)
        if ext=='.pdf': assert output.is_file()
        if output.exists(): OUTPUTS.append(str(output.relative_to(ROOT)))
    CHECKS['Figure'+number]={'description':description,'renderer':'reproducibility/'+script}

def figure1():
    redraw('draw_lorenz_current.py','1','concentration_rank',
           ['Source_Data/Fig1_concentration/concentration_rank_plot_data.csv'],
           'Single-panel Lorenz curves for events and wallets; retained coordinates and full-array Ginis.')

def figure2():
    points=read_csv("Source_Data/Fig2_powerlaw/three_tail_plot_points.csv")
    fits={x['key']:x for x in read_csv("Source_Data/Fig2_powerlaw/three_tail_fit_summary.csv")}
    panels=[('top1000_selected','Top-1,000 event volumes','Event volume (USDC)'),
            ('leaderboard_volume','Volume leaderboard','Reported volume (USDC)'),
            ('leaderboard_pnl','PnL leaderboard','Positive PnL (USDC)')]
    c=new("three_pareto_compatible_tails",1000,480)
    c.saveState();c.translate(0,40)
    diagnostics=[]
    for j,(key,title,xlabel) in enumerate(panels):
        f=fits[key]; rows=[r for r in points if r['key']==key]
        x=np.array([float(r['x']) for r in rows]); y=np.array([float(r['empirical_survival']) for r in rows]); n=len(x)
        assert n==1000 and (np.diff(x)>=0).all()
        assert np.allclose(y,(n-np.arange(n))/n)
        alpha,xmin,nt=map(float,[f['alpha'],f['xmin'],f['n_tail']])
        assert int((x>=xmin).sum())==int(nt)
        for r in rows:
            if r['selected_tail']=='True':
                expected=(nt/n)*(float(r['x'])/xmin)**(1-alpha)
                assert math.isclose(float(r['fitted_pareto_survival']),expected,rel_tol=1e-12)
        lo,hi=math.floor(math.log10(x.min())),math.ceil(math.log10(x.max()))
        left=75+j*310; width=270; p=axes(c,(left,100,width,239),(10**lo,10**hi),(3e-4,1.2),True,True)
        text(c,left,414,f"{'abc'[j]}  {title}",18,bold=True)
        text(c,left,389,f"α = {alpha:.3f}",16)
        text(c,left,366,f"Tail n = {int(nt):,};  xmin = {xmin:,.0f}",14.7)
        for k in range(lo,hi+1):
            xp=p(10**k,.001)[0];line(c,xp,100,xp,339,"#E7EAEE",.7);power(c,xp,78,k,size=15)
        for k in [-3,-2,-1,0]:
            yp=p(x[0],10**k)[1];line(c,left,yp,left+width,yp,"#E7EAEE",.7)
            if j==0:power(c,left-9,yp-5,k,"right",15)
        cut=p(xmin,.001)[0];line(c,cut,100,cut,339,"#596A7B",1.2,[2,3])
        for xv,yv in zip(x,y):circle(c,*p(xv,yv),1.12,"#246BCE")
        grid=np.geomspace(xmin,x.max(),220)
        polyline(c,[p(v,(nt/n)*(v/xmin)**(1-alpha)) for v in grid],"#D06B32",2.4,[6,4])
        text(c,left+width/2,47,xlabel,16,align="center")
        if key=='leaderboard_volume':
            # Keep model comparison outside the data region and in the caption.
            text(c,left+width/2,23,"Exponentially truncated alternative favored",12,"#8A3C24",align="center")
        diagnostics.append({"key":key,"n":n,"tail_n":int(nt),"pointwise_survival_and_fit_verified":True})
    ylabel(c,18,220,"Survival probability P(X ≥ x)",17)
    c.restoreState()
    circle(c,140,18,2.8,'#246BCE');text(c,154,13,'Empirical observations',15)
    line(c,387,18,424,18,'#D06B32',2.4,[6,4]);text(c,436,13,'Selected Pareto tail',15)
    line(c,673,18,710,18,'#596A7B',1.2,[2,3]);text(c,722,13,'Selected cutoff',15)
    CHECKS["Figure2"]={"panels":diagnostics,"display":"All 1,000 observations per panel; no empirical connecting lines; dashed Pareto curve and dotted selected cutoff.","model_fit_unchanged":True}
    save(c,"three_pareto_compatible_tails")

def figure4():
    redraw('draw_atlas_current.py','4','01_atlas_bundled',
           ['Source_Data/Fig4_complete_groups/atlas/island_data.json','Source_Data/Fig4_complete_groups/atlas/garden_xy.json'],
           '149 amount communities; all 1,289 off-diagonal pairs; family-guided positions and curved links.')

def subject_family(s):
    x=s.lower()
    for key,f in [('politic','Politics'),('sport','Sports'),('crypto','Crypto'),('bitcoin','Crypto'),
                  ('finance','Finance'),('econom','Finance'),('commod','Finance'),('culture','Culture'),
                  ('entertain','Culture'),('meta','Meta'),('weather','Weather')]:
        if key in x:return f
    return 'Other'

SHORT=['Bitcoin-linked','Commodities','Altcoins/DeFi/stablecoins','Major crypto (non-Bitcoin)','Intraday crypto','Macro economy',
       'Entertainment/awards','Stocks/technology','Mentions/trending','Other','Other/global politics','US federal politics',
       'US state/local politics','Baseball','Basketball','Esports','Hockey','NFL/CFB','Other sports','Soccer','Tennis','Weather/climate']

def election_positions(m,a):
    names=m['network_order'];strengths=[]
    for n in names:
        v=np.asarray(m['networks']['amount'][n],float).sum(1);strengths.append(v/v.sum())
    means=np.mean(strengths,axis=0);core=set(sorted(range(22),key=lambda i:(-means[i],i))[:6])
    hierarchy=list(map(int,a['hierarchical_order']));assert sorted(hierarchy)==list(range(22))
    pos=np.zeros((22,2))
    for ids,r,phase in [([i for i in hierarchy if i not in core],.46,-math.pi/2),([i for i in hierarchy if i in core],.205,-math.pi/2+.18)]:
        for j,i in enumerate(ids):theta=phase+2*math.pi*j/len(ids);pos[i]=[.5+r*math.cos(theta),.5+r*math.sin(theta)]
    return pos

def subject_key(c,labels,top):
    text(c,24,top,"Subject key",16,bold=True)
    for i,label in enumerate(SHORT):
        col=i//8;row=i%8;x=27+329*col;y=top-25-row*20
        circle(c,x,y+4,4.4,FAMILY[subject_family(labels[i])]);text(c,x+12,y,f"{i+1}. {label}",14.5)

def election_panel(c,W,pos,labels,x0,y0,side,signed=False,k=22):
    assert W.shape==(22,22) and np.isfinite(W).all() and np.allclose(W,W.T)
    assert np.allclose(np.diag(W),0)
    candidates=[(float(W[i,j]),i,j) for i in range(22) for j in range(i+1,22)]
    selected=sorted(candidates,key=lambda e:(-abs(e[0]) if signed else -e[0],e[1],e[2]))[:k]
    selected=[e for e in selected if e[0]!=0];assert len(selected)==k
    maxe=max(abs(e[0]) for e in selected)
    strength=W.sum(1);mx=max(abs(strength)) if signed else max(strength)
    xy=pos*side+np.array([x0,y0])
    for v,i,j in reversed(selected):
        color=('#2878B5' if v>0 else '#C64747') if signed else INK
        line(c,*xy[i],*xy[j],color,.8+4.3*math.sqrt(abs(v)/maxe),alpha=.50)
    for i,(x,y) in enumerate(xy):
        # area, not radius, receives square-root compression.
        area=260+720*math.sqrt(abs(strength[i])/mx)
        outline=('#2878B5' if strength[i]>=0 else '#C64747') if signed else '#FFFFFF'
        circle(c,x,y,math.sqrt(area/math.pi),FAMILY[subject_family(labels[i])],outline,1.3)
        text(c,x,y-4.3,str(i+1),12.2,'#FFFFFF',align='center',bold=True)
    return [{'a':i,'b':j,'weight':v} for v,i,j in selected]

def elections(include_difference=True):
    m=read_json('Source_Data/Fig6_8_election/multiview.json');a=read_json('Source_Data/Fig6_8_election/analysis.json')
    labels=m['subjects'];names=m['network_order'];assert len(labels)==22 and len(names)==4
    pos=election_positions(m,a)
    c=new('election_amount_networks_journal',1000,1060)
    rec=[]
    assert set(names)=={'Harris Top 5%','Trump Top 5%','Harris Bottom 5%','Trump Bottom 5%'}
    for j,n in enumerate(['Harris Top 5%','Trump Top 5%','Harris Bottom 5%','Trump Bottom 5%']):
        x0=65+(j%2)*500;y0=680-(j//2)*380
        W=np.array(m['networks']['amount'][n],float);edges=election_panel(c,W,pos,labels,x0,y0,300)
        text(c,x0,y0+350,f"{'abcd'[j]}  {n.replace('Harris ','Harris-only ').replace('Trump ','Trump-only ')}",19,bold=True)
        mass=sum(e['weight'] for e in edges)/np.triu(W,1).sum()
        assert math.isclose(mass,m['summaries']['amount'][n]['concentration']['top22_mass'],rel_tol=1e-12)
        text(c,x0,y0+325,f"22 strongest edges; {100*mass:.1f}% of edge weight",15,GRAY)
        rec.append({'cohort':n,'display_edges':edges,'retained_mass':mass})
    subject_key(c,labels,248)
    CHECKS['Figure6']={'panels':rec,'matrix_inference':'all 231 pairs, not only displayed edges','fixed_positions':pos.tolist(),
                      'scaling':'Node area = 260+720*sqrt(s/max(s)); edge width = 0.8+4.3*sqrt(w/max(w)), separately within each panel.'}
    save(c,'election_amount_networks_journal')
    if not include_difference:
        return
    c=new('election_signed_edge_differences',1000,710)
    text(c,500,677,'Candidate-cohort differences in normalized coallocation',23,align='center',bold=True)
    line(c,250,643,283,643,'#2878B5',4);text(c,293,637,'Higher Harris-only share',16)
    line(c,560,643,593,643,'#C64747',4);text(c,603,637,'Higher Trump-only share',16)
    rec=[]
    for j,tier in enumerate(['Bottom 5%','Top 5%']):
        h,t=names[2*j:2*j+2];H=np.array(m['networks']['amount'][h],float);T=np.array(m['networks']['amount'][t],float)
        D=H/np.triu(H,1).sum()-T/np.triu(T,1).sum();assert abs(np.triu(D,1).sum())<1e-12
        x0=55+500*j;edges=election_panel(c,D,pos,labels,x0,231,320,True,30)
        dist=m['pairwise_network']['amount'][h][t]['JSD_bits']
        text(c,x0,604,f"{'ab'[j]}  {tier}",20,bold=True)
        text(c,x0,581,f"30 largest absolute differences; JSD = {dist:.3f} bits",15,GRAY)
        rec.append({'tier':tier,'display_edges':edges,'JSD_bits':dist})
    subject_key(c,labels,205)
    CHECKS['Figure7']={'panels':rec,'normalization':'each matrix divided by its upper-triangle off-diagonal sum before subtraction',
                      'display':'30 largest absolute differences; all 231 pairs used in JSD',
                      'scaling':'Node area = 260+720*sqrt(abs(delta strength)/max(abs(delta strength))); edge width = 0.8+4.3*sqrt(abs(delta)/max(abs(delta))), within panel.',
                      'fixed_positions_identical_to_Figure6':True}
    save(c,'election_signed_edge_differences')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',choices=['1','2','4','6','7'],default=['1','2','4','6','7'])
    args=parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    for number,build in [('1',figure1),('2',figure2),('4',figure4)]:
        if number in args.figures:build()
    if set(args.figures)&{'6','7'}:elections(include_difference='7' in args.figures)
    assert all(sha(ROOT/p)==h for p,h in INPUTS.items()),'An input changed during rendering'
    prior=json.loads(AUDIT.read_text(encoding='utf-8')) if AUDIT.exists() else {}
    prior_inputs=prior.get('inputs_sha256',{});prior_inputs.update(INPUTS)
    prior_checks=prior.get('checks',{});prior_checks.update(CHECKS)
    prior_outputs=prior.get('outputs_sha256',{});prior_outputs.update({p:sha(ROOT/p) for p in OUTPUTS})
    AUDIT.write_text(json.dumps({'builder':'reproducibility/build_reviewed_main_figures.py','inputs_sha256':prior_inputs,
                                'checks':prior_checks,'outputs_sha256':prior_outputs,'no_model_refit':True,
                                'no_raw_archive_scan':True,'palette':FAMILY},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'outputs':OUTPUTS,'audit':str(AUDIT)},indent=2))

if __name__=='__main__':main()
