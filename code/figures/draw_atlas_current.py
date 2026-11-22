"""Draw the published subject-guided atlas from the frozen quotient and display coordinates."""
import json, os, shutil, subprocess
from collections import Counter
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "Source_Data" / "Fig4_complete_groups" / "atlas"
out = ROOT / "figures"
out.mkdir(exist_ok=True)
data = json.loads((src / "island_data.json").read_bytes())
garden = json.loads((src / "garden_xy.json").read_bytes())
ns,es = data["nodes"],data["edges"]
col={'Crypto':'#008979','Sports':'#2864D7','Finance':'#834BCC','Politics':'#D36917','Meta':'#77633C','Culture':'#C73175'}
fg='#182E38';mute='#455B65';rule='#CCD6D9';white='#FFFFFF'
cnt=Counter(n['family'] for n in ns); assert [n['id'] for n in ns]==list(range(149))
assert cnt==dict(zip(col,(93,29,13,9,3,2)))
assert len(es)==len({tuple(sorted((e['s'],e['t']))) for e in es})==1289
assert set(range(149))-{i for e in es for i in (e['s'],e['t'])}=={43}
mx=max(e['w'] for e in es)
font_dir=Path(os.environ.get("POLYMARKET_FONT_DIR", "C:/Windows/Fonts"))
for name,file,fallback in [('Regular','segoeui.ttf','Helvetica'),('Medium','seguisb.ttf','Helvetica-Bold'),('Bold','segoeuib.ttf','Helvetica-Bold')]:
    if (font_dir/file).is_file():
        pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
    else:
        pdfmetrics.registerFont(pdfmetrics.Font(name, fallback, "WinAnsiEncoding"))
def mix(a,b,t):
    return '#'+''.join('%02x'%round(int(a[i:i+2],16)*(1-t)+int(b[i:i+2],16)*t) for i in [1,3,5])
def tt(c,x,y,s,sz=11,b='Regular',cl=fg,align='left',halo=False):
    w=pdfmetrics.stringWidth(s,b,sz); x-=w*{'left':0,'mid':.5,'right':1}[align]
    if halo:
        c.saveState();ob=c.beginText(x,y);ob.setFont(b,sz);ob.setTextRenderMode(2)
        c.setLineJoin(1);c.setLineWidth(3.3);c.setStrokeColor(HexColor(white));c.setFillColor(HexColor(cl));ob.textOut(s);c.drawText(ob);c.restoreState()
    c.setFillColor(HexColor(cl));c.setFont(b,sz);c.drawString(x,y,s)
def line(c,x,y,u,v,color=rule,lw=.60):
    c.setStrokeColor(HexColor(color));c.setLineWidth(lw);c.line(x,y,u,v)
def circle(c,x,y,r,color,border=.75):
    c.setFillColor(HexColor(color));c.setStrokeColor(HexColor(white));c.setLineWidth(border);c.circle(x,y,r,fill=1,stroke=int(bool(border)))
def lerp(a,b,t):return tuple(x+(y-x)*t for x,y in zip(a,b))
def node_color(i):return col[ns[i]['family']]
def label_color(i):return '#252920' if ns[i]['family']=='Politics' else white
def positions(mode,scale,origin):
    assert mode=='garden'
    return [(origin[0]+x*scale,origin[1]+y*scale*.95) for x,y in garden['xy']]
def mapgraph(c,mode,sc,origin,focus=False):
    p=positions(mode,sc,origin)
    def rad(i):return sc*(1.36*ns[i]['r']+.22)
    fam={n['id']:n['family'] for n in ns}
    centers={k:tuple(sum(vv)/len(vv) for vv in zip(*(p[i] for i in range(149) if i!=43 and fam[i]==k))) for k in col}
    shown=0
    # Each observed pair is drawn once. Only control points depend on the subject map.
    ordered=sorted(es,key=lambda e:e['w'])
    for e in ordered:
        i,j=e['s'],e['t'];a,b=p[i],p[j];t=(e['w']/mx)**.30;same=fam[i]==fam[j]
        special=focus and {i,j}=={5,6}
        w=sc*(.15+1.10*e['width'])
        if special: color='#154986'
        elif same:color=mix(white,node_color(i),(.23 if focus else .34)+(.52 if focus else .62)*t)
        else:color=mix(white,'#2B5366',(.072 if focus else .105)+(.51 if focus else .68)*t)
        c.setLineCap(1);c.setLineWidth(w);c.setStrokeColor(HexColor(color))
        u,v=lerp(a,b,1/3),lerp(a,b,2/3)
        if mode=='garden' and not same:
            # Light attraction of controls to each family-to-family corridor; no summed edge.
            ca,cb=centers[fam[i]],centers[fam[j]]
            u=lerp(u,lerp(ca,cb,.26),.58);v=lerp(v,lerp(ca,cb,.74),.58)
        elif mode=='garden' and same:
            cx,cy=centers[fam[i]]
            u=lerp(u,(cx,cy),.15);v=lerp(v,(cx,cy),.15)
        if special:
            # Route the one highlighted near-neighbor link into visible whitespace.
            u=(a[0]-21*sc,a[1]+4*sc);v=(b[0]-21*sc,b[1]-4*sc)
        c.bezier(*a,*u,*v,*b);shown+=1
    assert shown==1289
    for i in sorted((i for i in range(149) if i!=43),key=rad):
        circle(c,*p[i],rad(i),node_color(i),min(.82,rad(i)*.14+.1))
    for n in ns:
        if n['rank']>6:continue
        i=n['id'];x,y=p[i];sz=min(10.6,n['r']*1.05)*sc
        tt(c,x,y-sz*.35,'C'+str(i),sz,'Bold',label_color(i),'mid')
    if mode=='garden':
        for family,(x,y) in garden['labels'].items():
            tt(c,origin[0]+x*sc,origin[1]+y*sc*.95,family,12.3*sc,'Bold',mix(col[family],fg,.22),'mid',True)
    return p
def footer(c,W,second):
    line(c,24,58,W-24,58)
    for j,(k,color) in enumerate(col.items()):
        x=27+j*(W-48)/6;circle(c,x+2,41,4,color,0)
        tt(c,x+12,37,k+'  '+str(cnt[k]),10.8,'Medium')
    tt(c,25,15,second,9.6,'Regular',mute)
def header(c,W,H,t,s):
    tt(c,24,H-37,t,22,'Medium');tt(c,25,H-55,s,10.7,'Regular',mute)
    line(c,24,H-65,W-24,H-65)
def main():
    W,H = 707,550
    pdf = out/'01_atlas_bundled.pdf'
    c = Canvas(str(pdf),pagesize=(W,H),pageCompression=1,invariant=1)
    c.setTitle('Connections among 149 Polymarket communities')
    c.setAuthor('Polymarket study | figure alternatives')
    header(c,W,H,'Connections among 149 Polymarket communities','Each node is a detected wallet–event community. Subjects determine the colors and the arrangement.')
    mapgraph(c,'garden',1,(22,58))
    circle(c,589,H-52,1.36*ns[43]['r']+.22,col['Crypto'])
    tt(c,598,H-48,'C43 · no',9.8,'Medium',mute)
    tt(c,598,H-59,'inter-community links',9.8,'Medium',mute)
    footer(c,W,'Node size: volume of member events.  Line width: participation across communities (USDC).  All 1,289 links are retained.')
    c.showPage(); c.save()
    if converter:=shutil.which('pdftoppm'):
        subprocess.run([converter,'-r','160','-png','-singlefile',str(pdf),str(pdf.with_suffix(''))],check=True)
if __name__=='__main__':main()
