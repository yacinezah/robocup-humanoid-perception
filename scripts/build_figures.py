"""Regenerate six publication figures from frozen aggregate tables only."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'results/tables'
OUT = ROOT / 'results/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
NAVY, TEAL, ORANGE, GRAY = '#17324D', '#087F8C', '#C9652C', '#657582'
provenance = []


def rows(stem):
    return list(csv.DictReader((TABLES/(stem+'-1.csv')).open(encoding='utf-8')))


def save(fig, name, sources, note):
    fig.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight', facecolor='white')
    provenance.append({'figure': name, 'sources': sources, 'interpretation': note,
                       'png_sha256': hashlib.sha256((OUT/(name+'.png')).read_bytes()).hexdigest()})
    plt.close(fig)


# Static technical diagram, not an image of robot hardware.
fig, ax = plt.subplots(figsize=(12, 4)); ax.set(xlim=(0,12), ylim=(0,4)); ax.axis('off')
boxes = [(0.15,1.4,2.3,1.1,'Images + metadata\ntrusted labels / exposure'),
         (3,2.45,2.9,1,'Neural perception\nfield + six classes'),
         (3,.55,2.9,1,'CPU execution\nONNX / OpenVINO'),
         (7,1.4,2.9,1.1,'Robot observations\nprojection / association')]
from matplotlib.patches import FancyBboxPatch
for x,y,w,h,label in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',fc='#F0F5F8',ec=NAVY,lw=1.2))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=NAVY)
for a,b in [((2.6,2.1),(2.85,2.9)),((2.6,1.7),(2.85,1.1)),((6.1,2.9),(6.8,2.1)),((6.1,1.1),(6.8,1.7))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':TEAL,'lw':1.8})
ax.text(11.25,2.0,'Pose\nSE(2)',ha='center',va='center',fontsize=15,color=NAVY)
ax.annotate('',xy=(10.7,2),xytext=(10.05,2),arrowprops={'arrowstyle':'->','color':TEAL})
ax.text(.1,3.85,'From reliable evaluation to probabilistic robotics',fontsize=16,weight='bold',color=NAVY)
ax.text(.1,.05,'Contributions: data audits, controlled architectures, CPU tools, replay and likelihood/association studies.',color=GRAY)
save(fig,'system-overview',['docs/reports/index.md'],'Conceptual system diagram; inherited robot/frameworks credited in README.')

r=rows('binary-field-segmentation'); t=rows('intel-nuc-segmentation-benchmark')
fig, axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for i,c in enumerate([TEAL,NAVY]):
    x=float(t[i]['Four-thread p95 ms'])
    for ax,key in zip(axs,['Dice','Boundary F1@3']):
        ax.scatter(x,float(r[i][key]),color=c,s=100)
        ax.annotate(['Fast-SCNN','MobileNet'][i],(x,float(r[i][key])),xytext=(7,9 if i==0 else -15),textcoords='offset points')
        ax.set(xlabel='Actual Intel NUC four-thread p95 (ms)',ylabel=key,xlim=(9,35))
        ax.grid(alpha=.18)
fig.suptitle('Binary field quality and CPU cost',color=NAVY,weight='bold')
save(fig,'segmentation-tradeoff',['binary-field-segmentation-1.csv','intel-nuc-segmentation-benchmark-1.csv'],
     'Open 77-image field bridge and actual NUC timing; explicit-line task excluded from this ranking.')

r=rows('controlled-yolov8n-yolo26n-comparison'); t=rows('intel-nuc-detector-benchmark')
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
axs[0].bar(['YOLOv8n-640','YOLO26n-640'],[float(r[0]['YOLOv8n-640']),float(r[0]['YOLO26n-640'])],color=[NAVY,TEAL])
axs[0].set(ylabel='Three-seed mAP50-95',ylim=(0,.62),title='Controlled sealed-test accuracy')
for i,key in enumerate(['YOLOv8n-640','YOLO26n-640']):axs[0].text(i,float(r[0][key])+.015,f'{float(r[0][key]):.4f}',ha='center')
v=[float(x['Four-thread p95 ms']) for x in t]
axs[1].barh(['YOLOv8n-640','YOLO26n-640','Production reference','Historical 320'],v,color=[NAVY,TEAL,'white','white'],edgecolor=[NAVY,TEAL,GRAY,GRAY])
axs[1].invert_yaxis();axs[1].set(xlabel='Four-thread p95 latency (ms)',xlim=(0,198),title='Actual Intel NUC')
for i,val in enumerate(v):axs[1].text(val+3,i,f'{val:.1f}',va='center')
axs[1].axvline(50,color=ORANGE,ls='--',lw=1)
save(fig,'detector-comparison',['controlled-yolov8n-yolo26n-comparison-1.csv','intel-nuc-detector-benchmark-1.csv'],
     'Accuracy comparison only for controlled pair. Hollow runtime anchors have incompatible accuracy protocols. Dashed line: 50 ms.')

r=rows('selective-sharing-multitask-perception')
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
axs[0].axis('off')
axs[0].text(0,1,'Selective sharing',fontsize=16,color=NAVY,weight='bold',va='top')
axs[0].text(0,.82,'640 RGB\nShared stride-4/8 detail\n\nPrivate detector context + head\nPrivate field context + decoder\n\n6 classes + 320-pixel field logits',fontsize=13,linespacing=1.65,va='top')
v=[float(x['Dice']) for x in r]
axs[1].bar(['Efficient\nfrozen + KD','Joint\nsharing + KD','Separate\nspecialists'],v,color=[TEAL,NAVY,GRAY])
axs[1].set(ylim=(.975,1.002),ylabel='Field Dice (truncated axis)',title='Matched open development evidence')
for i,val in enumerate(v):axs[1].text(i,val+.0007,f'{val:.4f}',ha='center')
save(fig,'multitask-sharing',['selective-sharing-multitask-perception-1.csv','docs/methods/multitask-contract.md'],
     'Selected seed, teacher/history-exposed field bridge. Confirmation variability prevents recipe-level noninferiority.')

r=rows('typed-landmark-localization')[:4]
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
labels=['Baseline','Exact\noracle L/T','Actual measures\n+ oracle ID','Learned\nassociation']
for ax,key,title in zip(axs,['Translation median/p95 m','Yaw median/p95 deg'],['Translation error (m)','Wrapped yaw error (deg)']):
    v=np.array([[float(y) for y in x[key].split('/')] for x in r]);xx=np.arange(4)
    ax.bar(xx-.17,v[:,0],.34,label='Median',color=TEAL);ax.bar(xx+.17,v[:,1],.34,label='p95',color=NAVY)
    ax.set_xticks(xx,labels);ax.set_title(title);ax.legend(frameon=False)
fig.suptitle('Useful measurements; difficult map association',color=NAVY,weight='bold')
save(fig,'localization-causal',['typed-landmark-localization-1.csv'],
     'Gazebo with synthetic command odometry. Oracle identity is diagnostic. Later map/plane audit qualifies an association-only diagnosis.')

r=rows('sensor-aided-localization');t=rows('measured-line-localization')
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,data,labels,title in [(axs[0],[r[0],r[2]],['Same-sensor\nbaseline','Robust L/T\nmixture'],'Fresh sensor-aided suite'),
                           (axs[1],t[:2],['Goalposts\n+ L/T','Goalposts\n+ lines'],'Reused development line suite')]:
    v=np.array([[float(y) for y in x['Translation median/p95 m'].split('/')] for x in data]);xx=np.arange(2)
    ax.bar(xx-.18,v[:,0],.36,color=TEAL,label='Median');ax.bar(xx+.18,v[:,1],.36,color=NAVY,label='p95')
    ax.set_xticks(xx,labels);ax.set(ylabel='Translation error (m)',title=title);ax.legend(frameon=False)
save(fig,'tracking-and-lines',['sensor-aided-localization-1.csv','measured-line-localization-1.csv'],
     'Separate protocols and y scales. Synthetic heading, no real-robot pose accuracy. Line result is development-only.')
(ROOT/'reproducibility/figure-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(f'Generated {len(provenance)} figures from frozen aggregates.')
