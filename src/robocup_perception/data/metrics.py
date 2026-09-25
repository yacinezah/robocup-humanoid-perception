"""Task metrics in a common canonical 640-coordinate system; no data access."""
import numpy as np
from scipy import ndimage
def field_metrics(p,y):
    p=p>=.5;y=y.astype(bool);tp=(p&y).sum();fp=(p&~y).sum();fn=(~p&y).sum()
    def boundary(m):return m^ndimage.binary_erosion(m,structure=np.ones((3,3)),border_value=0)
    a,b=boundary(p),boundary(y);structure=np.ones((3,3))
    pr=(a&ndimage.binary_dilation(b,structure=structure,iterations=3)).sum()/max(a.sum(),1)
    re=(b&ndimage.binary_dilation(a,structure=structure,iterations=3)).sum()/max(b.sum(),1)
    border=np.zeros_like(y);border[:16]=border[-16:]=True;border[:,:16]=border[:,-16:]=True
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'pixels':int(y.size),'dice':float(2*tp/max(2*tp+fp+fn,1)),'bf1':float(2*pr*re/max(pr+re,1e-12)),'false_field':float(fp/y.size),'border_recall':float((p&y&border).sum()/max((y&border).sum(),1))}
def aggregate_field(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    return {'dice':2*tp/max(2*tp+fp+fn,1),'bf1':float(np.mean([r['bf1'] for r in rows])),'false_field':fp/max(sum(r['pixels'] for r in rows),1),'border_recall':float(np.mean([r['border_recall'] for r in rows]))}
def ious(a,b):
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    inter=np.maximum(np.minimum(a[:,None,2:4],b[None,:,2:4])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
    area=lambda x:np.maximum(x[:,2:4]-x[:,:2],0).prod(1)
    return inter/np.maximum(area(a)[:,None]+area(b)[None,:]-inter,1e-12)
def detection_metrics(records,thresholds=None):
    thresholds=thresholds or [.25]*6;all_ap=[];classes=[]
    for cid in range(6):
        scores=[];matches=[];ngt=0;point_tp=point_fp=point_fn=tp=fp=fn=0;errors=[]
        for r in records:
            pred=np.array(r['pred'],np.float32).reshape(-1,6);gt=np.array(r['gt'],np.float32).reshape(-1,5)
            p=pred[pred[:,5]==cid];g=gt[gt[:,4]==cid];p=p[np.argsort(-p[:,4],kind='stable')];ngt+=len(g)
            ov=ious(p,g);correct=np.zeros((len(p),10),bool)
            for t,thr in enumerate(np.arange(.5,.951,.05)):
                used=set()
                for i in range(len(p)):
                    options=[j for j in np.argsort(-ov[i]) if j not in used and ov[i,j]>=thr]
                    if options:correct[i,t]=True;used.add(options[0])
            scores.extend(p[:,4]);matches.extend(correct.tolist());selected=p[:,4]>=thresholds[cid]
            nt=int(correct[selected,0].sum());tp+=nt;fp+=int(selected.sum())-nt;fn+=len(g)-nt
            if cid>=3:
                pp=p[selected];d=np.linalg.norm((pp[:,None,:2]+pp[:,None,2:4])/2-(g[None,:,:2]+g[None,:,2:4])/2,axis=2)
                used=set();n=0
                for i in range(len(pp)):
                    options=[j for j in np.argsort(d[i]) if j not in used and d[i,j]<=10]
                    if options:used.add(options[0]);n+=1;errors.append(float(d[i,options[0]]))
                point_tp+=n;point_fp+=len(pp)-n;point_fn+=len(g)-n
        ap=[]
        if ngt and len(scores):
            order=np.argsort(-np.array(scores),kind='stable');c=np.array(matches)[order];tpc=c.cumsum(0);fpc=(~c).cumsum(0)
            for k in range(10):
                recall=tpc[:,k]/ngt;precision=tpc[:,k]/(tpc[:,k]+fpc[:,k]);mr=np.r_[0,recall,1];mp=np.r_[1,precision,0];mp=np.maximum.accumulate(mp[::-1])[::-1]
                ap.append(float(np.trapezoid(np.interp(np.linspace(0,1,101),mr,mp),np.linspace(0,1,101))))
        else:ap=[0.]*10
        if ngt:all_ap.extend(ap)
        classes.append({'class':cid,'gt':ngt,'ap50':ap[0],'map':float(np.mean(ap)),'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/max(2*tp+fp+fn,1),'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'point_tp':point_tp,'point_fp':point_fp,'point_fn':point_fn,'point_f1':2*point_tp/max(2*point_tp+point_fp+point_fn,1),'center_p95':float(np.percentile(errors,95)) if errors else None})
    return {'map':float(np.mean(all_ap)) if all_ap else 0,'classes':classes,'lt_f1':float(np.mean([classes[c]['point_f1'] for c in [3,4]]))}
