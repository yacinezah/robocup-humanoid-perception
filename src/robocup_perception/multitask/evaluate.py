import json,pathlib,time
import cv2,numpy as np,torch
from torchvision.ops import batched_nms
from .data import HERE,read_image,tensor,letterbox,field_ground_truth,restore_field,load_labels
from robocup_perception.data.metrics import field_metrics,aggregate_field,detection_metrics
def decoded(pred):
    # Native YOLOv8 inference: xywh + six probabilities; class-aware NMS is external.
    if isinstance(pred,tuple):pred=pred[0]
    x=pred[0].transpose(0,1);conf,cls=x[:,4:].max(1);keep=conf>=.001;x=x[keep];conf=conf[keep];cls=cls[keep]
    boxes=torch.cat([x[:,:2]-x[:,2:4]/2,x[:,:2]+x[:,2:4]/2],1)
    idx=batched_nms(boxes,conf,cls,.7)[:300]
    return torch.cat([boxes[idx],conf[idx,None],cls[idx,None]],1).float().cpu().numpy()
@torch.inference_mode()
def evaluate_field(model,rows,device):
    model.eval();records=[]
    for row in rows:
        im=read_image(row);x=tensor(letterbox(im)[0])[None].to(device)
        prob=model(x,task='field')[0,0].float().sigmoid().cpu().numpy()
        prob=restore_field(prob,letterbox(im,320)[1]);r=field_metrics(prob,field_ground_truth(row));r.update(id=row['id'],group=row['group']);records.append(r)
    return aggregate_field(records),records
@torch.inference_mode()
def evaluate_detector(model,rows,device):
    model.eval();records=[]
    for row in rows:
        im=read_image(row);image,geo=letterbox(im);s,l,t,_,_,w,h=geo;x=tensor(image)[None].to(device)
        pred=decoded(model(x,task='detection'))
        if len(pred):
            pred[:,[0,2]]=(pred[:,[0,2]]-l)/s;pred[:,[1,3]]=(pred[:,[1,3]]-t)/s
            pred[:,[0,2]]=pred[:,[0,2]].clip(0,w);pred[:,[1,3]]=pred[:,[1,3]].clip(0,h)
            pred[:,:4]*=640/w
        lab=load_labels(row);g=np.zeros((len(lab),5),np.float32)
        if len(lab):
            g[:,0]=(lab[:,1]-lab[:,3]/2)*640;g[:,2]=(lab[:,1]+lab[:,3]/2)*640
            g[:,1]=(lab[:,2]-lab[:,4]/2)*h*640/w;g[:,3]=(lab[:,2]+lab[:,4]/2)*h*640/w;g[:,4]=lab[:,0]
        records.append({'id':row['id'],'group':row['group'],'domain':row['domain'],'pred':pred.tolist(),'gt':g.tolist()})
    return detection_metrics(records),records
def field_reference(tag,rows):
    import openvino as ov
    compiled=ov.Core().compile_model(str(HERE/f'references/{tag}.xml'),'CPU',{'INFERENCE_NUM_THREADS':'4','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f32'})
    output=[o for o in compiled.outputs if list(o.shape)==[1,1,320,320]][0]
    records=[]
    for row in rows:
        im=cv2.resize(read_image(row),(320,320));x=im.astype(np.float32)/255
        x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        logits=compiled([np.ascontiguousarray(x.transpose(2,0,1)[None])])[output][0,0]
        p=1/(1+np.exp(-np.clip(logits,-80,80)));r=field_metrics(p,field_ground_truth(row));r.update(id=row['id'],group=row['group']);records.append(r)
    return aggregate_field(records),records
def teacher_cache():
    import openvino as ov
    from .data import cache_name
    rows=json.loads((HERE/'manifest.json').read_text())['field_train'];out=HERE/'teacher_cache';out.mkdir(exist_ok=True)
    compiled=ov.Core().compile_model(str(HERE/'references/mobile.xml'),'CPU',{'INFERENCE_NUM_THREADS':'4','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f32'})
    output=[o for o in compiled.outputs if list(o.shape)==[1,1,320,320]][0]
    for i,row in enumerate(rows):
        path=out/cache_name(row)
        if path.exists():continue
        im=cv2.resize(read_image(row),(320,320));x=(im.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        logits=compiled([np.ascontiguousarray(x.transpose(2,0,1)[None])])[output][0,0]
        p=(1/(1+np.exp(-np.clip(logits,-80,80)))).astype(np.float16)
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.save(f,p)
        tmp.replace(path)
        if i%250==0:print('teacher_cache',i,len(rows),flush=True)
    (out/'receipt.json').write_text(json.dumps({'rows':len(rows),'teacher':'binary-field-quality OpenVINO FP32','supervised_train_only':True,'native_preprocessing':'direct 320 RGB ImageNet','cache_dtype':'float16 probability; regularization only'},indent=2))
# Teacher-cache generation is an explicit Python function, never an import side effect.
