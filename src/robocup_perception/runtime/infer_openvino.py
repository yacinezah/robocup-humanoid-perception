"""Minimal CPU-only inference for released selective-sharing raw-head graphs.

Example: python infer_openvino.py --model multitask.xml --image frame.jpg
No field rejection, tracker, map association or production integration is applied.
"""
import argparse,json,pathlib
import cv2,numpy as np,openvino as ov
from robocup_perception.detection.decode import decode_raw
CLASSES=['ball','goalpost','robot','L','T','X']
def letterbox(im,size):
    h,w=im.shape[:2];s=min(size/w,size/h);rw,rh=round(w*s),round(h*s);l,t=(size-rw)//2,(size-rh)//2
    x=np.full((size,size,3),114,np.uint8);x[t:t+rh,l:l+rw]=cv2.resize(im,(rw,rh));return x,(s,l,t,rw,rh)
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--image',required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--thresholds',help='Calibrated JSON; otherwise common .25 detector confidence');p.add_argument('--output',default='prediction.json');a=p.parse_args()
    bgr=cv2.imread(a.image);assert bgr is not None,a.image;im=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=im.shape[:2];x,geo=letterbox(im,640)
    compiled=ov.Core().compile_model(a.model,'CPU',{'INFERENCE_NUM_THREADS':str(a.threads),'NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f32'});z=compiled([np.ascontiguousarray(x.transpose(2,0,1)[None].astype(np.float32)/255)])
    d=next(o for o in compiled.outputs if len(o.shape)==3);f=next(o for o in compiled.outputs if len(o.shape)==4);pred=decode_raw(z[d]);ts=json.loads(pathlib.Path(a.thresholds).read_text())['thresholds'] if a.thresholds else [.25]*6
    pred=pred[np.array([row[4]>=ts[int(row[5])] for row in pred],bool)];s,l,t,_,_=geo
    if len(pred):
        pred[:,[0,2]]=(pred[:,[0,2]]-l)/s;pred[:,[1,3]]=(pred[:,[1,3]]-t)/s;pred[:,[0,2]]=pred[:,[0,2]].clip(0,w);pred[:,[1,3]]=pred[:,[1,3]].clip(0,h)
    prob=1/(1+np.exp(-np.clip(z[f][0,0],-80,80)));_,(_,l,t,rw,rh)=letterbox(im,320);prob=cv2.resize(prob[t:t+rh,l:l+rw],(w,h));out=pathlib.Path(a.output);np.save(out.with_suffix('.field_probability.npy'),prob)
    result={'class_order':CLASSES,'source_shape':[h,w],'detector_thresholds':ts,'nms_iou':.7,'detections':[{'xyxy':r[:4].tolist(),'confidence':float(r[4]),'class':CLASSES[int(r[5])],'center':((r[:2]+r[2:4])/2).tolist()} for r in pred],'field_probability':str(out.with_suffix('.field_probability.npy')),'field_threshold':.5,'warning':'Research development artifact. Strict raw parity not passed; no independent-generalization or production approval. X is not enabled for localization.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
