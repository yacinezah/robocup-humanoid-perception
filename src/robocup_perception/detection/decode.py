"""Deterministic NumPy/OpenCV decoder for stable raw heads; no training runtime."""
import numpy as np,cv2
AX=[];ST=[]
for stride in [8,16,32]:
    side=640//stride;y,x=np.meshgrid(np.arange(side),np.arange(side),indexing='ij');AX.append(np.stack([x.ravel()+.5,y.ravel()+.5],axis=0));ST.append(np.full(side*side,stride))
ANCHORS=np.concatenate(AX,1).astype(np.float32);STRIDES=np.concatenate(ST).astype(np.float32)
def decode_raw(array,conf=.001,iou=.7):
    raw=np.asarray(array,dtype=np.float32)[0]
    if raw.shape==(300,6):d=raw.copy()
    else:
        if raw.shape[0]==70:
            z=raw[:64].reshape(4,16,-1);z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);dist=(p*np.arange(16,dtype=np.float32)[None,:,None]).sum(1);scores=1/(1+np.exp(-np.clip(raw[64:],-80,80)))
        elif raw.shape[0]==10:
            dist=raw[:4];scores=1/(1+np.exp(-np.clip(raw[4:],-80,80)))
        else:raise ValueError(('Unsupported raw contract',raw.shape))
        boxes=np.concatenate([ANCHORS-dist[:2],ANCHORS+dist[2:]],0)*STRIDES[None]
        if raw.shape[0]==70:
            ids=scores.argmax(0);confidence=scores[ids,np.arange(scores.shape[1])];d=np.column_stack([boxes.T,confidence,ids])
        else:
            # Match the native Y26 two-stage top-K, including distinct class rows
            # for a shared grid cell. Do not silently convert to class argmax.
            cells=np.argsort(-scores.max(0),kind='stable')[:300];flat=scores[:,cells].T.reshape(-1)
            idx=np.argsort(-flat,kind='stable')[:300];d=np.column_stack([boxes[:,cells[idx//6]].T,flat[idx],idx%6])
    d=d[np.isfinite(d).all(1)&(d[:,4]>=conf)]
    if not len(d):return np.empty((0,6),np.float32)
    xywh=d[:,:4].copy();xywh[:,2:]-=xywh[:,:2]
    idx=cv2.dnn.NMSBoxesBatched(xywh.tolist(),d[:,4].tolist(),d[:,5].astype(int).tolist(),conf,iou)
    return d[np.asarray(idx,dtype=int).reshape(-1)[:300]].astype(np.float32)
