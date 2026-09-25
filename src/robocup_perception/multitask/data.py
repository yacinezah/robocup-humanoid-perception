import json,pathlib,hashlib,os
import cv2,numpy as np,torch
from torch.utils.data import Dataset
HERE=pathlib.Path(os.environ.get('PERCEPTION_WORKROOT', '.')).resolve()
def read_image(row):
    im=cv2.imread(str(HERE/row['image']));assert im is not None,row['id']
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
def letterbox(im,size=640,fill=114,interp=cv2.INTER_LINEAR):
    h,w=im.shape[:2];scale=min(size/w,size/h);rw,rh=round(w*scale),round(h*scale)
    l,t=(size-rw)//2,(size-rh)//2
    x=cv2.resize(im,(rw,rh),interpolation=interp)
    shape=(size,size)+im.shape[2:];out=np.full(shape,fill,dtype=im.dtype);out[t:t+rh,l:l+rw]=x
    return out,(scale,l,t,rw,rh,w,h)
def load_labels(row):
    txt=(HERE/row['target']).read_text().strip()
    return np.array([list(map(float,line.split())) for line in txt.splitlines()],np.float32).reshape(-1,5)
def tensor(im):return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float()/255
def cache_name(row):return hashlib.sha256(row['image'].encode()).hexdigest()+'.npy'
class Samples(Dataset):
    def __init__(self,rows,seed,offset,count,task,kd=False):self.rows=rows;self.seed=seed;self.offset=offset;self.count=count;self.task=task;self.kd=kd
    def __len__(self):return self.count
    def __getitem__(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.offset+index,0 if self.task=='field' else 1]))
        row=self.rows[int(rng.integers(len(self.rows)))];im=read_image(row);h,w=im.shape[:2]
        flip=rng.random()<.5
        if self.task=='field':
            mask=cv2.imread(str(HERE/row['target']),0);assert mask is not None
            target=(mask>25).astype(np.float32)
            assert target.shape==(h,w)
            teacher=None
            if self.kd:
                teacher=np.load(HERE/'teacher_cache'/cache_name(row)).astype(np.float32)
                teacher=cv2.resize(teacher,(w,h),interpolation=cv2.INTER_LINEAR)
            if flip:
                im=im[:,::-1];target=target[:,::-1]
                if teacher is not None:teacher=teacher[:,::-1]
            target,geo=letterbox(target,320,0,cv2.INTER_NEAREST)
            valid=np.zeros((320,320),np.float32);_,l,t,rw,rh,_,_=geo;valid[t:t+rh,l:l+rw]=1
            q=np.zeros((320,320),np.float32) if teacher is None else letterbox(teacher,320,0)[0]
            extra={'target':torch.from_numpy(target[None]),'valid':torch.from_numpy(valid[None]),'teacher':torch.from_numpy(q[None])}
        else:
            y=load_labels(row)
            if flip:im=im[:,::-1];y[:,1]=1-y[:,1]
            _,geo=letterbox(im);s,l,t,_,_,_,_=geo
            if len(y):y[:,1]=(y[:,1]*w*s+l)/640;y[:,2]=(y[:,2]*h*s+t)/640;y[:,3]*=w*s/640;y[:,4]*=h*s/640
            extra={'labels':torch.from_numpy(y)}
        contrast=rng.uniform(.9,1.1);bright=rng.uniform(-12,12)
        im=np.clip((im.astype(np.float32)-127.5)*contrast+127.5+bright,0,255).astype(np.uint8)
        extra['img']=tensor(letterbox(im)[0]);return extra
def collate(samples):
    out={'img':torch.stack([s['img'] for s in samples])}
    if 'labels' in samples[0]:
        labels=torch.cat([s['labels'] for s in samples]);out['cls']=labels[:,:1];out['bboxes']=labels[:,1:]
        out['batch_idx']=torch.cat([torch.full((len(s['labels']),),i,dtype=torch.float32) for i,s in enumerate(samples)])
    else:
        for k in ['target','valid','teacher']:out[k]=torch.stack([s[k] for s in samples])
    return out

def field_ground_truth(row):
    m=cv2.imread(str(HERE/row['target']),0);assert m is not None
    return cv2.resize((m>25).astype(np.uint8),(320,320),interpolation=cv2.INTER_NEAREST)
def restore_field(prob,geo):
    _,l,t,rw,rh,_,_=geo
    # geo must be 320-letterbox metadata, not blindly halved rounded 640 geometry.
    return cv2.resize(prob[t:t+rh,l:l+rw],(320,320),interpolation=cv2.INTER_LINEAR)
