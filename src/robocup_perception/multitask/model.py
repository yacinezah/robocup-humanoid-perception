"""Static selective sharing: native YOLOv8 through stride 8, private field semantics."""
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from pathlib import Path

class Conv(nn.Sequential):
    def __init__(self,ci,co,k=3,s=1,g=1):
        super().__init__(nn.Conv2d(ci,co,k,s,k//2,groups=g,bias=False),nn.BatchNorm2d(co),nn.SiLU())
class DS(nn.Sequential):
    def __init__(self,ci,co,s=1):super().__init__(Conv(ci,ci,3,s,ci),Conv(ci,co,1))
class MB(nn.Module):
    def __init__(self,ci,co,s):
        super().__init__(); self.net=nn.Sequential(Conv(ci,ci*2,1),Conv(ci*2,ci*2,3,s,ci*2),nn.Conv2d(ci*2,co,1,bias=False),nn.BatchNorm2d(co));self.skip=ci==co and s==1
    def forward(self,x):return self.net(x)+x if self.skip else self.net(x)
class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry=Conv(64,32,1)
        self.down=nn.ModuleList([nn.Sequential(MB(32,48,2),MB(48,48,1)),nn.Sequential(MB(48,96,2),MB(96,96,1)),nn.Sequential(MB(96,128,2),MB(128,128,1))])
        self.lat=nn.ModuleList([Conv(96,64,1),Conv(48,48,1),Conv(64,32,1),Conv(32,16,1)])
        self.up=nn.ModuleList([Conv(128,64,1),Conv(64,48,1),Conv(48,32,1),Conv(32,16,1)])
        self.refine=nn.ModuleList([nn.Sequential(DS(c,c),DS(c,c)) for c in [64,48,32,16]])
        self.out=nn.Conv2d(16,1,1)
    def forward(self,s4,s8):
        x=self.entry(s8); cs=[]
        for block in self.down:x=block(x);cs.append(x)
        for lateral,project,refine,skip in zip(self.lat,self.up,self.refine,[cs[1],cs[0],s8,s4]):
            x=refine(F.interpolate(project(x),size=skip.shape[-2:],mode='bilinear',align_corners=False)+lateral(skip))
        return F.interpolate(self.out(x),size=(320,320),mode='bilinear',align_corners=False)
class MultiTask(nn.Module):
    def __init__(self,checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError('An existing local architecture YAML or trusted checkpoint is required')
        super().__init__(); self.detector=YOLO(str(checkpoint)).model.float();self.field=Field()
        assert self.detector.model[-1].nc==6 and not self.detector.model[-1].end2end
    def forward(self,x,task='both'):
        ys=[];s4=s8=None; field=None
        for m in self.detector.model:
            if m.f!=-1:x=ys[m.f] if isinstance(m.f,int) else [x if j==-1 else ys[j] for j in m.f]
            x=m(x);ys.append(x)
            if m.i==2:s4=x
            if m.i==4:
                s8=x
                if task=='field':return self.field(s4,s8)
        if task=='detection':return x
        return x,self.field(s4,s8)
    def configure(self,arm):
        for p in self.parameters():p.requires_grad_(False)
        if arm!='S-D':
            for p in self.field.parameters():p.requires_grad_(True)
        if arm in ['J','J-KD','S-D']:
            for p in self.detector.parameters():p.requires_grad_(True)
        elif arm=='S-F':
            for m in list(self.detector.model)[:5]:
                for p in m.parameters():p.requires_grad_(True)
        self.arm=arm
    def train(self,mode=True):
        super().train(mode)
        # Batch composition must not rewrite pretrained detector running statistics.
        for m in self.detector.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self

def raw(pred):
    if isinstance(pred,tuple):pred=pred[1]
    return torch.cat([pred['boxes'],pred['scores']],1)
class Export(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):
        d,f=self.model(x)
        return raw(d),f

def field_loss(logits,target,valid,teacher=None):
    b=F.max_pool2d(target,7,1,3)+F.max_pool2d(-target,7,1,3)
    w=(1+3*b)*valid;p=logits.sigmoid()
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*w).sum()/w.sum().clamp_min(1)
    dims=(1,2,3);dice=(1-(2*(p*target*valid).sum(dims)+1)/((p*valid).sum(dims)+(target*valid).sum(dims)+1)).mean()
    kd=logits.sum()*0
    if teacher is not None:
        q=teacher.clamp(1e-5,1-1e-5)
        kd=((F.binary_cross_entropy_with_logits(logits,q,reduction='none')+q*q.log()+(1-q)*(1-q).log())*valid).sum()/valid.sum().clamp_min(1)
    return bce+dice+0.25*kd,{'bce':bce.detach(),'dice_loss':dice.detach(),'kd':kd.detach()}

def detector_kd(student,teacher):
    if isinstance(student,tuple):student=student[1]
    if isinstance(teacher,tuple):teacher=teacher[1]
    # BF16 rounds 1-1e-6 to 1, causing 0*log(0). KL arithmetic must be FP32.
    q=teacher['scores'].detach().float().sigmoid().clamp(1e-6,1-1e-6)
    w=q.amax(1,keepdim=True).clamp_min(.01)
    cls=F.binary_cross_entropy_with_logits(student['scores'].float(),q,reduction='none')+q*q.log()+(1-q)*(1-q).log()
    cls=(cls*w).sum()/(w.sum()*6)
    bs=student['boxes'].shape[0]
    a=student['boxes'].float().reshape(bs,4,16,-1).log_softmax(2)
    b=teacher['boxes'].detach().float().reshape(bs,4,16,-1).softmax(2)
    box=(F.kl_div(a,b,reduction='none').sum(2)*w).sum()/(w.sum()*4)
    return cls+box
