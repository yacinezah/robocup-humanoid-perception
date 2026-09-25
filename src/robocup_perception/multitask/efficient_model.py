"""Efficient frozen-sharing field-context family, efficient frozen six-class detector, wider taps."""
import torch
from torch import nn
from ultralytics import YOLO
from pathlib import Path
from .model import MultiTask,Field,Conv
class EfficientMTL(MultiTask):
    def __init__(self,checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError('An existing local architecture YAML or trusted checkpoint is required')
        nn.Module.__init__(self);self.detector=YOLO(str(checkpoint)).model.float()
        h=self.detector.model[-1];assert h.nc==6 and h.end2end and h.reg_max==1
        self.field=Field();self.field.entry=Conv(128,32,1)
        self.field.lat[2]=Conv(128,32,1);self.field.lat[3]=Conv(64,16,1)
def raw_efficient(pred):
    if isinstance(pred,tuple):pred=pred[1]
    p=pred['one2one'];return torch.cat([p['boxes'],p['scores']],1)
class EfficientExport(nn.Module):
    def __init__(self,m,native=False):super().__init__();self.m=m;self.native=native
    def forward(self,x):
        d,f=self.m(x);return (d[0] if self.native else raw_efficient(d)),f
