"""Explicit CPU re-export of a trusted selected state dictionary. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    import torch
    import openvino as ov
    from robocup_perception.multitask.model import MultiTask, Export
    from robocup_perception.multitask.efficient_model import EfficientMTL, EfficientExport
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architecture',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--family',required=True,choices=['joint','efficient'])
    p.add_argument('--native',action='store_true')
    p.add_argument('--output-directory',required=True,type=Path)
    a=p.parse_args()
    if a.native and a.family!='efficient':p.error('--native requires efficient family')
    if a.output_directory.exists() and any(a.output_directory.iterdir()):p.error('Use a new empty output directory')
    torch.set_num_threads(1);torch.manual_seed(0)
    m=(EfficientMTL if a.family=='efficient' else MultiTask)(a.architecture)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    m.load_state_dict(checkpoint['model'],strict=True);m.eval()
    # Archived export fuses detector Conv/BN. This creates new export bytes, not
    # a replacement for a frozen model; numerical status is measured separately.
    m.detector.fuse(verbose=False)
    m.detector.model[-1].export=False
    wrapped=EfficientExport(m,native=a.native) if a.family=='efficient' else Export(m)
    wrapped.eval();x=torch.rand(1,3,640,640)
    a.output_directory.mkdir(parents=True,exist_ok=True)
    target=a.output_directory/'multitask.onnx'
    torch.onnx.export(wrapped,x,str(target),opset_version=17,input_names=['images'],
                      output_names=['detections','field_logits'],dynamic_axes=None)
    model=ov.convert_model(str(target));ov.save_model(model,str(target.with_suffix('.xml')),compress_to_fp16=False)
    compiled=ov.Core().compile_model(model,'CPU',{'INFERENCE_NUM_THREADS':'1','INFERENCE_PRECISION_HINT':'f32'})
    with torch.no_grad():expected=wrapped(x)
    actual=compiled([x.numpy()])
    errors=[float(np.max(np.abs(t.numpy()-actual[o]))) for t,o in zip(expected,compiled.outputs)]
    result={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'random_fixture_raw_maxima':errors,'strict_tolerance':1e-4,
            'single_fixture_pass':all(v<=1e-4 for v in errors),
            'warning':'New technical export only. One random fixture is not historical parity certification.'}
    (a.output_directory/'export-smoke.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
