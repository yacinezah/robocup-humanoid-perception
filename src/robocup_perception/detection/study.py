"""Portable controlled-detector recipe. No automatic data or weight downloads.

The recipe is extracted from the frozen controlled comparison. User-supplied
data must have its own legal split. This CLI does not authorize old test access.
"""
import argparse
from pathlib import Path


def training_arguments():
    return dict(imgsz=640, epochs=40, patience=10, batch=16, nbs=32,
                workers=8, optimizer='AdamW', lr0=1e-3, weight_decay=5e-4,
                cos_lr=True, warmup_epochs=3., deterministic=True, amp=False,
                fliplr=.5, flipud=0., hsv_h=.015, hsv_s=.5, hsv_v=.3,
                translate=.10, scale=.25, mosaic=.5, close_mosaic=10,
                mixup=0., copy_paste=0., perspective=0., shear=0., degrees=0.,
                rect=False, cache=False, save=True, save_period=1, plots=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate'])
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=930)
    p.add_argument('--device', default='cpu')
    p.add_argument('--authorize-training', action='store_true')
    a = p.parse_args()
    if not a.weights.is_file() or not a.data.is_file():
        p.error('Explicit local weights and dataset YAML are required.')
    if a.output.exists():
        p.error('Use a new output directory; existing results are immutable.')
    if a.mode == 'train' and not a.authorize_training:
        p.error('Training requires --authorize-training and an eligible dataset.')
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    common = dict(data=str(a.data), project=str(a.output.parent),
                  name=a.output.name, exist_ok=False, device=a.device)
    if a.mode == 'train':
        model.train(**training_arguments(), **common, seed=a.seed)
    else:
        model.val(**common, imgsz=640, split='val', conf=.001, iou=.7, max_det=300)


if __name__ == '__main__':
    main()
