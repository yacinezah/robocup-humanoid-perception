# Selective-sharing implementation contract

Input is RGB float32 in [0,1], shape 1x3x640x640, static letterbox with padding 114.
The field branch uses stride-4/8 detail, private context stages and additive
upsampling. It returns one 320x320 logit. A field probability is contextual
evidence, not a ground-truth field boundary or a hard detector veto.

The YOLOv8 joint model exposes raw 16-bin distance-distribution and six-class
logits, shape 1x70x8400. Efficient YOLO26 exposes one-to-one raw values,
1x10x8400, or the native decoded 1x300x6 contract. The decoder preserves
YOLO26's two-stage top-K, including distinct class rows from one grid cell.

`model.py` contains the private field hierarchy, frozen-statistics behavior,
masked boundary-BCE/Dice field objective and FP32-safe specialist KL terms.
`efficient_model.py` preserves the detector while adapting tap widths.
`train.py` is the cleaned final joint-family recipe, not a universal trainer.
The exported efficient model was trained by a distinct frozen-transfer recipe.

The supervised task loss is applied only on that task's annotated microbatch.
Teacher predictions are regularizers, not replacement ground truth. Padding
is ignored by the valid-pixel mask. Field teacher preprocessing is a native
320 RGB/ImageNet resize, mapped into student coordinates. Do not assume it is
equivalent to halving a rounded 640 letterbox.

No checkpoint is loaded from a model-name string. Pass an existing local YAML
or checkpoint. No implicit network weights are permitted by public examples.
Native/raw interface equivalence is distinct from strict cross-backend parity.
The selected models failed the historical raw 1e-4 ceiling.
