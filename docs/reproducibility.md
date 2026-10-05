# Reproducibility

## Three levels

1. **Public CPU smoke tests:** synthetic inputs and analytical fixtures, no data or weights.
2. **Inference:** explicitly obtained model files and a user-owned image, with model-card preprocessing.
3. **Scientific reproduction:** the correct licensed data, permitted split, pinned historical environment and protocol. The repository does not redistribute these data or authorize reopening frozen evaluations.

```bash
python -m pip install -e ".[test,runtime]"
python -m pytest
python examples/synthetic_smoke.py
```

Use `python scripts/audit_publication.py` to check links, nomenclature and file
boundaries. Figure generation reads published aggregate CSV tables only.

## Environments and numerical identity

The multitask training archive used Torch 2.11.0+cu130 and Ultralytics 8.4.96.
Laptop export used Torch 2.5.1+cu121, Ultralytics 8.4.60 and OpenVINO 2026.2.0.
The detector NUC benchmark used OpenVINO 2024.2. These are different environments.
Do not silently replace them with a single current dependency lock.

The public cleanup changes imports and configuration paths. It is not claimed
to be the byte-identical historical training tree. The source registry records
source and publication hashes. Earlier discovery trainer revisions remain
archived; the cleaned latest trainer does not reproduce every earlier arm's
implementation by substitution. GPU bit-exact restart was not independently
demonstrated by the historical study.

## Current separate stack

The [current reference](system-and-evidence.md) is DEP-R1
`rect_compact_buffered`. Install the package with the `runtime` extra as above.
The public contract code and dataset-free tests require no private material.
Model-backed inference requires a separately authorized ITAndroids handoff;
the production detector and latest dense binaries are **not public release
downloads**. Do not substitute similarly shaped historical release graphs.

Keep each `model.xml` beside its corresponding `model.bin`. For the default
profile, supply `detector-rectangular/model.xml` and
`segmenter-compact/model.xml` from that handoff:

```bash
perception-separate --detector artifacts/detector-rectangular/model.xml \
  --segmenter artifacts/segmenter-compact/model.xml \
  --profile rect_compact_buffered --threads 4 \
  --image my-frame.jpg --output output/prediction.json
```

The equivalent invocation without a refreshed console-script installation is
`python -m robocup_perception.runtime.separate` with the same arguments.
Choose a new `.json` output path. A sibling `.npz` stores a float32 array named
`probabilities`, shaped `[3,320,320]` in background/line/exclusive-field order.
Field union is channels 1+2. JSON records source detections, sparse line pixel
centres, the actual letterbox transform, class order, model hashes and effective
CPU properties. It never fabricates polygons, map identity or pose.

The [bundled manifest](../src/robocup_perception/runtime/separate_models.json)
defines four distinct buffered profiles: `rect_compact_buffered` (CPU),
`square_compact_buffered`, `rect_refit_buffered` and `square_refit_buffered`
(quality configuration). Each binds the exact XML and BIN hashes and tensor
shapes. Hash mismatch stops before compilation; don't edit the manifest to make
an arbitrary export look like a validated reference. The hash identifies bytes,
not authenticity or redistribution permission. Verify the handoff source too.

Detector input is RGB float32 [0,1], no-scaleup letterbox to 480×640 or 640×640,
padding 114. Output is decoded xywh plus six probabilities in order ball,
goalpost, robot, L, T, X. Predecode floor is >0.25, class-aware NMS IoU 0.7,
then robot confidence >=0.6 and other classes >=0.3. Dense input is an independent
RGB 320×320 warp; ImageNet normalization is inside the graph. Sparse support
uses threshold 0.30, 16×16 tiles of 20×20 native pixels and at most 32 diverse
points, mapped through pixel centres. This is not the raw-head multitask decoder.

The CLI requests FP32, one stream, 1 or 4 inference threads, P cores and no
hyperthreading. It does not change process affinity, power plan or system settings.
The archived benchmark additionally pinned its worker to audited P-core primaries.
Effective properties depend on the CPU/plugin. The extraction adds validation
and a concurrency guard: its latency has **not** been benchmarked as 46.993 ms.
One instance owns reusable buffers and rejects overlapping calls; returned arrays
remain valid after later inference. Create separate instances for concurrent
callers and budget their aggregate threads explicitly.

The quality/timing evidence used landscape images with aspect ratio at least
4:3. Other user images are technically accepted but have no corresponding
validation claim. Strict historical Torch/IR parity failures remain failures;
an exact archived-runtime extraction check does not repair export parity.
No ROS, camera, calibrated projection, particle filter or real-robot accuracy is
included. The old `perception-benchmark` measures one model on synthetic input;
it cannot reproduce the archived complete separate-call p95 by itself.

## Training interface

The extracted trainer requires `PERCEPTION_WORKROOT` pointing to a separately
authorized work directory and `PERCEPTION_TRAINING_AUTHORIZED=1`. That directory
must contain `manifest.json`, `PROTOCOL.md`, explicit local teacher files and
any teacher cache required by the selected arm. It never constructs a split
from public report tables. Training is not run in CI or during publication.

Use the [multitask contract](methods/multitask-contract.md) before invoking any
training code. Its selected historical models remain development artifacts.

## Releases

Each asset has SHA-256, purpose, licence status, input/output contract and
validation status. Do not load untrusted PyTorch pickle checkpoints. Prefer
ONNX/OpenVINO for inspection where available. No model file downloads on import.

The sixteen reports are condensed public editions of verified original
authorities. Their hashes and ENSEIRB mapping are recorded under
`reproducibility/`. Cite the publication commit, not a moving branch, for a fixed result.
