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
