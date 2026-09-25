"""Model-only CPU timing on deterministic synthetic input, not accuracy evidence."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np


def main():
    import openvino as ov
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--threads', type=int, choices=[1, 4], default=4)
    p.add_argument('--warmups', type=int, default=100)
    p.add_argument('--calls', type=int, default=500)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--power-state', required=True, choices=['AC', 'battery', 'unknown'])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.calls, a.repetitions) <= 0 or a.warmups < 0:
        p.error('Invalid iteration count')
    compiled = ov.Core().compile_model(str(a.model), 'CPU', {'INFERENCE_NUM_THREADS': str(a.threads),
        'NUM_STREAMS': '1', 'INFERENCE_PRECISION_HINT': 'f32'})
    shape = list(compiled.input().shape)
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    for _ in range(a.warmups):
        compiled([x])
    rows = []
    for rep in range(a.repetitions):
        times = []
        for _ in range(a.calls):
            start = time.perf_counter_ns()
            outputs = compiled([x])
            times.append((time.perf_counter_ns()-start)/1e6)
        if not all(np.isfinite(v).all() for v in outputs.values()):
            raise ValueError('Nonfinite model output')
        rows.append({'repetition': rep+1, 'mean_ms': float(np.mean(times)),
                     'p50_ms': float(np.percentile(times, 50)), 'p95_ms': float(np.percentile(times, 95)),
                     'calls_ms': times})
    files = [a.model] + ([a.model.with_suffix('.bin')] if a.model.suffix == '.xml' else [])
    result = {'surface': 'MODEL_ONLY_SYNTHETIC_INPUT', 'shape': shape, 'platform': platform.platform(),
        'processor': platform.processor(), 'openvino': ov.__version__, 'threads': a.threads,
        'power_state_user_declared': a.power_state, 'continuous_power_or_thermal_telemetry': False,
        'warmups': a.warmups, 'repetitions': rows,
        'model_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'warning': 'Excludes preprocessing, decoding, camera and disk. Not comparable to historical complete-call p95.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
