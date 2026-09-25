"""Strict raw-tensor comparison, separate from operational equivalence."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate, tolerance=1e-4):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        return {'passed': False, 'reason': 'shape mismatch', 'reference_shape': list(a.shape), 'candidate_shape': list(b.shape)}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'passed': False, 'reason': 'nonfinite output'}
    maximum = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.
    return {'passed': maximum <= tolerance, 'maximum_absolute_error': maximum,
            'tolerance': tolerance, 'shape': list(a.shape), 'ordering_sensitive': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error('tolerance must be nonnegative')
    result = compare(np.load(args.reference, allow_pickle=False), np.load(args.candidate, allow_pickle=False), args.tolerance)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
