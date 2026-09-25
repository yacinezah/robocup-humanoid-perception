"""Dataset-free analytical checks, not a robot accuracy experiment."""
import json
import numpy as np
from robocup_perception.data.eligibility import audit
from robocup_perception.localization.weights import clipped_best, normalize_log_weights
from robocup_perception.localization.lines import line_potential

records = [{'id': 'a', 'session': 'walk-1', 'split': 'train'},
           {'id': 'b', 'session': 'walk-1', 'split': 'validation'}]
leak = audit(records)
scores = [clipped_best([-2.]), clipped_best([-40.])]
weights = normalize_log_weights(scores)
support = {'robot_xy_m': [[.2, 0], [.4, 0], [.6, 0], [.8, 0]], 'confidence': [.9]*4}
line_scores = line_potential([[0, 0, 0], [0, 1.5, 0]], support, [[[-3, 0], [3, 0]]], .75)
assert leak['cross_split_components'] and weights[0] > weights[1] and line_scores[0] > line_scores[1]
print(json.dumps({'status': 'PASS', 'evidence': 'synthetic analytical smoke only',
                  'sequence_leak_detected': True, 'normalized_weights': weights.tolist(),
                  'line_scores': line_scores.tolist()}, indent=2))
