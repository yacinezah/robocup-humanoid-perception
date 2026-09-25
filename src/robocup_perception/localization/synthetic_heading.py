"""Causal synthetic sensor diagnostic. Not measured real IMU performance."""
import hashlib
import math
import numpy as np

def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def seed_for(scenario, replicate, purpose):
    return int.from_bytes(hashlib.sha256(f'{scenario}|{replicate}|LOC-R2|{purpose}'.encode()).digest()[:8], 'big')

def simulate_heading(rows, replicate):
    """Benchmark generator alone accesses GT; no identities/measurements used.

    Sequential loop is prefix invariant, including RNG consumption. Teleports
    invalidate heading without exposing their true rotation to the estimator.
    """
    rng = np.random.default_rng(seed_for(rows[0]['scenario_id'], replicate, 'sensor'))
    bias = rng.normal(0, math.radians(.5))
    initial = wrap(float(rows[0]['ground_truth_se2'][2]) + rng.normal(0, math.radians(10)))
    out = []
    for i, row in enumerate(rows):
        stamp = float(row['elapsed_s'])
        dt = 0. if i == 0 else stamp-float(rows[i-1]['elapsed_s'])
        if i and dt <= 0: raise ValueError('Nonmonotonic sensor timestamps')
        valid = not bool(row['reset_event'])
        delta = 0.
        if i and valid:
            delta = float(wrap(row['ground_truth_se2'][2]-rows[i-1]['ground_truth_se2'][2]))
            delta += bias*dt + rng.normal(0, math.radians(.5)*math.sqrt(dt))
        out.append({'stamp_s':stamp, 'dt_s':dt, 'delta_yaw_rad':float(delta),
                    'valid':valid, 'heading_lost':bool(row['reset_event']),
                    'initial_heading_rad':float(initial) if i == 0 else None,
                    'source':'EXPLICIT_SYNTHETIC_NOISY_RELATIVE_HEADING'})
    return out

def validate_sample(sample, camera_stamp_s):
    allowed = {'stamp_s','dt_s','delta_yaw_rad','valid','heading_lost','initial_heading_rad','source'}
    if set(sample) != allowed: raise ValueError('Unexpected sensor fields')
    if sample['stamp_s'] > camera_stamp_s+1e-9: raise ValueError('Future sensor sample')
    if sample['dt_s'] < 0 or not np.isfinite([sample['stamp_s'],sample['dt_s'],sample['delta_yaw_rad']]).all():
        raise ValueError('Invalid sensor sample')
    if sample['initial_heading_rad'] is not None and not np.isfinite(sample['initial_heading_rad']):
        raise ValueError('Invalid initial heading')
    return sample

def aided_filter_class(legacy, stream, initial_mode='known'):
    """Same sensor-aware prediction for every condition; no GT feature reads."""
    class SensorFilter(legacy.ParticleFilter):
        def __init__(self, *args):
            super().__init__(*args)  # Existing declared synthetic POSITION prior.
            self.sensor_index = 0
            heading = stream[0]['initial_heading_rad']
            self.particles[:,2] = wrap(self.rng.normal(heading, math.radians(10), self.n))
            if initial_mode == 'ambiguous':
                self.particles[self.n//2:,2] = wrap(self.particles[self.n//2:,2]+np.pi)
        def predict(self, odometry):
            sample = stream[self.sensor_index]
            self.sensor_index += 1
            if not sample['valid']:
                return super().predict(odometry)
            dt = float(sample['dt_s'])
            velocity = np.asarray(odometry['velocity_robot_frame'],dtype=float)
            noise = self.rng.normal(0, [self.params['odometry_sigma_vx_mps'],self.params['odometry_sigma_vy_mps']],(self.n,2))
            delta = (velocity[:2]+noise)*dt
            c,s = np.cos(self.particles[:,2]),np.sin(self.particles[:,2])
            self.particles[:,0] += c*delta[:,0]-s*delta[:,1]
            self.particles[:,1] += s*delta[:,0]+c*delta[:,1]
            self.particles[:,2] = wrap(self.particles[:,2]+sample['delta_yaw_rad']+
                self.rng.normal(0,math.radians(.5)*math.sqrt(dt),self.n))
    return SensorFilter
