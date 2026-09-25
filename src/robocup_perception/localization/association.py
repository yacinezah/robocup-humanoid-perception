"""GT-free observation scoring and mode extraction; no replay/file access."""
import math
import numpy as np


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def public_observation(o):
    # Explicit allowlist: GT identities/centres cannot enter the learned method.
    return {k: o[k] for k in ('kind', 'confidence', 'robot_xy_m', 'covariance_robot_m2')}


def expected(particles, point):
    delta = np.asarray(point) - particles[:, :2]
    c, s = np.cos(particles[:, 2]), np.sin(particles[:, 2])
    return np.column_stack((c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1]))


def component_scores(particles, observation, maps):
    """Normalized 2-D Student-t mixture plus uniform clutter within 12 m.

    nu=3, 10cm projection/systematic floor and 3% range-dependent scale.
    These engineering settings are frozen for the first discovery fit.
    Detector confidence controls an inlier prior, not a hard accept decision.
    """
    o = public_observation(observation)
    z = np.asarray(o['robot_xy_m'], dtype=float)
    cov = np.asarray(o['covariance_robot_m2'], dtype=float)
    if z.shape != (2,) or cov.shape != (2, 2) or not np.isfinite(z).all() or not np.isfinite(cov).all():
        raise ValueError('Invalid measurement')
    confidence = float(o['confidence'])
    if not np.isfinite(confidence):
        raise ValueError('Invalid confidence')
    p = float(np.clip(confidence, .05, .95))
    cov = (cov+cov.T)/2
    eig = np.linalg.eigvalsh(cov)
    if eig[0] < -1e-8:
        raise ValueError('Indefinite covariance')
    cov += np.eye(2) * (.10**2 + (.03*np.linalg.norm(z))**2)
    inverse = np.linalg.inv(cov)
    nu = 3.
    norm = math.lgamma((nu+2)/2)-math.lgamma(nu/2)-math.log(nu*math.pi)-.5*np.linalg.slogdet(cov)[1]
    values = []
    for point in maps.values():
        residual = z-expected(particles, point)
        mahal = np.einsum('ni,ij,nj->n', residual, inverse, residual)
        values.append(math.log(p/len(maps))+norm-(nu+2)/2*np.log1p(mahal/nu))
    clutter = np.full(len(particles), math.log(1-p)-math.log(math.pi*12**2))
    return np.column_stack(values+[clutter])


def mixture_likelihood(particles, observations, maps, estimate):
    total = np.zeros(len(particles))
    diagnostic = []
    for observation in observations:
        kind_map = maps[observation['kind']]
        try:
            scores = component_scores(particles, observation, kind_map)
            total += np.logaddexp.reduce(scores, axis=1)
            d = component_scores(np.asarray(estimate).reshape(1, 3), observation, kind_map)[0]
            posterior = np.exp(d-np.logaddexp.reduce(d))
            j = int(np.argmax(posterior))
            match = list(kind_map)[j] if j < len(kind_map) and posterior[j] >= .5 else None
        except ValueError:
            match = None  # Invalid sensor input is an uninformative update.
        diagnostic.append((observation, match))
    return total, diagnostic


def extract_modes(particles, weights):
    """Local posterior modes, avoiding a mean between incompatible poses.

    Deterministic coarse-bin seeds; bounded 24 local neighbourhoods. This is
    pose extraction only: particles/weights are not changed or mirror-selected
    from GT. Two equivalent field modes remain ambiguous.
    """
    scales = np.array([.5, .5, math.radians(20)])
    bins = np.floor(particles/scales).astype(int)
    _, inverse = np.unique(bins, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    seeds = np.argsort(-mass, kind='stable')[:24]
    seeds = [s for s in seeds if mass[s] > 0]
    centres = np.column_stack([np.bincount(inverse, weights=weights*particles[:, j])/np.maximum(mass, 1e-300) for j in (0, 1)])
    sin = np.bincount(inverse, weights=weights*np.sin(particles[:, 2]))
    cos = np.bincount(inverse, weights=weights*np.cos(particles[:, 2]))
    poses = np.column_stack((centres, np.arctan2(sin, cos)))
    modes = []
    for seed in seeds:
        pose = poses[seed]
        mask = (np.linalg.norm(particles[:, :2]-pose[:2], axis=1) < .75) & (np.abs(wrap(particles[:, 2]-pose[2])) < math.radians(25))
        w = weights*mask
        total = float(w.sum())
        if total <= 0:
            continue
        w /= total
        centre = np.array([w@particles[:, 0], w@particles[:, 1], math.atan2(w@np.sin(particles[:, 2]), w@np.cos(particles[:, 2]))])
        if any(np.linalg.norm(centre[:2]-p[:2]) < .75 and abs(wrap(centre[2]-p[2])) < math.radians(25) for _, p in modes):
            continue
        modes.append((total, centre))
    return sorted(modes, key=lambda x: -x[0])
