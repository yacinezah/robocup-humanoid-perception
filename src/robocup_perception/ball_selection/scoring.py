"""Frozen scoring primitives. No parameter fitting or data access at import."""
import math
import numpy as np
def logit(value):
    value=float(np.clip(value,1e-5,1-1e-5)); return math.log(value/(1-value))


def vector(candidate, provider, stage):
    f=candidate["providers"][provider]
    if not f["segmentation_valid"] or not np.isfinite(f["local_segmentation_uncertainty"]): return None
    if f["local_segmentation_uncertainty"] >= 0.88: return None
    boundary=float(np.clip(f["predicted_boundary_distance_box_heights"],-4,4))
    values=[logit(candidate["confidence"]),f["contact_disk_field_probability"],f["ring_field_probability"],math.tanh(boundary)]
    names=["detector_logit_confidence","contact_field_probability","ring_field_probability","tanh_boundary_box_heights"]
    if stage in ("F3_GEOMETRY","F4_TEMPORAL"):
        uncertainty=f["local_segmentation_uncertainty"]
        scale=math.exp(-min(float(f["apparent_scale_log_error"] or 5.0),5.0))
        protection=uncertainty*math.exp(-abs(boundary))
        values.extend([1-uncertainty,f["contact_minus_center_probability"],scale,float(f["projection_valid"]),protection])
        names.extend(["segmentation_certainty","contact_minus_center","scale_consistency","projection_valid","uncertain_boundary_protection"])
    return np.asarray(values,float),names


def score(candidate,provider,stage,parameters):
    item=vector(candidate,provider,stage)
    if item is None:return None
    values,_=item; z=parameters["intercept"]+values@np.asarray(parameters["coefficients"])
    return float(1/(1+math.exp(-float(np.clip(z,-40,40)))))


def old_hard_valid(candidate,provider):
    f=candidate["providers"][provider]
    if not f["segmentation_valid"] or f["local_segmentation_uncertainty"]>=0.88:return None
    return bool(f["contact_disk_field_probability"]>=0.5 and f["predicted_boundary_distance_box_heights"]>=-0.10)
