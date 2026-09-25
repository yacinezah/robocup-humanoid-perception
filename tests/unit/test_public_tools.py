import numpy as np
import pytest
from robocup_perception.data.eligibility import audit
from robocup_perception.data.bootstrap import paired_cluster_mean_difference
from robocup_perception.localization.weights import clipped_best, normalize_log_weights
from robocup_perception.localization.association import component_scores, public_observation
from robocup_perception.runtime.parity import compare
from robocup_perception.detection.decode import decode_raw
from robocup_perception.runtime.infer_openvino import letterbox
from robocup_perception.ball_selection.scoring import score


def test_component_exposure_and_leakage():
    rows = [{'id':'a','session':'s','split':'train','exposures':['teacher']},
            {'id':'b','session':'s','split':'validation'}, {'id':'c','split':'final'}]
    result = audit(rows, [('b','c')])
    assert result['cross_split_components']['a'] == ['final','train','validation']
    assert result['component_exposures']['a'] == ['teacher']


def test_duplicate_id_rejected():
    with pytest.raises(ValueError): audit([{'id':'a'}, {'id':'a'}])


def test_paired_cluster_determinism():
    a = paired_cluster_mean_difference([0,1,2,3], [1,2,3,4], ['a','a','b','b'], draws=100)
    assert a['delta'] == 1 and a['ci95'] == [1,1]


def test_likelihood_gate_and_normalization():
    assert clipped_best([-40]) == -10
    assert clipped_best([-2]) > clipped_best([-40])
    w = normalize_log_weights([-1000,-1001,-1020])
    assert np.isfinite(w).all() and np.isclose(w.sum(), 1)


def test_strict_parity_does_not_hide_permutation():
    assert not compare([1,2],[2,1])['passed']
    assert not compare([1],[np.nan])['passed']
    assert compare([],[])['passed']


def test_decoder_empty_and_class_aware():
    assert decode_raw(np.zeros((1,300,6),np.float32),.25).shape == (0,6)
    x = np.zeros((1,300,6),np.float32)
    x[0,:2] = [[1,1,12,12,.9,0], [1,1,12,12,.8,1]]
    assert len(decode_raw(x,.25)) == 2


@pytest.mark.parametrize('shape', [(480,640),(301,703),(703,301)])
def test_letterbox_geometry_roundtrip(shape):
    im = np.zeros((*shape,3),np.uint8)
    out,(s,l,t,rw,rh) = letterbox(im,640)
    p = np.array([13.,29.])
    np.testing.assert_allclose(((p*s+[l,t])-[l,t])/s,p)
    assert out.shape == (640,640,3) and rw <= 640 and rh <= 640


def test_gt_not_consumed_by_association():
    o = {'kind':'L','confidence':.8,'robot_xy_m':[1.,0.],
         'covariance_robot_m2':[[.01,0],[0,.01]],'gt_identity':'secret'}
    assert 'gt_identity' not in public_observation(o)
    s = component_scores(np.array([[0,0,0],[0,1,0]]),o,{'map_l':[1,0]})
    assert s.shape == (2,2) and np.isfinite(s).all()


def test_invalid_segmentation_fails_open_signal():
    c = {'confidence':.9,'providers':{'field':{'segmentation_valid':False}}}
    assert score(c,'field','F2_SOFT_STATIC',{}) is None
