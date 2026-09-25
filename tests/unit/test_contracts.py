import numpy as np
import pytest
from robocup_perception.segmentation.labels import decode_mask, field_union
from robocup_perception.detection.study import training_arguments


def test_mask_contract():
    a = np.array([[0, 128, 255]], dtype=np.uint8)
    assert decode_mask(a).tolist() == [[0, 1, 2]]
    assert field_union(a).tolist() == [[0, 1, 1]]
    with pytest.raises(ValueError):
        decode_mask(np.array([[1]]))


def test_frozen_detector_recipe():
    a = training_arguments()
    assert (a['imgsz'], a['epochs'], a['amp']) == (640, 40, False)
    assert a['optimizer'] == 'AdamW'
