"""Dataset-free regression checks for the current separate-stack contract."""
import hashlib
import json
import sys

import cv2
import numpy as np
import pytest

from robocup_perception.runtime import separate
from robocup_perception.localization.sparse_support import support_pixels, source_coordinates


@pytest.mark.parametrize("height", [480, 640])
@pytest.mark.parametrize("shape", [(240, 320), (479, 641), (1080, 1920), (703, 301)])
def test_letterbox_and_buffer_retention(height, shape):
    image = np.random.default_rng(7).integers(0, 256, (*shape, 3), dtype=np.uint8)
    buffers = [np.empty((1, 3, height, 640), np.float32), np.empty((1, 3, 320, 320), np.float32)]
    a, b, meta = separate.prepare(image, height)
    c, d, buffered_meta = separate.prepare(image, height, buffers)
    assert meta == buffered_meta
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert np.shares_memory(c, buffers[0])
    # An asymmetric source box catches swapped axes and guessed square gains.
    source = np.array([[13., 27., 200., 215., .9, 0]], np.float32)
    network = source.copy()
    network[:, [0, 2]] = source[:, [0, 2]] * meta["r"] + meta["pad"][0]
    network[:, [1, 3]] = source[:, [1, 3]] * meta["r"] + meta["pad"][1]
    np.testing.assert_allclose(separate.restore(network, meta), source, atol=1e-4)
    expected = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (320, 320))
    np.testing.assert_array_equal(b, expected.transpose(2, 0, 1)[None].astype(np.float32) / 255.)


def test_small_source_is_never_scaled_up():
    x, _, meta = separate.prepare(np.full((240, 320, 3), (0, 127, 255), np.uint8))
    assert meta["r"] == 1. and meta["pad"] == (160, 120)
    np.testing.assert_allclose(x[0, :, 120, 160], [1., 127 / 255., 0.])
    np.testing.assert_allclose(x[0, :, 0, 0], [114 / 255.] * 3)


@pytest.mark.parametrize("image", [np.zeros((2, 2)), np.zeros((2, 2, 3)), np.zeros((0, 2, 3), np.uint8)])
def test_invalid_input_rejected(image):
    with pytest.raises(ValueError):
        separate.prepare(image)


def test_probability_decoder_nms_is_class_aware_and_stable():
    raw = np.zeros((1, 10, 6300), np.float32)
    raw[0, :4, :4] = np.array([[50, 50, 20, 20]] * 4).T
    raw[0, 4, :2] = .8  # Same-class tie: retain first.
    raw[0, 5, 2] = .8  # Different class: retain despite overlapping box.
    raw[0, 4, 3] = .25  # Exact predecode floor is excluded.
    boxes = separate.decode_probabilities(raw)
    assert boxes.shape == (2, 6)
    np.testing.assert_array_equal(boxes[:, 5], [0, 1])
    np.testing.assert_array_equal(boxes[:, :4], [[40, 40, 60, 60]] * 2)
    assert separate.decode_probabilities(np.zeros_like(raw)).shape == (0, 6)


@pytest.mark.parametrize("kind", ["e2e", "nan", "logits", "negative_width"])
def test_wrong_detector_contract_rejected(kind):
    raw = np.zeros((1, 10, 6300), np.float32)
    if kind == "e2e":
        raw = np.zeros((1, 300, 6), np.float32)
    elif kind == "nan":
        raw[0, 0, 0] = np.nan
    elif kind == "logits":
        raw[0, 4, 0] = -3
    else:
        raw[0, 2, 0] = -1
    with pytest.raises((ValueError, FloatingPointError)):
        separate.decode_probabilities(raw)


def test_softmax_and_native_sparse_pixel_centres():
    logits = np.zeros((1, 3, 320, 320), np.float32)
    logits[0, 1, 50, 60] = 20
    probabilities = separate.probabilities(logits)
    np.testing.assert_allclose(probabilities.sum(0), 1., atol=1e-6)
    points = support_pixels(probabilities[1], .45)
    assert points.shape == (1, 3)
    np.testing.assert_allclose(source_coordinates(points, 640, 480)[0, :2], [120.5, 75.25])
    empty = support_pixels(np.zeros((320, 320)), .30)
    assert empty.shape == (0, 3)
    assert source_coordinates(empty, 640, 480).shape == (0, 3)
    logits[0, 0, 0, 0] = np.inf
    with pytest.raises(FloatingPointError):
        separate.probabilities(logits)


def test_sparse32_budget_and_deterministic_spatial_coverage():
    prob = np.full((320, 320), .5, np.float32)
    points = support_pixels(prob)
    assert len(points) == len(np.unique(points[:, :2], axis=0)) == 32
    np.testing.assert_array_equal(points[:2, :2], [[0, 0], [300, 300]])
    np.testing.assert_array_equal(points, support_pixels(prob))
    with pytest.raises(ValueError):
        support_pixels(prob, max_points=16)


@pytest.mark.parametrize("changed", ["xml", "bin"])
def test_both_graph_and_weights_are_hash_checked(tmp_path, changed):
    expected = {}
    for suffix in ("xml", "bin"):
        contents = suffix.encode()
        (tmp_path / f"model.{suffix}").write_bytes(contents)
        expected[f"{suffix}_sha256"] = hashlib.sha256(contents).hexdigest()
    assert separate.verify_artifact(tmp_path / "model.xml", expected)["xml"] == expected["xml_sha256"]
    (tmp_path / f"model.{changed}").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        separate.verify_artifact(tmp_path / "model.xml", expected)


@pytest.fixture
def synthetic_stack(tmp_path, monkeypatch):
    """Exercise actual OpenVINO requests; bypass only identity for artificial graphs."""
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    detector_input = ops.parameter([1, 3, 480, 640], np.float32)
    mean = ops.reduce_mean(detector_input, ops.constant([0, 1, 2, 3]), False)
    detector_output = ops.broadcast(mean, ops.constant([1, 10, 6300]))
    dense_input = ops.parameter([1, 3, 320, 320], np.float32)
    dense_output = ops.multiply(dense_input, ops.constant(np.array([1, 2, 3], np.float32).reshape(1, 3, 1, 1)))
    for name, model in (("detector", ov.Model([detector_output], [detector_input])),
                        ("segmenter", ov.Model([dense_output], [dense_input]))):
        ov.save_model(model, tmp_path / f"{name}.xml", compress_to_fp16=False)
    monkeypatch.setattr(separate, "verify_artifact", lambda *args: {"test_only": "synthetic graph"})
    return separate.SeparateStack(tmp_path / "detector.xml", tmp_path / "segmenter.xml", threads=1)


def test_actual_requests_keep_returned_arrays_alive_and_recover_from_invalid_input(synthetic_stack):
    stack = synthetic_stack
    first = stack.infer(np.full((240, 320, 3), 100, np.uint8))
    saved = {key: first[key].copy() for key in ("detections", "probabilities", "source_line_pixels")}
    second = stack.infer(np.full((240, 320, 3), 200, np.uint8))
    for key, value in saved.items():
        np.testing.assert_array_equal(first[key], value)
        assert not np.shares_memory(first[key], second[key])
    assert not np.array_equal(first["probabilities"], second["probabilities"])
    with pytest.raises(ValueError):
        stack.infer(np.zeros((0, 3, 3), np.uint8))
    assert stack.infer(np.zeros((2, 2, 3), np.uint8))["probabilities"].shape == (3, 320, 320)


def test_concurrent_reentry_fails_explicitly(synthetic_stack):
    stack = synthetic_stack
    stack._busy.acquire()
    try:
        with pytest.raises(RuntimeError, match="non-reentrant"):
            stack.infer(np.zeros((10, 10, 3), np.uint8))
    finally:
        stack._busy.release()


def test_cli_writes_explicit_contract(synthetic_stack, tmp_path, monkeypatch):
    image = tmp_path / "user.png"
    cv2.imwrite(str(image), np.full((240, 320, 3), 100, np.uint8))
    output = tmp_path / "prediction.json"
    monkeypatch.setattr(separate, "SeparateStack", lambda *args: synthetic_stack)
    monkeypatch.setattr(sys, "argv", ["perception-separate", "--detector", str(tmp_path / "detector.xml"),
                                    "--segmenter", str(tmp_path / "segmenter.xml"), "--image", str(image),
                                    "--output", str(output)])
    separate.main()
    result = json.loads(output.read_text())
    assert result["dense_class_order"] == ["background", "line", "exclusive_field"]
    with np.load(output.with_suffix(".npz")) as arrays:
        assert arrays["probabilities"].shape == (3, 320, 320)
    with pytest.raises(SystemExit):
        separate.main()  # Refuse accidental output overwrite.
