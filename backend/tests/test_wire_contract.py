"""Numeric-only wire contract; this does not prove live TPU compatibility."""

import numpy as np
import pytest

from app.inference.tpu_wire import PyTask, deserialize, serialize


def test_numeric_tensors_metadata_and_worker_report_round_trip():
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = serialize(PyTask({"images": data}, metadata=[{"score": np.array([0.5])}],
                               worker_pid="fixture", status_code=7, error_message="fixture"))
    result, metadata, report = deserialize(payload)
    np.testing.assert_array_equal(result["images"], data)
    np.testing.assert_array_equal(metadata[0]["score"], [0.5])
    assert report["status_code"] == 7
    assert report["worker_pid"] == "fixture"


def test_object_tensor_is_rejected_before_serialization():
    with pytest.raises(ValueError, match="numeric"):
        serialize(PyTask({"unsafe": np.array([object()], dtype=object)}))


def test_oversized_wire_message_is_rejected_before_parsing():
    with pytest.raises(ValueError, match="size"):
        deserialize(b"x" * (64 * 1024 * 1024 + 1))


def test_decode_known_protocol_vector():
    # Serialized fixture containing only float32 [1, 2], worker name and status.
    payload = bytes.fromhex("0a220a076f75747075743012170a07666c6f61743332120201021a080000803f000000402207666978747572653803")
    data, metadata, report = deserialize(payload)
    np.testing.assert_array_equal(data["output0"], [[1.0, 2.0]])
    assert metadata == ()
    assert report["worker_pid"] == "fixture"
    assert report["status_code"] == 3
