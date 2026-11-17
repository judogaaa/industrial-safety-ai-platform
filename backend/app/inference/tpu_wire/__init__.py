"""Project-owned LinQ wire adapter built from the protocol's field contract.

No vendor Python source is distributed. Tensor payloads use protobuf and
fixed-size numeric NumPy dtypes; executable/object payloads are rejected.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf import timestamp_pb2  # noqa: F401 -- register dependency

MESSAGE_ID_ATTR_NAME = "tpu_pycloud_message_id"
RESPONSE_ROUTING_KEYS_ATTR_NAME = "response_routing_keys"
TARGET_URI_ATTR_NAME = "uri"
_MAX_BYTES = 64 * 1024 * 1024


def _task_type():
    schema = descriptor_pb2.FileDescriptorProto(
        name="portfolio_linq_wire.proto", package="portfolio_linq", syntax="proto3",
        dependency=["google/protobuf/timestamp.proto"],
    )

    def field(message, name, number, kind, *, repeated=False, ref=""):
        result = message.field.add(name=name, number=number, type=kind,
                                   label=3 if repeated else 1)
        if ref:
            result.type_name = ref

    array = schema.message_type.add(name="Array")
    field(array, "dtype", 1, 9)
    field(array, "shape", 2, 5, repeated=True)
    field(array, "byte_data", 3, 12)

    def tensor_map(message, name, number):
        entry = message.nested_type.add(name="TensorEntry")
        entry.options.map_entry = True
        field(entry, "key", 1, 9)
        field(entry, "value", 2, 11, ref=".portfolio_linq.Array")
        field(message, name, number, 11, repeated=True,
              ref=f".portfolio_linq.{message.name}.TensorEntry")

    meta = schema.message_type.add(name="MetadataDict")
    tensor_map(meta, "items", 1)
    task = schema.message_type.add(name="Task")
    tensor_map(task, "dict_data", 1)
    field(task, "list_metadata", 2, 11, repeated=True,
          ref=".portfolio_linq.MetadataDict")
    for number, name in enumerate(("data_path", "worker_pid", "thread_name"), 3):
        field(task, name, number, 9)
    field(task, "inference_time", 6, 2)
    field(task, "status_code", 7, 5)
    field(task, "error_message", 8, 9)
    field(task, "inference_opening_timestamp", 9, 11,
          ref=".google.protobuf.Timestamp")
    pool = descriptor_pool.Default()
    pool.AddSerializedFile(schema.SerializeToString())
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("portfolio_linq.Task"))


_Task = _task_type()


@dataclass(frozen=True)
class PyTask:
    data: Any
    metadata: Any = None
    worker_pid: str = ""
    thread_name: str = ""
    inference_opening_timestamp: datetime | None = None
    inference_time: float = 0.0
    status_code: int = 0
    error_message: str = ""


def _encode_array(target, value):
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufc" or array.nbytes > _MAX_BYTES:
        raise ValueError("Only bounded numeric tensors are accepted")
    target.dtype = array.dtype.name
    target.shape.extend(array.shape)
    target.byte_data = array.tobytes(order="C")


def serialize(value: PyTask) -> bytes:
    message = _Task()
    for name in ("worker_pid", "thread_name", "inference_time", "status_code", "error_message"):
        setattr(message, name, getattr(value, name))
    if value.inference_opening_timestamp is not None:
        message.inference_opening_timestamp.FromDatetime(value.inference_opening_timestamp)
    if isinstance(value.data, Path):
        message.data_path = str(value.data)
    elif isinstance(value.data, (str, bytes)):
        raise TypeError("Use a tensor mapping or pathlib.Path")
    else:
        for name, tensor in value.data.items():
            _encode_array(message.dict_data[name], tensor)
    for item in value.metadata or ():
        output = message.list_metadata.add()
        for name, tensor in item.items():
            _encode_array(output.items[str(name)], tensor)
    payload = message.SerializeToString()
    if len(payload) > _MAX_BYTES:
        raise ValueError("Wire message exceeds size limit")
    return payload


def _decode_array(value):
    dtype = np.dtype(value.dtype)
    if dtype.hasobject or dtype.kind not in "biufc":
        raise ValueError("Non-numeric tensor rejected")
    shape = tuple(value.shape)
    if len(shape) > 16 or any(d < 0 for d in shape):
        raise ValueError("Invalid tensor dimensions")
    return np.frombuffer(value.byte_data, dtype=dtype).reshape(shape).copy()


def deserialize(payload: bytes):
    if len(payload) > _MAX_BYTES:
        raise ValueError("Wire message exceeds size limit")
    message = _Task()
    message.ParseFromString(payload)
    tensors = {name: _decode_array(value) for name, value in message.dict_data.items()}
    metadata = tuple(
        {None if name == "None" else name: _decode_array(value)
         for name, value in item.items.items()}
        for item in message.list_metadata
    )
    report = {name: getattr(message, name) for name in
              ("worker_pid", "thread_name", "inference_time", "status_code", "error_message")}
    report["inference_opening_timestamp"] = message.inference_opening_timestamp.seconds
    return Path(message.data_path) if message.data_path else tensors, metadata, report
