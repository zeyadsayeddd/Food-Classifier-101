"""Locate float32 weight matrices in the serialized, immutable ONNX graph."""
import hashlib
import math
import onnx


def weight_layout(raw):
    tensors = []
    position = 0
    for tensor in onnx.load_model_from_string(raw).graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT or len(tensor.dims) < 2:
            continue
        data = tensor.raw_data
        offset = raw.index(data, position)
        rows, columns = tensor.dims[0], math.prod(tensor.dims[1:])
        assert len(data) == rows * columns * 4 and columns >= 2
        tensors.append([offset, rows, columns])
        position = offset + len(data)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "tensors": tensors}
