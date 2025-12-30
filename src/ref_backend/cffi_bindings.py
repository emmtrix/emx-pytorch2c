import ctypes
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


class RefBackendError(RuntimeError):
    pass


class RefDType:
    REF_F32 = 0


class RefOpKind:
    REF_OP_ADD = 0
    REF_OP_MATMUL = 1


class RefTensorView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int32),
        ("sizes", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("dtype", ctypes.c_int32),
    ]


class RefOpCall(ctypes.Structure):
    _fields_ = [
        ("inputs", ctypes.POINTER(RefTensorView)),
        ("n_inputs", ctypes.c_int32),
        ("outputs", ctypes.POINTER(RefTensorView)),
        ("n_outputs", ctypes.c_int32),
        ("params", ctypes.c_void_p),
    ]


@dataclass
class TensorViewBuffers:
    sizes: ctypes.Array
    strides: ctypes.Array


class RefBackendLibrary:
    def __init__(self) -> None:
        module = importlib.import_module("ref_backend._ref_backend")
        self.lib = ctypes.CDLL(module.__file__)
        self.lib.ref_run_op.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(RefOpCall),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.ref_run_op.restype = ctypes.c_int32

    def run_op(self, op_kind: int, call: RefOpCall) -> None:
        err_cap = 512
        err_buf = ctypes.create_string_buffer(err_cap)
        rc = self.lib.ref_run_op(op_kind, ctypes.byref(call), err_buf, err_cap)
        if rc != 0:
            msg = err_buf.value.decode("utf-8")
            raise RefBackendError(f"ref_run_op failed ({rc}): {msg}")


_lib_instance: Optional[RefBackendLibrary] = None


def _get_library() -> RefBackendLibrary:
    global _lib_instance
    if _lib_instance is None:
        _lib_instance = RefBackendLibrary()
    return _lib_instance


def _dtype_to_ref(dtype: torch.dtype) -> int:
    if dtype is torch.float32:
        return RefDType.REF_F32
    raise RefBackendError(f"Unsupported dtype: {dtype}")


def _tensor_to_view(tensor: torch.Tensor) -> Tuple[RefTensorView, TensorViewBuffers]:
    sizes = (ctypes.c_int64 * tensor.ndim)(*tensor.size())
    strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
    view = RefTensorView(
        data=ctypes.c_void_p(tensor.data_ptr()),
        ndim=ctypes.c_int32(tensor.ndim),
        sizes=sizes,
        strides=strides,
        dtype=_dtype_to_ref(tensor.dtype),
    )
    return view, TensorViewBuffers(sizes=sizes, strides=strides)


def run_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if a.dtype is not torch.float32 or b.dtype is not torch.float32 or out.dtype is not torch.float32:
        raise RefBackendError("add supports only torch.float32 tensors")
    if a.shape != b.shape or a.shape != out.shape:
        raise RefBackendError("add requires inputs and output to have identical shapes")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("add requires contiguous tensors")

    a_view, a_buf = _tensor_to_view(a)
    b_view, b_buf = _tensor_to_view(b)
    out_view, out_buf = _tensor_to_view(out)

    inputs = (RefTensorView * 2)(a_view, b_view)
    outputs = (RefTensorView * 1)(out_view)
    call = RefOpCall(
        inputs=inputs,
        n_inputs=ctypes.c_int32(2),
        outputs=outputs,
        n_outputs=ctypes.c_int32(1),
        params=None,
    )

    _ = (a_buf, b_buf, out_buf)
    _get_library().run_op(RefOpKind.REF_OP_ADD, call)


def run_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if a.dtype is not torch.float32 or b.dtype is not torch.float32 or out.dtype is not torch.float32:
        raise RefBackendError("matmul supports only torch.float32 tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise RefBackendError("matmul requires 2D inputs and output")
    if a.shape[1] != b.shape[0]:
        raise RefBackendError("matmul requires inner dimensions to match")
    if out.shape != (a.shape[0], b.shape[1]):
        raise RefBackendError("matmul requires output shape (m, n)")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("matmul requires contiguous tensors")

    a_view, a_buf = _tensor_to_view(a)
    b_view, b_buf = _tensor_to_view(b)
    out_view, out_buf = _tensor_to_view(out)

    inputs = (RefTensorView * 2)(a_view, b_view)
    outputs = (RefTensorView * 1)(out_view)
    call = RefOpCall(
        inputs=inputs,
        n_inputs=ctypes.c_int32(2),
        outputs=outputs,
        n_outputs=ctypes.c_int32(1),
        params=None,
    )

    _ = (a_buf, b_buf, out_buf)
    _get_library().run_op(RefOpKind.REF_OP_MATMUL, call)
