from __future__ import annotations

import numbers

import torch

from codegen_backend.errors import CodegenBackendError
from codegen_backend.dtypes import (
    _C_TYPE_BY_DTYPE,
    _CodegenDType,
    _EMBEDDING_INDEX_DTYPES,
    _INTEGER_CODEGEN_DTYPES,
)


def _input_c_type(dtype: torch.dtype, graph_dtype: _CodegenDType) -> str:
    if dtype is graph_dtype.torch_dtype:
        return graph_dtype.c_type
    if dtype is torch.bool:
        return _C_TYPE_BY_DTYPE[torch.bool]
    if dtype in _EMBEDDING_INDEX_DTYPES:
        return _C_TYPE_BY_DTYPE[dtype]
    raise CodegenBackendError(
        "codegen backend supports only torch.float32, torch.int8, torch.int32, torch.int64, or torch.bool tensors"
    )


def _dtype_to_c_type(dtype: torch.dtype, graph_dtype: _CodegenDType) -> str:
    if dtype is graph_dtype.torch_dtype:
        return graph_dtype.c_type
    c_type = _C_TYPE_BY_DTYPE.get(dtype)
    if c_type is not None:
        return c_type
    raise CodegenBackendError(
        "codegen backend supports only torch.float32, torch.int8, torch.int32, torch.int64, or torch.bool tensors"
    )


def dtype_to_c_type(dtype: torch.dtype) -> str:
    c_type = _C_TYPE_BY_DTYPE.get(dtype)
    if c_type is not None:
        return c_type
    raise CodegenBackendError(
        "codegen backend supports only torch.float32, torch.int8, torch.int32, torch.int64, or torch.bool tensors"
    )


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in _INTEGER_CODEGEN_DTYPES


def _format_scalar_literal(value: float, dtype: _CodegenDType) -> str:
    if dtype.torch_dtype is torch.bool:
        return str(int(value))
    if _is_integer_dtype(dtype.torch_dtype):
        return str(int(value))
    if dtype.torch_dtype is torch.float32:
        return f"{float(value)}f"
    raise CodegenBackendError(
        "codegen addmm-like ops support only floating point or integer tensors"
    )


def _normalize_scalar_value(op_name: str, value: object) -> float | int | bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(f"codegen {op_name} expects a scalar value")
        value = value.item()
    if isinstance(value, numbers.Number):
        return value
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects a scalar value"
        ) from exc


def format_scalar(value: object, dtype: _CodegenDType) -> str:
    normalized = _normalize_scalar_value("scalar", value)
    return _format_scalar_literal(normalized, dtype)
