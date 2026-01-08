from __future__ import annotations

import operator
from collections.abc import Sequence as ABCSequence
from typing import Dict, Sequence, Tuple

import torch
import torch.fx

from codegen_backend.c_types import _normalize_scalar_value
from codegen_backend.errors import CodegenBackendError
from codegen_backend.param_normalize import normalize_int_or_tuple


def is_out_overload(target: object) -> bool:
    schema = getattr(target, "_schema", None)
    return schema is not None and schema.overload_name == "out"


def error_expected_tensor(op_name: str) -> CodegenBackendError:
    return CodegenBackendError(f"codegen {op_name} expects tensor inputs only")


def error_kwarg_specified_once(op_name: str, kwarg: str) -> CodegenBackendError:
    return CodegenBackendError(f"codegen {op_name} expects {kwarg} to be specified once")


def normalize_param(normalizer, *args, **kwargs):
    try:
        return normalizer(*args, **kwargs)
    except ValueError as exc:
        raise CodegenBackendError(str(exc)) from exc


def channels_last_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    if len(shape) != 4:
        raise CodegenBackendError("required rank 4 tensor to use channels_last format")
    batch, channels, height, width = (
        max(shape[0], 1),
        max(shape[1], 1),
        max(shape[2], 1),
        max(shape[3], 1),
    )
    return (
        height * width * channels,
        1,
        width * channels,
        channels,
    )


def channels_last_3d_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    if len(shape) != 5:
        raise CodegenBackendError(
            "required rank 5 tensor to use channels_last_3d format"
        )
    batch, channels, depth, height, width = (
        max(shape[0], 1),
        max(shape[1], 1),
        max(shape[2], 1),
        max(shape[3], 1),
        max(shape[4], 1),
    )
    return (
        depth * height * width * channels,
        1,
        height * width * channels,
        width * channels,
        channels,
    )


def normalize_col2im_output_size(op_name: str, value: object) -> Tuple[int, int]:
    if isinstance(value, torch.Size):
        value = tuple(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CodegenBackendError(
            f"codegen {op_name} expects output_size to be a tuple of two ints"
        )
    try:
        return normalize_int_or_tuple("output_size", value, 2)
    except ValueError as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects output_size to be a tuple of ints"
        ) from exc


def resolve_scalar_arg(
    op_name: str,
    value: object,
    scalar_values: Dict[torch.fx.Node, object],
) -> float | int | bool:
    if isinstance(value, torch.fx.Node):
        if value in scalar_values:
            return _normalize_scalar_value(op_name, scalar_values[value])
        meta_value = value.meta.get("val")
        if meta_value is None:
            meta_value = value.meta.get("example_value")
        if meta_value is not None:
            return _normalize_scalar_value(op_name, meta_value)
        raise CodegenBackendError(f"codegen {op_name} expects a scalar value")
    return _normalize_scalar_value(op_name, value)


def normalize_as_strided_sequence(
    op_name: str,
    value: object,
    arg_name: str,
    *,
    scalar_values: Dict[torch.fx.Node, object] | None = None,
) -> Tuple[int, ...]:
    if isinstance(value, torch.fx.Node):
        if scalar_values is not None and value in scalar_values:
            value = scalar_values[value]
        else:
            for key in ("val", "example_value"):
                if key in value.meta:
                    value = value.meta[key]
                    break
            else:
                raise CodegenBackendError(
                    f"codegen {op_name} expects {arg_name} to be a sequence; "
                    "missing node.meta for keys: val, example_value"
                )
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = (value.item(),)
        else:
            if value.dim() != 1:
                raise CodegenBackendError(
                    f"codegen {op_name} expects {arg_name} to be a 1D tensor"
                )
            value = value.tolist()
    if isinstance(value, torch.Size):
        seq = tuple(value)
    elif isinstance(value, ABCSequence) and not isinstance(value, (str, bytes)):
        seq = value
    else:
        raise CodegenBackendError(
            f"codegen {op_name} expects {arg_name} to be a sequence"
        )
    normalized: list[int] = []
    for item in seq:
        if isinstance(item, torch.fx.Node):
            if scalar_values is not None and item in scalar_values:
                item = scalar_values[item]
        normalized.append(parse_constant_int(op_name, arg_name, item))
    return tuple(normalized)


def normalize_flip_dims(op_name: str, dims: object, rank: int) -> Tuple[int, ...]:
    if dims is None:
        raise CodegenBackendError(f"codegen {op_name} expects dims to be provided")
    if isinstance(dims, torch.fx.Node):
        raise CodegenBackendError(
            f"codegen {op_name} expects dims to be an int or tuple of ints"
        )
    if isinstance(dims, (tuple, list)):
        dims_list = list(dims)
    else:
        dims_list = [dims]
    if not dims_list:
        return ()
    if rank == 0:
        raise CodegenBackendError(
            f"codegen {op_name} expects dims to be within the input rank"
        )
    normalized = []
    seen = set()
    for item in dims_list:
        if isinstance(item, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be an int or tuple of ints"
            )
        try:
            dim = operator.index(item)
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be an int or tuple of ints"
            ) from exc
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be within the input rank"
            )
        if dim in seen:
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be unique"
            )
        seen.add(dim)
        normalized.append(dim)
    return tuple(normalized)


def normalize_reduction_dims(
    op_name: str, dim: object | None, rank: int
) -> Tuple[int, ...]:
    if dim is None:
        return tuple(range(rank))
    if isinstance(dim, torch.fx.Node):
        raise CodegenBackendError(
            f"codegen {op_name} expects dim to be an int or tuple of ints"
        )
    if rank == 0:
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        for item in dims:
            if isinstance(item, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_name} expects dim to be an int or tuple of ints"
                )
            try:
                dim_value = operator.index(item)
            except TypeError as exc:
                raise CodegenBackendError(
                    f"codegen {op_name} expects dim to be an int or tuple of ints"
                ) from exc
            if dim_value not in (-1, 0):
                raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        return ()
    if isinstance(dim, (tuple, list)):
        dims = dim
    else:
        dims = (dim,)
    normalized: list[int] = []
    seen: set[int] = set()
    for item in dims:
        if isinstance(item, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects dim to be an int or tuple of ints"
            )
        try:
            dim_value = operator.index(item)
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects dim to be an int or tuple of ints"
            ) from exc
        if dim_value < 0:
            dim_value += rank
        if dim_value < 0 or dim_value >= rank:
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        if dim_value in seen:
            continue
        seen.add(dim_value)
        normalized.append(dim_value)
    return tuple(sorted(normalized))


def _constant_error_message(
    op_name: str, name: str, expected_type: type, *, node_error: bool = False
) -> str:
    if expected_type is int:
        return f"codegen {op_name} expects {name} to be an int"
    if expected_type is bool:
        return f"codegen {op_name} expects {name} to be a bool"
    if expected_type is float:
        if node_error:
            return f"codegen {op_name} expects {name} to be constant"
        return f"codegen {op_name} expects {name} to be numeric"
    raise ValueError(f"Unsupported expected type: {expected_type}")


def resolve_node_constant(
    value: object,
    expected_type: type,
    *,
    fallback_meta_keys: Tuple[str, ...] = ("val", "example_value"),
    allow_scalar_tensor: bool = False,
    op_name: str,
    name: str,
) -> object:
    type_error_message = _constant_error_message(op_name, name, expected_type)
    node_error_message = _constant_error_message(
        op_name, name, expected_type, node_error=True
    )
    if isinstance(value, torch.fx.Node):
        for key in fallback_meta_keys:
            if key in value.meta:
                value = value.meta[key]
                break
        else:
            keys_list = ", ".join(fallback_meta_keys)
            raise CodegenBackendError(
                f"{node_error_message}; missing node.meta for keys: {keys_list}"
            )
    if allow_scalar_tensor and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(type_error_message)
        value = value.item()
    if expected_type is int:
        if isinstance(value, bool):
            raise CodegenBackendError(type_error_message)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError as exc:
                raise CodegenBackendError(type_error_message) from exc
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise CodegenBackendError(type_error_message)
        try:
            return operator.index(value)
        except TypeError as exc:
            raise CodegenBackendError(type_error_message) from exc
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        raise CodegenBackendError(type_error_message)
    if expected_type is float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        raise CodegenBackendError(type_error_message)
    raise ValueError(f"Unsupported expected type: {expected_type}")


def parse_constant_float(op_name: str, name: str, value: object) -> float:
    return resolve_node_constant(value, float, op_name=op_name, name=name)


def parse_constant_int(op_name: str, name: str, value: object) -> int:
    return resolve_node_constant(
        value,
        int,
        allow_scalar_tensor=True,
        op_name=op_name,
        name=name,
    )


def parse_constant_bool(op_name: str, name: str, value: object) -> bool:
    return resolve_node_constant(
        value,
        bool,
        allow_scalar_tensor=True,
        op_name=op_name,
        name=name,
    )


def parse_bitwise_scalar(op_name: str, value: object, dtype: torch.dtype) -> object:
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise CodegenBackendError(
                f"codegen {op_name} expects scalar to be constant"
            )
        return parse_bitwise_scalar(op_name, meta_value, dtype)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(
                f"codegen {op_name} expects scalar to be a single value"
            )
        return parse_bitwise_scalar(op_name, value.item(), dtype)
    if dtype is torch.bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise CodegenBackendError(
                    f"codegen {op_name} expects scalar to be a boolean value"
                )
            return bool(int(value))
        try:
            return bool(operator.index(value))
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects scalar to be a boolean value"
            ) from exc
    if isinstance(value, bool):
        return int(value)
    return parse_constant_int(op_name, "scalar", value)
