from __future__ import annotations

from abc import ABC
from typing import Dict, List, Protocol, Sequence, Tuple

import torch

from codegen_backend.c_types import _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.indexing import (
    _contiguous_strides,
    format_input_access,
    format_output_access,
)
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec


def _format_array_suffix(
    shape: Sequence[int],
    dim_names: Dict[int, str] | None = None,
) -> str:
    dim_names = dim_names or {}
    return (
        "".join(
            f"[{dim_names.get(idx, dim)}]" for idx, dim in enumerate(shape)
        )
        or "[1]"
    )


def _format_dim_args(dim_order: Sequence[str]) -> str:
    if not dim_order:
        return ""
    return ", ".join(f"int {dim_name}" for dim_name in dim_order)


def _is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    expected = _contiguous_strides(shape)
    return all(
        size == 1 or stride == expected_stride
        for size, stride, expected_stride in zip(shape, strides, expected)
    )


def emit_signature(
    node_index: int,
    op_spec: _OpSpec,
    output_shape: Sequence[int],
    input_shapes: Sequence[Sequence[int]],
    input_dtypes: Sequence[torch.dtype],
    dtype: _CodegenDType,
    params: Dict[str, object] | None = None,
    *,
    signature_kind: str = "unary",
    input_dim_names: Sequence[Dict[int, str]] | None = None,
    output_dim_names: Dict[int, str] | None = None,
    dim_order: Sequence[str] = (),
) -> str:
    input_dim_names = input_dim_names or []
    output_dim_names = output_dim_names or {}
    dim_args = _format_dim_args(dim_order)
    dim_prefix = f"{dim_args}, " if dim_args else ""
    out_suffix = _format_array_suffix(output_shape, output_dim_names)
    if signature_kind == "binary":
        a_shape, b_shape = input_shapes
        a_suffix = _format_array_suffix(
            a_shape, input_dim_names[0] if input_dim_names else None
        )
        b_suffix = _format_array_suffix(
            b_shape, input_dim_names[1] if len(input_dim_names) > 1 else None
        )
        a_c_type = _input_c_type(input_dtypes[0], dtype)
        b_c_type = _input_c_type(input_dtypes[1], dtype)
        return (
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"{dim_prefix}const {a_c_type} a{a_suffix}, "
            f"const {b_c_type} b{b_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        )
    if signature_kind == "binary_scalar":
        a_shape = input_shapes[0]
        a_suffix = _format_array_suffix(
            a_shape, input_dim_names[0] if input_dim_names else None
        )
        a_c_type = _input_c_type(input_dtypes[0], dtype)
        return (
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"{dim_prefix}const {a_c_type} a{a_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        )
    if signature_kind == "clamp_tensor":
        params = params or {}
        signature_parts = []
        a_shape = input_shapes[0]
        a_suffix = _format_array_suffix(
            a_shape, input_dim_names[0] if input_dim_names else None
        )
        a_c_type = _input_c_type(input_dtypes[0], dtype)
        signature_parts.append(f"const {a_c_type} a{a_suffix}")
        input_index = 1
        if params.get("has_min"):
            min_shape = input_shapes[input_index]
            min_suffix = _format_array_suffix(
                min_shape,
                input_dim_names[input_index]
                if len(input_dim_names) > input_index
                else None,
            )
            min_c_type = _input_c_type(input_dtypes[input_index], dtype)
            signature_parts.append(f"const {min_c_type} min{min_suffix}")
            input_index += 1
        if params.get("has_max"):
            max_shape = input_shapes[input_index]
            max_suffix = _format_array_suffix(
                max_shape,
                input_dim_names[input_index]
                if len(input_dim_names) > input_index
                else None,
            )
            max_c_type = _input_c_type(input_dtypes[input_index], dtype)
            signature_parts.append(f"const {max_c_type} max{max_suffix}")
            input_index += 1
        signature_parts.append(f"{dtype.c_type} out{out_suffix}")
        if dim_prefix:
            signature_parts.insert(0, dim_args)
        return (
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"{', '.join(signature_parts)}) {{"
        )
    if signature_kind == "where":
        params = params or {}
        input_index = 0
        cond_shape = input_shapes[input_index]
        cond_suffix = _format_array_suffix(
            cond_shape, input_dim_names[0] if input_dim_names else None
        )
        cond_c_type = _input_c_type(input_dtypes[input_index], dtype)
        input_index += 1
        signature_parts = [
            f"const {cond_c_type} cond{cond_suffix}",
        ]
        if "a_scalar" not in params:
            a_shape = input_shapes[input_index]
            a_suffix = _format_array_suffix(
                a_shape,
                input_dim_names[input_index]
                if len(input_dim_names) > input_index
                else None,
            )
            a_c_type = _input_c_type(input_dtypes[input_index], dtype)
            signature_parts.append(f"const {a_c_type} a{a_suffix}")
            input_index += 1
        if "b_scalar" not in params:
            b_shape = input_shapes[input_index]
            b_suffix = _format_array_suffix(
                b_shape,
                input_dim_names[input_index]
                if len(input_dim_names) > input_index
                else None,
            )
            b_c_type = _input_c_type(input_dtypes[input_index], dtype)
            signature_parts.append(f"const {b_c_type} b{b_suffix}")
            input_index += 1
        signature_parts.append(f"{dtype.c_type} out{out_suffix}")
        if dim_prefix:
            signature_parts.insert(0, dim_args)
        return (
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"{', '.join(signature_parts)}) {{"
        )
    a_suffix = _format_array_suffix(
        input_shapes[0], input_dim_names[0] if input_dim_names else None
    )
    a_c_type = _input_c_type(input_dtypes[0], dtype)
    return (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{dim_prefix}const {a_c_type} a{a_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )


def emit_loops(
    output_shape: Sequence[int],
    dim_names: Dict[int, str] | None = None,
) -> List[str]:
    lines: List[str] = []
    dim_names = dim_names or {}
    if output_shape:
        for dim, size in enumerate(output_shape):
            size = dim_names.get(dim, size)
            lines.append(
                f"for (ssize_t i{dim} = 0; i{dim} < {size}; ++i{dim}) {{"
            )
    return lines


def _close_loops(loop_count: int) -> List[str]:
    lines: List[str] = []
    for _ in range(loop_count):
        lines.append("}")
    return lines


def emit_output_access(
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    *,
    c_type: str,
) -> str:
    return format_output_access(
        "out",
        output_shape,
        output_strides,
        c_type=c_type,
        output_is_contiguous=_is_contiguous(output_shape, output_strides),
    )


def emit_input_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    broadcast_contiguous: bool,
    c_type: str,
) -> str:
    return format_input_access(
        name,
        input_shape,
        input_strides,
        output_shape,
        broadcast_contiguous=broadcast_contiguous,
        c_type=c_type,
        input_is_contiguous=_is_contiguous(input_shape, input_strides),
    )


def emit_footer(output_shape: Sequence[int]) -> List[str]:
    lines = _close_loops(len(output_shape))
    lines.append("}")
    return lines


def map_reduction_dims(
    input_rank: int, reduction_dims: Sequence[int]
) -> Dict[int, int]:
    dim_to_output: Dict[int, int] = {}
    output_idx = 0
    reduction_set = set(reduction_dims)
    for dim in range(input_rank):
        if dim in reduction_set:
            continue
        dim_to_output[dim] = output_idx
        output_idx += 1
    return dim_to_output


class KindEmitter(Protocol):
    def emit(self, req: KernelEmitRequest) -> List[str]: ...

class KindEmitterBase(ABC):
    emit_signature = staticmethod(emit_signature)
    emit_loops = staticmethod(emit_loops)
    emit_footer = staticmethod(emit_footer)
    emit_input_access = staticmethod(emit_input_access)
    emit_output_access = staticmethod(emit_output_access)
    map_reduction_dims = staticmethod(map_reduction_dims)
