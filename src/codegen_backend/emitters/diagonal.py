from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    emit_footer,
    emit_loops,
    emit_output_access,
)
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec


def _format_diagonal_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    dim1: int,
    dim2: int,
    offset: int,
    c_type: str,
) -> str:
    output_rank = len(output_shape)
    diag_index = f"i{output_rank - 1}"
    other_indices = [f"i{idx}" for idx in range(output_rank - 1)]
    other_iter = iter(other_indices)
    terms = []
    for dim, stride in enumerate(input_strides):
        if stride == 0:
            continue
        if dim == dim1:
            if offset >= 0:
                index_expr = diag_index
            else:
                index_expr = f"({diag_index} - ({offset}))"
        elif dim == dim2:
            if offset >= 0:
                index_expr = f"({diag_index} + {offset})"
            else:
                index_expr = diag_index
        else:
            index_expr = next(other_iter)
        terms.append(f"{index_expr} * {stride}")
    index_expr = " + ".join(terms) if terms else "0"
    return f"(({c_type}*){name})[{index_expr}]"


def _write_diagonal_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    input_dtype: torch.dtype,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    dim1: int,
    dim2: int,
    offset: int,
) -> List[str]:
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} a{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    lines = [signature]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    input_access = _format_diagonal_access(
        "a",
        input_shape,
        input_strides,
        output_shape,
        dim1=dim1,
        dim2=dim2,
        offset=offset,
        c_type=input_c_type,
    )
    lines.append(f"{indent}{output_access} = {input_access};")
    lines.extend(emit_footer(output_shape, indent))
    return lines


class DiagonalEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("diagonal requires op spec and dtype")
        return _write_diagonal_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.input_dtypes[0],
            req.output_shape,
            req.output_strides or (),
            dtype,
            dim1=int(req.params.get("dim1", 0)),
            dim2=int(req.params.get("dim2", 1)),
            offset=int(req.params.get("offset", 0)),
        )
