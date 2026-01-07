from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _input_c_type
from codegen_backend.dtypes import _CODEGEN_DTYPES, _CodegenDType, _INTEGER_CODEGEN_DTYPES
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
    emit_footer,
    emit_loops,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec


def _write_cumsum_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_strides: Sequence[int],
    cumsum_dim: int,
    graph_dtype: _CodegenDType,
    output_dtype: torch.dtype,
) -> List[str]:
    output_dtype_info = _CODEGEN_DTYPES.get(output_dtype)
    if output_dtype_info is None:
        raise CodegenBackendError(
            "codegen cumsum supports only torch.float32, torch.int8, or torch.int32"
        )
    output_c_type = output_dtype_info.c_type
    input_c_type = _input_c_type(graph_dtype.torch_dtype, graph_dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{graph_dtype.suffix}("
        f"const {input_c_type} input{_format_array_suffix(input_shape)}, "
        f"{output_c_type} out{_format_array_suffix(input_shape)}) {{"
    )
    lines = [signature]
    if not input_shape:
        lines.append(f"    out[0] = ({output_c_type})input[0];")
        lines.append("}")
        return lines
    loop_lines, indent = emit_loops(input_shape)
    lines.extend(loop_lines)
    output_access = _emit_strided_access(
        "out",
        [f"i{dim}" for dim in range(len(input_shape))],
        output_strides,
        _is_contiguous(input_shape, output_strides),
        sizes=input_shape,
        c_type=output_dtype_info.c_type,
    )
    acc_init = "0" if output_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    lines.append(f"{indent}{output_dtype_info.c_type} acc = {acc_init};")
    lines.append(
        f"{indent}for (int64_t r{cumsum_dim} = 0; r{cumsum_dim} <= i{cumsum_dim}; ++r{cumsum_dim}) {{"
    )
    inner_indent = f"{indent}    "
    input_indices = [
        f"r{cumsum_dim}" if dim == cumsum_dim else f"i{dim}"
        for dim in range(len(input_shape))
    ]
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=input_c_type,
    )
    lines.append(f"{inner_indent}acc += ({output_c_type}){input_access};")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}{output_access} = acc;")
    lines.extend(emit_footer(input_shape, indent))
    return lines


class CumsumEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        output_dtype = req.params.get("output_dtype")
        if op_spec is None or dtype is None or output_dtype is None:
            raise CodegenBackendError("cumsum requires op spec, dtype, and output dtype")
        return _write_cumsum_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_strides or (),
            int(req.params["dim"]),
            dtype,
            output_dtype,
        )
