from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _dtype_to_c_type
from codegen_backend.dtypes import _CodegenDType
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
from codegen_backend.templates import get_template_env


def _write_gather_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    index_shape: Sequence[int],
    input_strides: Sequence[int],
    index_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    index_dtype: torch.dtype,
    gather_dim: int,
    dtype: _CodegenDType,
) -> List[str]:
    gather_template = get_template_env().get_template("gather_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    index_suffix = _format_array_suffix(index_shape)
    out_suffix = _format_array_suffix(output_shape)
    index_c_type = _dtype_to_c_type(index_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {index_c_type} index{index_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    loop_lines, indent = emit_loops(output_shape)
    output_indices = [f"i{dim}" for dim in range(len(output_shape))]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    index_access = _emit_strided_access(
        "index",
        output_indices,
        index_strides,
        _is_contiguous(index_shape, index_strides),
        sizes=index_shape,
        c_type=index_c_type,
    )
    input_indices = [
        "idx" if dim == gather_dim else f"i{dim}"
        for dim in range(len(input_shape))
    ]
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    body_lines = [
        f"{indent}size_t idx = (size_t)({index_access});",
        f"{indent}{output_access} = {input_access};",
    ]
    footer_lines = emit_footer(output_shape, indent)
    rendered = gather_template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


class GatherEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("gather requires op spec and dtype")
        return _write_gather_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.input_strides[0],
            req.input_strides[1],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[1],
            int(req.params["dim"]),
            dtype,
        )
