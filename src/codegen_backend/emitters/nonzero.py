from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.c_types import _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
    emit_footer,
    emit_loops,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _emit_nonzero_input_access(
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    *,
    input_c_type: str,
    input_is_contiguous: bool,
) -> str:
    if not input_shape:
        if input_is_contiguous:
            return "input[0]"
        return f"(({input_c_type}*)input)[0]"
    input_indices = [f"i{dim}" for dim in range(len(input_shape))]
    return _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        input_is_contiguous,
        sizes=input_shape,
        c_type=input_c_type,
    )


def _emit_nonzero_output_access(
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dim_index: int,
    *,
    output_is_contiguous: bool,
) -> str:
    output_indices = ["out_index", str(dim_index)]
    return _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        output_is_contiguous,
        sizes=output_shape,
        c_type="int64_t",
    )


def _write_nonzero_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    input_dtype: torch.dtype,
    dtype: _CodegenDType,
) -> List[str]:
    nonzero_template = get_template_env().get_template("nonzero_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    out_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} input{input_suffix}, "
        f"int64_t out{out_suffix}) {{"
    )
    loop_lines, indent = emit_loops(input_shape)
    loop_lines = ["    size_t out_index = 0;"] + loop_lines
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    input_access = _emit_nonzero_input_access(
        input_shape,
        input_strides,
        input_c_type=input_c_type,
        input_is_contiguous=input_is_contiguous,
    )
    body_lines = [f"{indent}if ({input_access} != 0) {{"]
    inner_indent = f"{indent}    "
    for dim in range(len(input_shape)):
        output_access = _emit_nonzero_output_access(
            output_shape,
            output_strides,
            dim,
            output_is_contiguous=output_is_contiguous,
        )
        body_lines.append(
            f"{inner_indent}{output_access} = (int64_t)i{dim};"
        )
    body_lines.append(f"{inner_indent}out_index += 1;")
    body_lines.append(f"{indent}}}")
    footer_lines = emit_footer(input_shape, indent)
    rendered = nonzero_template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


class NonzeroEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("nonzero requires op spec and dtype")
        return _write_nonzero_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[0],
            dtype,
        )
