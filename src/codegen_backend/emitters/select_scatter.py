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
    emit_input_access,
    emit_loops,
    emit_output_access,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_select_scatter_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    src_shape: Sequence[int],
    input_strides: Sequence[int],
    src_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    input_dtype: torch.dtype,
    src_dtype: torch.dtype,
    select_dim: int,
    select_index: int,
    dtype: _CodegenDType,
) -> List[str]:
    template = get_template_env().get_template("select_scatter_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    src_suffix = _format_array_suffix(src_shape)
    out_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    src_c_type = _input_c_type(src_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} input{input_suffix}, "
        f"const {src_c_type} src{src_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    loop_lines, indent = emit_loops(output_shape)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    input_access = emit_input_access(
        "input",
        input_shape,
        input_strides,
        output_shape,
        broadcast_contiguous=False,
        c_type=input_c_type,
    )
    if not src_shape:
        src_access = "src[0]"
    else:
        src_indices = [
            f"i{dim}" for dim in range(len(output_shape)) if dim != select_dim
        ]
        src_access = _emit_strided_access(
            "src",
            src_indices,
            src_strides,
            _is_contiguous(src_shape, src_strides),
            sizes=src_shape,
            c_type=src_c_type,
        )
    body_lines = [
        f"{indent}if (i{select_dim} == {select_index}) {{",
        f"{indent}    {output_access} = {src_access};",
        f"{indent}}} else {{",
        f"{indent}    {output_access} = {input_access};",
        f"{indent}}}",
    ]
    footer_lines = emit_footer(output_shape, indent)
    rendered = template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


class SelectScatterEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("select_scatter requires op spec and dtype")
        return _write_select_scatter_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.input_strides[0],
            req.input_strides[1],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[0],
            req.input_dtypes[1],
            int(req.params["dim"]),
            int(req.params["index"]),
            dtype,
        )
