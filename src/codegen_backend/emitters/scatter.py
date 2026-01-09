from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.c_types import _dtype_to_c_type, _format_scalar_literal, _input_c_type
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


def _write_scatter_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    index_shape: Sequence[int],
    src_shape: Sequence[int] | None,
    input_strides: Sequence[int],
    index_strides: Sequence[int],
    src_strides: Sequence[int] | None,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    input_dtype: torch.dtype,
    index_dtype: torch.dtype,
    src_dtype: torch.dtype | None,
    dim: int,
    dtype: _CodegenDType,
    *,
    value: float | int | bool | None = None,
) -> List[str]:
    scatter_template = get_template_env().get_template("scatter_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    index_suffix = _format_array_suffix(index_shape)
    output_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    index_c_type = _dtype_to_c_type(index_dtype, dtype)
    signature_parts = [
        f"const {input_c_type} input{input_suffix}",
        f"const {index_c_type} index{index_suffix}",
    ]
    src_c_type = None
    if src_shape is not None and src_dtype is not None:
        src_suffix = _format_array_suffix(src_shape)
        src_c_type = _input_c_type(src_dtype, dtype)
        signature_parts.append(f"const {src_c_type} src{src_suffix}")
    signature_parts.append(f"{dtype.c_type} out{output_suffix}")
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{', '.join(signature_parts)}) {{"
    )

    copy_loop_lines = emit_loops(output_shape)
    output_indices = [f"i{dim}" for dim in range(len(output_shape))]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    input_access = _emit_strided_access(
        "input",
        output_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=input_c_type,
    )
    copy_body_lines = [f"{output_access} = {input_access};"]
    copy_footer_lines = emit_footer(output_shape)

    update_loop_lines = emit_loops(index_shape)
    loop_indices = [f"i{dim}" for dim in range(len(index_shape))]
    index_access = _emit_strided_access(
        "index",
        loop_indices,
        index_strides,
        _is_contiguous(index_shape, index_strides),
        sizes=index_shape,
        c_type=index_c_type,
    )
    update_body_lines = [
        f"ssize_t idx = (ssize_t)({index_access});",
        f"if (idx < 0) {{ idx += {input_shape[dim]}; }}",
    ]
    scatter_indices = []
    for dim_index in range(len(output_shape)):
        if dim_index == dim:
            scatter_indices.append("idx")
        else:
            scatter_indices.append(loop_indices[dim_index])
    scatter_access = _emit_strided_access(
        "out",
        scatter_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    if src_shape is not None and src_strides is not None and src_c_type is not None:
        src_access = _emit_strided_access(
            "src",
            loop_indices,
            src_strides,
            _is_contiguous(src_shape, src_strides),
            sizes=src_shape,
            c_type=src_c_type,
        )
        update_body_lines.append(f"{scatter_access} = {src_access};")
    else:
        if value is None:
            raise CodegenBackendError("scatter value kernel requires a value")
        value_literal = _format_scalar_literal(value, dtype)
        update_body_lines.append(f"{scatter_access} = {value_literal};")
    update_footer_lines = emit_footer(index_shape)

    rendered = scatter_template.render(
        signature=signature,
        copy_loop_lines=copy_loop_lines,
        copy_body_lines=copy_body_lines,
        copy_footer_lines=copy_footer_lines,
        update_loop_lines=update_loop_lines,
        update_body_lines=update_body_lines,
        update_footer_lines=update_footer_lines,
    )
    return rendered.splitlines()


class ScatterEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("scatter requires op spec and dtype")
        has_src = op_spec.name == "scatter_src"
        src_shape = req.input_shapes[2] if has_src else None
        src_strides = req.input_strides[2] if has_src else None
        src_dtype = req.input_dtypes[2] if has_src else None
        return _write_scatter_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            src_shape,
            req.input_strides[0],
            req.input_strides[1],
            src_strides,
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[0],
            req.input_dtypes[1],
            src_dtype,
            int(req.params["dim"]),
            dtype,
            value=req.params.get("value"),
        )
