from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.c_types import _dtype_to_c_type, _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _close_loops,
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


def _write_embedding_dense_backward_kernel(
    node_index: int,
    op_spec: _OpSpec,
    grad_output_shape: Sequence[int],
    indices_shape: Sequence[int],
    grad_output_strides: Sequence[int],
    indices_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    indices_dtype: torch.dtype,
    dtype: _CodegenDType,
    padding_idx: int,
) -> List[str]:
    embedding_template = get_template_env().get_template(
        "embedding_dense_backward_kernel.c.j2"
    )
    grad_suffix = _format_array_suffix(grad_output_shape)
    indices_suffix = _format_array_suffix(indices_shape)
    out_suffix = _format_array_suffix(output_shape)
    index_c_type = _dtype_to_c_type(indices_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} grad_output{grad_suffix}, "
        f"const {index_c_type} indices{indices_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    init_loop_lines = emit_loops(output_shape)
    init_output_indices = [f"i{dim}" for dim in range(len(output_shape))]
    init_output_access = _emit_strided_access(
        "out",
        init_output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    zero_literal = _format_scalar_literal(0.0, dtype)
    init_body_lines = [f"{init_output_access} = {zero_literal};"]
    init_footer_lines = _close_loops(len(output_shape))

    grad_loop_lines = emit_loops(grad_output_shape)
    grad_output_indices = [f"i{dim}" for dim in range(len(grad_output_shape))]
    grad_output_access = _emit_strided_access(
        "grad_output",
        grad_output_indices,
        grad_output_strides,
        _is_contiguous(grad_output_shape, grad_output_strides),
        sizes=grad_output_shape,
        c_type=dtype.c_type,
    )
    index_indices = [f"i{dim}" for dim in range(len(indices_shape))]
    index_access = _emit_strided_access(
        "indices",
        index_indices,
        indices_strides,
        _is_contiguous(indices_shape, indices_strides),
        sizes=indices_shape,
        c_type=index_c_type,
    )
    output_indices = ["idx", f"i{len(grad_output_shape) - 1}"]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    grad_body_lines = [f"ssize_t idx = (ssize_t)({index_access});"]
    if padding_idx != -1:
        grad_body_lines.extend(
            [
                f"if (idx == {padding_idx}) {{",
                "continue;",
                "}",
            ]
        )
    grad_body_lines.append(f"{output_access} += {grad_output_access};")
    grad_footer_lines = emit_footer(grad_output_shape)
    rendered = embedding_template.render(
        signature=signature,
        init_loop_lines=init_loop_lines,
        init_body_lines=init_body_lines,
        init_footer_lines=init_footer_lines,
        grad_loop_lines=grad_loop_lines,
        grad_body_lines=grad_body_lines,
        grad_footer_lines=grad_footer_lines,
    )
    return rendered.splitlines()


class EmbeddingDenseBackwardEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError(
                "embedding_dense_backward requires op spec and dtype"
            )
        return _write_embedding_dense_backward_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.input_strides[0],
            req.input_strides[1],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[1],
            dtype,
            padding_idx=int(req.params.get("padding_idx", -1)),
        )
