from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.c_types import _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
    emit_input_access,
    emit_output_access,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_masked_scatter_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    mask_shape: Sequence[int],
    source_shape: Sequence[int],
    input_strides: Sequence[int],
    mask_strides: Sequence[int],
    source_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    input_dtype: torch.dtype,
    mask_dtype: torch.dtype,
    source_dtype: torch.dtype,
    dtype: _CodegenDType,
) -> List[str]:
    template = get_template_env().get_template("masked_scatter_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    mask_suffix = _format_array_suffix(mask_shape)
    source_suffix = _format_array_suffix(source_shape)
    out_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    mask_c_type = _input_c_type(mask_dtype, dtype)
    source_c_type = _input_c_type(source_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} input{input_suffix}, "
        f"const {mask_c_type} mask{mask_suffix}, "
        f"const {source_c_type} source{source_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
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
    mask_access = emit_input_access(
        "mask",
        mask_shape,
        mask_strides,
        output_shape,
        broadcast_contiguous=False,
        c_type=mask_c_type,
    )
    source_is_contiguous = _is_contiguous(source_shape, source_strides)
    source_indices = [f"src_i{dim}" for dim in range(len(source_shape))]
    source_access = _emit_strided_access(
        "source",
        source_indices,
        source_strides,
        source_is_contiguous,
        sizes=source_shape,
        c_type=source_c_type,
    )
    rendered = template.render(
        signature=signature,
        source_shape=source_shape,
        output_shape=output_shape,
        mask_access=mask_access,
        input_access=input_access,
        output_access=output_access,
        source_access=source_access,
        source_is_contiguous=source_is_contiguous,
    )
    return rendered.splitlines()


class MaskedScatterEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("masked_scatter requires op spec and dtype")
        return _write_masked_scatter_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.input_shapes[2],
            req.input_strides[0],
            req.input_strides[1],
            req.input_strides[2],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[0],
            req.input_dtypes[1],
            req.input_dtypes[2],
            dtype,
        )
