from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _dtype_to_c_type, _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_embedding_kernel(
    node_index: int,
    op_spec: _OpSpec,
    weight_shape: Sequence[int],
    indices_shape: Sequence[int],
    weight_strides: Sequence[int],
    indices_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    indices_dtype: torch.dtype,
    dtype: _CodegenDType,
    padding_idx: int,
) -> List[str]:
    embedding_template = get_template_env().get_template("embedding_kernel.c.j2")
    weight_suffix = _format_array_suffix(weight_shape)
    indices_suffix = _format_array_suffix(indices_shape)
    out_suffix = _format_array_suffix(output_shape)
    index_c_type = _dtype_to_c_type(indices_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} weight{weight_suffix}, "
        f"const {index_c_type} indices{indices_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    indices_rank = len(indices_shape)
    output_rank = len(output_shape)
    output_indices = [f"i{dim}" for dim in range(output_rank)]
    output_access_expr = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    output_access = {
        "expr": output_access_expr,
        "sizes": list(output_shape),
    }
    index_indices = [f"i{dim}" for dim in range(indices_rank)]
    index_access = _emit_strided_access(
        "indices",
        index_indices,
        indices_strides,
        _is_contiguous(indices_shape, indices_strides),
        sizes=indices_shape,
        c_type=index_c_type,
    )
    weight_access = _emit_strided_access(
        "weight",
        ["idx", f"i{output_rank - 1}"],
        weight_strides,
        _is_contiguous(weight_shape, weight_strides),
        sizes=weight_shape,
        c_type=dtype.c_type,
    )
    zero_literal = _format_scalar_literal(0.0, dtype) if padding_idx != -1 else None
    rendered = embedding_template.render(
        signature=signature,
        output_access=output_access,
        index_access=index_access,
        weight_access=weight_access,
        padding_idx=padding_idx,
        zero_literal=zero_literal,
    )
    return rendered.splitlines()


class EmbeddingEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("embedding requires op spec and dtype")
        return _write_embedding_kernel(
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
