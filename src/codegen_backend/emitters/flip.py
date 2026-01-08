from __future__ import annotations

from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _input_c_type
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _is_contiguous,
    emit_footer,
    emit_loops,
    emit_output_access,
    emit_signature,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest


def _emit_flip_input_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    flip_dims: Sequence[int],
    *,
    c_type: str,
) -> str:
    if not input_shape:
        return f"(({c_type}*){name})[0]"
    indices = []
    flip_dim_set = set(flip_dims)
    for dim, size in enumerate(input_shape):
        if dim in flip_dim_set:
            indices.append(f"({size - 1} - i{dim})")
        else:
            indices.append(f"i{dim}")
    return _emit_strided_access(
        name,
        indices,
        input_strides,
        contig=_is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=c_type,
    )


class FlipEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        if op_spec is None:
            raise CodegenBackendError("flip requires op spec")
        input_shape = req.input_shapes[0]
        input_strides = req.input_strides[0]
        input_dtype = req.input_dtypes[0]
        output_shape = req.output_shape
        output_strides = req.output_strides
        lines = [
            emit_signature(
                req.node_index,
                op_spec,
                output_shape,
                [input_shape],
                [input_dtype],
                req.dtype,
                signature_kind="unary",
                input_dim_names=req.input_dim_names,
                output_dim_names=req.output_dim_names,
                dim_order=req.dim_order,
            )
        ]
        loop_lines, indent = emit_loops(output_shape, req.output_dim_names)
        lines.extend(loop_lines)
        output_access = emit_output_access(
            output_shape, output_strides, c_type=req.dtype.c_type
        )
        input_access = _emit_flip_input_access(
            "a",
            input_shape,
            input_strides,
            req.params.get("dims", ()),
            c_type=_input_c_type(input_dtype, req.dtype),
        )
        lines.append(f"{indent}{output_access} = {input_access};")
        lines.extend(emit_footer(output_shape, indent))
        return lines
