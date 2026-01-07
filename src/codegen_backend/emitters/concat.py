from __future__ import annotations

from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _close_loops,
    _format_array_suffix,
    _is_contiguous,
    emit_loops,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec


def _write_concat_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shapes: Sequence[Sequence[int]],
    input_strides: Sequence[Sequence[int]],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    concat_dim: int,
    dtype: _CodegenDType,
) -> List[str]:
    input_args = []
    for idx, shape in enumerate(input_shapes):
        suffix = _format_array_suffix(shape)
        input_args.append(f"const {dtype.c_type} a{idx}{suffix}")
    out_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{', '.join(input_args)}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    lines = [signature]
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    offset = 0
    for idx, (shape, strides) in enumerate(zip(input_shapes, input_strides)):
        loop_lines, indent = emit_loops(shape)
        lines.extend(loop_lines)
        indices = [f"i{dim}" for dim in range(len(shape))]
        input_access = _emit_strided_access(
            f"a{idx}",
            indices,
            strides,
            _is_contiguous(shape, strides),
            sizes=shape,
            c_type=dtype.c_type,
        )
        output_indices = [
            f"i{dim} + {offset}" if dim == concat_dim else f"i{dim}"
            for dim in range(len(shape))
        ]
        output_access = _emit_strided_access(
            "out",
            output_indices,
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        )
        lines.append(f"{indent}{output_access} = {input_access};")
        close_lines, indent = _close_loops(len(shape), indent)
        lines.extend(close_lines)
        offset += shape[concat_dim]
    lines.append("}")
    return lines


class ConcatEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("concat requires op spec and dtype")
        return _write_concat_kernel(
            req.node_index,
            op_spec,
            req.input_shapes,
            req.input_strides,
            req.output_shape,
            req.output_strides or (),
            int(req.params.get("dim", 0)),
            dtype,
        )
