from __future__ import annotations

from typing import List

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _input_c_type
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    emit_footer,
    emit_loops,
    emit_output_access,
)
from codegen_backend.kinds import KernelEmitRequest


class ViewEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        if op_spec is None:
            raise CodegenBackendError("view requires op spec")
        input_shape = req.input_shapes[0]
        input_dtype = req.input_dtypes[0]
        output_shape = req.output_shape
        output_strides = req.output_strides
        input_suffix = _format_array_suffix(input_shape)
        output_suffix = _format_array_suffix(output_shape)
        input_c_type = _input_c_type(input_dtype, req.dtype)
        signature = (
            f"void node{req.node_index}_{op_spec.name}_{req.dtype.suffix}("
            f"const {input_c_type} a{input_suffix}, "
            f"{req.dtype.c_type} out{output_suffix}) {{"
        )
        lines = [signature]
        lines.append(
            f"    const {input_c_type}* a_ptr = (const {input_c_type}*)a;"
        )
        loop_lines, indent = emit_loops(output_shape)
        lines.extend(loop_lines)
        view_strides = req.params.get("view_strides", ())
        storage_offset = int(req.params.get("storage_offset", 0))
        if view_strides:
            offset_terms = [
                f"i{dim} * {stride}"
                for dim, stride in enumerate(view_strides)
            ]
            offset_expr = " + ".join(offset_terms)
        else:
            offset_expr = "0"
        if storage_offset:
            offset_expr = f"{storage_offset} + {offset_expr}"
        lines.append(f"{indent}int64_t offset = {offset_expr};")
        output_access = emit_output_access(
            output_shape, output_strides, c_type=req.dtype.c_type
        )
        lines.append(f"{indent}{output_access} = a_ptr[offset];")
        lines.extend(emit_footer(output_shape, indent))
        return lines
