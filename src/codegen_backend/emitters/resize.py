from __future__ import annotations

from typing import List

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.c_types import _input_c_type
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    emit_footer,
    emit_loops,
    emit_output_access,
)
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import KernelEmitRequest


class ResizeEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        if op_spec is None:
            raise RefBackendError("resize requires op spec")
        input_shape = req.input_shapes[0]
        input_strides = req.input_strides[0]
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
        output_contig_strides = _contiguous_strides(output_shape)
        if output_contig_strides:
            linear_terms = [
                f"i{dim} * {stride}"
                for dim, stride in enumerate(output_contig_strides)
            ]
            linear_expr = " + ".join(linear_terms)
        else:
            linear_expr = "0"
        lines.append(f"{indent}int64_t linear = {linear_expr};")
        if input_shape:
            lines.append(f"{indent}int64_t remaining = linear;")
            index_vars = []
            for dim in range(len(input_shape) - 1, -1, -1):
                size = input_shape[dim]
                index_name = f"idx{dim}"
                lines.append(
                    f"{indent}int64_t {index_name} = remaining % {size};"
                )
                if dim != 0:
                    lines.append(f"{indent}remaining /= {size};")
                index_vars.append(index_name)
            index_vars.reverse()
            offset_terms = [
                f"{index_vars[dim]} * {stride}"
                for dim, stride in enumerate(input_strides)
            ]
            offset_expr = " + ".join(offset_terms) if offset_terms else "0"
        else:
            offset_expr = "0"
        lines.append(f"{indent}int64_t offset = {offset_expr};")
        output_access = emit_output_access(
            output_shape, output_strides, c_type=req.dtype.c_type
        )
        lines.append(f"{indent}{output_access} = a_ptr[offset];")
        lines.extend(emit_footer(output_shape, indent))
        return lines
