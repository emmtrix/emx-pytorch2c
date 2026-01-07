from __future__ import annotations

from typing import List

from codegen_backend.c_types import _format_scalar_literal
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    emit_footer,
    emit_loops,
    emit_output_access,
)
from codegen_backend.kinds import KernelEmitRequest


class ArangeEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_node = req.op_node
        out_suffix = _format_array_suffix(req.output_shape)
        signature = (
            f"void node{req.node_index}_{op_node.spec.name}_{req.dtype.suffix}("
            f"{req.dtype.c_type} out{out_suffix}) {{"
        )
        lines = [signature]
        loop_lines, indent = emit_loops(req.output_shape)
        lines.extend(loop_lines)
        output_access = emit_output_access(
            req.output_shape, req.output_strides, c_type=req.dtype.c_type
        )
        start = _format_scalar_literal(op_node.p("start"), req.dtype)
        step = _format_scalar_literal(op_node.p("step"), req.dtype)
        index_expr = "i0" if req.output_shape else "0"
        lines.append(
            f"{indent}{output_access} = {start} + ({step} * {index_expr});"
        )
        lines.extend(emit_footer(req.output_shape, indent))
        return lines
