from __future__ import annotations

from typing import Dict, List

from codegen_backend.c_types import _input_c_type
from codegen_backend.emitters.base import (
    _close_loops,
    _format_array_suffix,
    _is_contiguous,
    KindEmitterBase,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest


class ArgReductionEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        input_shape = req.input_shapes[0]
        input_strides = req.input_strides[0]
        output_shape = req.output_shape
        output_strides = req.output_strides
        reduction_dims = req.reduction_dims or ()
        keepdim = bool(req.keepdim)
        reduce_all = bool(req.params.get("reduce_all", False))
        input_c_type = _input_c_type(req.dtype.torch_dtype, req.dtype)
        signature = (
            f"void node{req.node_index}_{req.op_spec.name}_{req.dtype.suffix}("
            f"const {input_c_type} a{_format_array_suffix(input_shape)}, "
            f"int64_t out{_format_array_suffix(output_shape)}) {{"
        )
        lines = [signature]
        loop_lines, indent = KindEmitterBase.emit_loops(output_shape)
        lines.extend(loop_lines)
        output_access = KindEmitterBase.emit_output_access(
            output_shape, output_strides, c_type="int64_t"
        )
        if not input_shape:
            lines.append(f"{indent}{output_access} = 0;")
            lines.extend(KindEmitterBase.emit_footer(output_shape, indent))
            return lines
        a_is_contiguous = _is_contiguous(input_shape, input_strides)
        compare_op = ">" if req.op_spec.name == "argmax" else "<"

        def linear_index_expr() -> str:
            expr = "r0" if input_shape else "0"
            for dim in range(1, len(input_shape)):
                expr = f"({expr} * {input_shape[dim]} + r{dim})"
            return expr

        if reduce_all:
            lines.append(f"{indent}bool has_value = false;")
            lines.append(f"{indent}{input_c_type} best_value = 0;")
            lines.append(f"{indent}int64_t best_index = 0;")
            reduction_indent = indent
            for dim, size in enumerate(input_shape):
                lines.append(
                    f"{reduction_indent}for (int64_t r{dim} = 0; r{dim} < {size}; ++r{dim}) {{"
                )
                reduction_indent += "    "
            input_access = _emit_strided_access(
                "a",
                [f"r{dim}" for dim in range(len(input_shape))],
                input_strides,
                contig=a_is_contiguous,
                sizes=input_shape,
                c_type=input_c_type,
            )
            lines.append(
                f"{reduction_indent}int64_t linear_index = {linear_index_expr()};"
            )
            lines.append(
                f"{reduction_indent}{input_c_type} value = {input_access};"
            )
            lines.append(f"{reduction_indent}if (!has_value) {{")
            lines.append(f"{reduction_indent}    best_value = value;")
            lines.append(f"{reduction_indent}    best_index = linear_index;")
            lines.append(f"{reduction_indent}    has_value = true;")
            lines.append(
                f"{reduction_indent}}} else if (value {compare_op} best_value) {{"
            )
            lines.append(f"{reduction_indent}    best_value = value;")
            lines.append(f"{reduction_indent}    best_index = linear_index;")
            lines.append(f"{reduction_indent}}}")
            close_lines, _ = _close_loops(len(input_shape), reduction_indent)
            lines.extend(close_lines)
            lines.append(f"{indent}{output_access} = best_index;")
            lines.extend(KindEmitterBase.emit_footer(output_shape, indent))
            return lines

        reduction_dim = reduction_dims[0]
        dim_to_output: Dict[int, int] = {}
        if not keepdim:
            dim_to_output = KindEmitterBase.map_reduction_dims(
                len(input_shape), (reduction_dim,)
            )
        init_indices = []
        loop_indices = []
        for dim in range(len(input_shape)):
            if dim == reduction_dim:
                init_indices.append("0")
                loop_indices.append(f"r{dim}")
            else:
                idx = f"i{dim}" if keepdim else f"i{dim_to_output[dim]}"
                init_indices.append(idx)
                loop_indices.append(idx)
        init_access = _emit_strided_access(
            "a",
            init_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=input_c_type,
        )
        loop_access = _emit_strided_access(
            "a",
            loop_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=input_c_type,
        )
        lines.append(f"{indent}{input_c_type} best_value = {init_access};")
        lines.append(f"{indent}int64_t best_index = 0;")
        lines.append(
            f"{indent}for (int64_t r{reduction_dim} = 1; r{reduction_dim} < {input_shape[reduction_dim]}; ++r{reduction_dim}) {{"
        )
        lines.append(f"{indent}    {input_c_type} value = {loop_access};")
        lines.append(f"{indent}    if (value {compare_op} best_value) {{")
        lines.append(f"{indent}        best_value = value;")
        lines.append(f"{indent}        best_index = r{reduction_dim};")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        lines.append(f"{indent}{output_access} = best_index;")
        lines.extend(KindEmitterBase.emit_footer(output_shape, indent))
        return lines
