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
        output_suffix = _format_array_suffix(output_shape)
        input_args = []
        input_names = []
        for idx, (shape, dtype) in enumerate(
            zip(req.input_shapes, req.input_dtypes)
        ):
            name = "a" if idx == 0 else f"in_{idx}"
            input_names.append(name)
            suffix = _format_array_suffix(shape)
            c_type = _input_c_type(dtype, req.dtype)
            input_args.append(f"const {c_type} {name}{suffix}")
        signature = (
            f"void node{req.node_index}_{op_spec.name}_{req.dtype.suffix}("
            f"{', '.join([*input_args, f'{req.dtype.c_type} out{output_suffix}'])}) {{"
        )
        lines = [signature]
        input_c_type = _input_c_type(input_dtype, req.dtype)
        lines.append(
            f"const {input_c_type}* a_ptr = (const {input_c_type}*){input_names[0]};"
        )
        loop_lines = emit_loops(output_shape)
        lines.extend(loop_lines)
        view_strides = req.params.get("view_strides", ())
        view_stride_input_index = req.params.get("view_strides_input_index")
        storage_offset = int(req.params.get("storage_offset", 0))
        if view_stride_input_index is not None:
            stride_name = input_names[int(view_stride_input_index)]
            stride_c_type = _input_c_type(
                req.input_dtypes[int(view_stride_input_index)], req.dtype
            )
            lines.append(
                f"const {stride_c_type}* view_strides_ptr = "
                f"(const {stride_c_type}*){stride_name};"
            )
            offset_terms = [
                f"(ssize_t)i{dim} * (ssize_t)view_strides_ptr[{dim}]"
                for dim in range(len(output_shape))
            ]
            offset_expr = " + ".join(offset_terms) if offset_terms else "0"
        elif view_strides:
            offset_terms = [
                f"(ssize_t)i{dim} * (ssize_t){stride}"
                for dim, stride in enumerate(view_strides)
            ]
            offset_expr = " + ".join(offset_terms)
        else:
            offset_expr = "0"
        if storage_offset:
            offset_expr = f"(ssize_t){storage_offset} + {offset_expr}"
        lines.append(f"ssize_t offset = {offset_expr};")
        output_access = emit_output_access(
            output_shape, output_strides, c_type=req.dtype.c_type
        )
        lines.append(f"{output_access} = a_ptr[offset];")
        lines.extend(emit_footer(output_shape))
        return lines
