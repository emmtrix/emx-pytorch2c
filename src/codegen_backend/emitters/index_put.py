from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.c_types import _dtype_to_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
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


def _write_index_put_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    index_shapes: Sequence[Sequence[int]],
    index_strides: Sequence[Sequence[int]],
    index_dtypes: Sequence[object],
    values_shape: Sequence[int],
    values_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    accumulate: bool,
) -> List[str]:
    index_put_template = get_template_env().get_template("index_put_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    values_suffix = _format_array_suffix(values_shape)
    signature_parts = [f"const {dtype.c_type} input{input_suffix}"]
    for index, (shape, index_dtype) in enumerate(zip(index_shapes, index_dtypes)):
        index_suffix = _format_array_suffix(shape)
        index_c_type = _dtype_to_c_type(index_dtype, dtype)
        signature_parts.append(
            f"const {index_c_type} index{index}{index_suffix}"
        )
    signature_parts.append(f"const {dtype.c_type} values{values_suffix}")
    signature_parts.append(f"{dtype.c_type} out{output_suffix}")
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{', '.join(signature_parts)}) {{"
    )

    copy_loop_lines, copy_indent = emit_loops(output_shape)
    output_indices = [f"i{dim}" for dim in range(len(output_shape))]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    input_access = _emit_strided_access(
        "input",
        output_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    copy_body_lines = [f"{copy_indent}{output_access} = {input_access};"]
    copy_footer_lines = emit_footer(output_shape, copy_indent)

    index_shape_rank = len(index_shapes[0]) if index_shapes else 0
    input_rank = len(input_shape)
    index_rank = len(index_shapes)
    is_mask = index_rank == 1 and index_dtypes[0] is torch.bool
    update_op = "+=" if accumulate else "="

    if is_mask:
        update_shape = tuple(index_shapes[0]) + tuple(values_shape)
        update_loop_lines, update_indent = emit_loops(update_shape)
        loop_indices = [f"i{dim}" for dim in range(len(update_shape))]
        mask_access = _emit_strided_access(
            "index0",
            loop_indices[:index_shape_rank],
            index_strides[0],
            _is_contiguous(index_shapes[0], index_strides[0]),
            sizes=index_shapes[0],
            c_type=_dtype_to_c_type(index_dtypes[0], dtype),
        )
        output_indices = [
            "i0",
            *[
                f"i{index_shape_rank + dim}"
                for dim in range(input_rank - 1)
            ],
        ]
        output_access = _emit_strided_access(
            "out",
            output_indices,
            output_strides,
            _is_contiguous(output_shape, output_strides),
            sizes=output_shape,
            c_type=dtype.c_type,
        )
        values_access = _emit_strided_access(
            "values",
            loop_indices[index_shape_rank:],
            values_strides,
            _is_contiguous(values_shape, values_strides),
            sizes=values_shape,
            c_type=dtype.c_type,
        )
        update_body_lines = [
            f"{update_indent}if ({mask_access}) {{",
            f"{update_indent}    {output_access} {update_op} {values_access};",
            f"{update_indent}}}",
        ]
        update_footer_lines = emit_footer(update_shape, update_indent)
    else:
        update_loop_lines, update_indent = emit_loops(values_shape)
        loop_indices = [f"i{dim}" for dim in range(len(values_shape))]
        index_accesses = []
        for idx, (shape, strides, index_dtype) in enumerate(
            zip(index_shapes, index_strides, index_dtypes)
        ):
            index_c_type = _dtype_to_c_type(index_dtype, dtype)
            index_accesses.append(
                _emit_strided_access(
                    f"index{idx}",
                    loop_indices[:index_shape_rank],
                    strides,
                    _is_contiguous(shape, strides),
                    sizes=shape,
                    c_type=index_c_type,
                )
            )
        update_body_lines = []
        for dim, access in enumerate(index_accesses):
            update_body_lines.append(
                f"{update_indent}size_t idx{dim} = (size_t)({access});"
            )
            update_body_lines.append(
                f"{update_indent}if (idx{dim} < 0) {{ idx{dim} += {input_shape[dim]}; }}"
            )
        output_indices = []
        for dim in range(input_rank):
            if dim < index_rank:
                output_indices.append(f"idx{dim}")
            else:
                output_indices.append(f"i{index_shape_rank + dim - index_rank}")
        output_access = _emit_strided_access(
            "out",
            output_indices,
            output_strides,
            _is_contiguous(output_shape, output_strides),
            sizes=output_shape,
            c_type=dtype.c_type,
        )
        values_access = _emit_strided_access(
            "values",
            loop_indices,
            values_strides,
            _is_contiguous(values_shape, values_strides),
            sizes=values_shape,
            c_type=dtype.c_type,
        )
        update_body_lines.append(
            f"{update_indent}{output_access} {update_op} {values_access};"
        )
        update_footer_lines = emit_footer(values_shape, update_indent)

    rendered = index_put_template.render(
        signature=signature,
        copy_loop_lines=copy_loop_lines,
        copy_body_lines=copy_body_lines,
        copy_footer_lines=copy_footer_lines,
        update_loop_lines=update_loop_lines,
        update_body_lines=update_body_lines,
        update_footer_lines=update_footer_lines,
    )
    return rendered.splitlines()


class IndexPutEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("index_put requires op spec and dtype")
        index_rank = int(req.params["index_rank"])
        input_shape = req.input_shapes[0]
        input_strides = req.input_strides[0]
        index_shapes = req.input_shapes[1 : 1 + index_rank]
        index_strides = req.input_strides[1 : 1 + index_rank]
        index_dtypes = req.input_dtypes[1 : 1 + index_rank]
        values_shape = req.input_shapes[1 + index_rank]
        values_strides = req.input_strides[1 + index_rank]
        output_strides = req.output_strides or ()
        return _write_index_put_kernel(
            req.node_index,
            op_spec,
            input_shape,
            input_strides,
            index_shapes,
            index_strides,
            index_dtypes,
            values_shape,
            values_strides,
            req.output_shape,
            output_strides,
            dtype,
            accumulate=bool(req.params.get("accumulate", False)),
        )
