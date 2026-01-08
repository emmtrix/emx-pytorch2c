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
    emit_footer,
    emit_loops,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_embedding_bag_kernel(
    node_index: int,
    op_spec: _OpSpec,
    weight_shape: Sequence[int],
    indices_shape: Sequence[int],
    offsets_shape: Sequence[int],
    weight_strides: Sequence[int],
    indices_strides: Sequence[int],
    offsets_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    indices_dtype: torch.dtype,
    offsets_dtype: torch.dtype,
    dtype: _CodegenDType,
    mode: int,
    padding_idx: int,
    include_last_offset: bool,
) -> List[str]:
    embedding_template = get_template_env().get_template("embedding_bag_kernel.c.j2")
    weight_suffix = _format_array_suffix(weight_shape)
    indices_suffix = _format_array_suffix(indices_shape)
    offsets_suffix = _format_array_suffix(offsets_shape)
    out_suffix = _format_array_suffix(output_shape)
    index_c_type = _dtype_to_c_type(indices_dtype, dtype)
    offsets_c_type = _dtype_to_c_type(offsets_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} weight{weight_suffix}, "
        f"const {index_c_type} indices{indices_suffix}, "
        f"const {offsets_c_type} offsets{offsets_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    loop_lines, indent = emit_loops(output_shape)
    output_indices = [f"i{dim}" for dim in range(len(output_shape))]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    offsets_access = _emit_strided_access(
        "offsets",
        ["i0"],
        offsets_strides,
        _is_contiguous(offsets_shape, offsets_strides),
        sizes=offsets_shape,
        c_type=offsets_c_type,
    )
    offsets_next_access = _emit_strided_access(
        "offsets",
        ["i0 + 1"],
        offsets_strides,
        _is_contiguous(offsets_shape, offsets_strides),
        sizes=offsets_shape,
        c_type=offsets_c_type,
    )
    indices_access = _emit_strided_access(
        "indices",
        ["j"],
        indices_strides,
        _is_contiguous(indices_shape, indices_strides),
        sizes=indices_shape,
        c_type=index_c_type,
    )
    weight_access = _emit_strided_access(
        "weight",
        ["idx", "i1"],
        weight_strides,
        _is_contiguous(weight_shape, weight_strides),
        sizes=weight_shape,
        c_type=dtype.c_type,
    )
    zero_literal = _format_scalar_literal(0.0, dtype)
    body_lines = [
        f"{indent}ssize_t start = (ssize_t)({offsets_access});",
    ]
    if include_last_offset:
        body_lines.append(
            f"{indent}ssize_t end = (ssize_t)({offsets_next_access});"
        )
    else:
        body_lines.append(
            f"{indent}ssize_t end = (i0 + 1 < {offsets_shape[0]}) "
            f"? (ssize_t)({offsets_next_access}) "
            f": {indices_shape[0]};"
        )
    body_lines.extend(
        [
            f"{indent}{dtype.c_type} acc = {zero_literal};",
            f"{indent}ssize_t count = 0;",
            f"{indent}for (ssize_t j = start; j < end; ++j) {{",
        ]
    )
    inner_indent = f"{indent}    "
    body_lines.append(
        f"{inner_indent}ssize_t idx = (ssize_t)({indices_access});"
    )
    if padding_idx != -1:
        body_lines.extend(
            [
                f"{inner_indent}if (idx == {padding_idx}) {{",
                f"{inner_indent}    continue;",
                f"{inner_indent}}}",
            ]
        )
    body_lines.extend(
        [
            f"{inner_indent}acc += {weight_access};",
            f"{inner_indent}count += 1;",
            f"{indent}}}",
        ]
    )
    if mode == 1:
        body_lines.extend(
            [
                f"{indent}if (count == 0) {{",
                f"{indent}    {output_access} = {zero_literal};",
                f"{indent}}} else {{",
                f"{indent}    {output_access} = acc / ({dtype.c_type})count;",
                f"{indent}}}",
            ]
        )
    else:
        body_lines.append(f"{indent}{output_access} = acc;")
    footer_lines = emit_footer(output_shape, indent)
    rendered = embedding_template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


class EmbeddingBagEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("embedding_bag requires op spec and dtype")
        return _write_embedding_bag_kernel(
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
            req.input_dtypes[1],
            req.input_dtypes[2],
            dtype,
            mode=int(req.params.get("mode", 0)),
            padding_idx=int(req.params.get("padding_idx", -1)),
            include_last_offset=bool(req.params.get("include_last_offset", False)),
        )
