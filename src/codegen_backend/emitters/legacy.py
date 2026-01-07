from __future__ import annotations

import math
from typing import List, Sequence, Tuple, TYPE_CHECKING

import torch

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.c_types import (
    _dtype_to_c_type,
    _format_scalar_literal,
    _input_c_type,
)
from codegen_backend.dtypes import (
    _CODEGEN_DTYPES,
    _CodegenDType,
    _INTEGER_CODEGEN_DTYPES,
)
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _close_loops,
    _format_array_suffix,
    _is_contiguous,
    emit_footer,
    emit_loops,
    emit_output_access,
)
from codegen_backend.indexing import (
    _emit_strided_access,
    _format_output_access,
)
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env

if TYPE_CHECKING:
    from codegen_backend.graph import _OpNode


def _format_diagonal_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    dim1: int,
    dim2: int,
    offset: int,
    c_type: str,
) -> str:
    output_rank = len(output_shape)
    diag_index = f"i{output_rank - 1}"
    other_indices = [f"i{idx}" for idx in range(output_rank - 1)]
    other_iter = iter(other_indices)
    terms = []
    for dim, stride in enumerate(input_strides):
        if stride == 0:
            continue
        if dim == dim1:
            if offset >= 0:
                index_expr = diag_index
            else:
                index_expr = f"({diag_index} - ({offset}))"
        elif dim == dim2:
            if offset >= 0:
                index_expr = f"({diag_index} + {offset})"
            else:
                index_expr = diag_index
        else:
            index_expr = next(other_iter)
        terms.append(f"{index_expr} * {stride}")
    index_expr = " + ".join(terms) if terms else "0"
    return f"(({c_type}*){name})[{index_expr}]"


def _write_empty_strided_kernel(
    node_index: int,
    op_spec: _OpSpec,
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    return [signature, "    (void)out;", "}"]


def _write_addmm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    mat1_shape: Sequence[int],
    mat2_shape: Sequence[int],
    input_strides: Sequence[int],
    mat1_strides: Sequence[int],
    mat2_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addmm_template = get_template_env().get_template("addmm_kernel.c.j2")
    mat1_is_contiguous = _is_contiguous(mat1_shape, mat1_strides)
    mat2_is_contiguous = _is_contiguous(mat2_shape, mat2_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    m, k = mat1_shape
    _, n = mat2_shape
    input_suffix = _format_array_suffix(input_shape)
    mat1_suffix = _format_array_suffix(mat1_shape)
    mat2_suffix = _format_array_suffix(mat2_shape)
    out_suffix = _format_array_suffix(output_shape)
    output_indices = ("i", "j")
    offset = len(output_indices) - len(input_shape)
    input_indices = output_indices[offset:] if input_shape else ()
    rendered = addmm_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} mat1{mat1_suffix}, "
            f"const {dtype.c_type} mat2{mat2_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        m=m,
        n=n,
        k=k,
        acc_type=acc_type,
        acc_init=acc_init,
        input_access=_emit_strided_access(
            "input",
            input_indices,
            input_strides,
            False,
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
        mat1_access=_emit_strided_access(
            "mat1",
            ("i", "t"),
            mat1_strides,
            mat1_is_contiguous,
            sizes=mat1_shape,
            c_type=dtype.c_type,
        ),
        mat2_access=_emit_strided_access(
            "mat2",
            ("t", "j"),
            mat2_strides,
            mat2_is_contiguous,
            sizes=mat2_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i", "j"),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
    )
    return rendered.strip().splitlines()


def _write_addbmm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    batch1_shape: Sequence[int],
    batch2_shape: Sequence[int],
    input_strides: Sequence[int],
    batch1_strides: Sequence[int],
    batch2_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addbmm_template = get_template_env().get_template("addbmm_kernel.c.j2")
    batch1_is_contiguous = _is_contiguous(batch1_shape, batch1_strides)
    batch2_is_contiguous = _is_contiguous(batch2_shape, batch2_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    batch, m, k = batch1_shape
    _, _, n = batch2_shape
    input_suffix = _format_array_suffix(input_shape)
    batch1_suffix = _format_array_suffix(batch1_shape)
    batch2_suffix = _format_array_suffix(batch2_shape)
    out_suffix = _format_array_suffix(output_shape)
    output_indices = ("i", "j")
    offset = len(output_indices) - len(input_shape)
    input_indices = output_indices[offset:] if input_shape else ()
    rendered = addbmm_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} batch1{batch1_suffix}, "
            f"const {dtype.c_type} batch2{batch2_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        batch=batch,
        m=m,
        n=n,
        k=k,
        acc_type=acc_type,
        acc_init=acc_init,
        input_access=_emit_strided_access(
            "input",
            input_indices,
            input_strides,
            False,
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
        batch1_access=_emit_strided_access(
            "batch1",
            ("b_idx", "i", "t"),
            batch1_strides,
            batch1_is_contiguous,
            sizes=batch1_shape,
            c_type=dtype.c_type,
        ),
        batch2_access=_emit_strided_access(
            "batch2",
            ("b_idx", "t", "j"),
            batch2_strides,
            batch2_is_contiguous,
            sizes=batch2_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i", "j"),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
    )
    return rendered.strip().splitlines()


def _write_addmv_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    mat_shape: Sequence[int],
    vec_shape: Sequence[int],
    input_strides: Sequence[int],
    mat_strides: Sequence[int],
    vec_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addmv_template = get_template_env().get_template("addmv_kernel.c.j2")
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    mat_is_contiguous = _is_contiguous(mat_shape, mat_strides)
    vec_is_contiguous = _is_contiguous(vec_shape, vec_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    m, n = mat_shape
    input_suffix = _format_array_suffix(input_shape)
    mat_suffix = _format_array_suffix(mat_shape)
    vec_suffix = _format_array_suffix(vec_shape)
    out_suffix = _format_array_suffix(output_shape)
    broadcast_input = input_shape != output_shape
    input_access = _emit_strided_access(
        "input",
        ("i",),
        input_strides,
        contig=input_is_contiguous and not broadcast_input,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    rendered = addmv_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} mat{mat_suffix}, "
            f"const {dtype.c_type} vec{vec_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        m=m,
        n=n,
        acc_type=acc_type,
        acc_init=acc_init,
        input_access=input_access,
        mat_access=_emit_strided_access(
            "mat",
            ("i", "t"),
            mat_strides,
            mat_is_contiguous,
            sizes=mat_shape,
            c_type=dtype.c_type,
        ),
        vec_access=_emit_strided_access(
            "vec",
            ("t",),
            vec_strides,
            vec_is_contiguous,
            sizes=vec_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i",),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
    )
    return rendered.strip().splitlines()


def _write_addr_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    vec1_shape: Sequence[int],
    vec2_shape: Sequence[int],
    input_strides: Sequence[int],
    vec1_strides: Sequence[int],
    vec2_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addr_template = get_template_env().get_template("addr_kernel.c.j2")
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    vec1_is_contiguous = _is_contiguous(vec1_shape, vec1_strides)
    vec2_is_contiguous = _is_contiguous(vec2_shape, vec2_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    m, n = output_shape
    input_suffix = _format_array_suffix(input_shape)
    vec1_suffix = _format_array_suffix(vec1_shape)
    vec2_suffix = _format_array_suffix(vec2_shape)
    out_suffix = _format_array_suffix(output_shape)
    skip_input = math.isclose(beta, 0.0)
    if not input_shape:
        input_access = f"(({dtype.c_type}*)input)[0]"
    else:
        input_rank = len(input_shape)
        input_indices = ("i", "j") if input_rank == 2 else ("j",)
        use_contig = input_is_contiguous and input_shape == output_shape
        input_access = _emit_strided_access(
            "input",
            input_indices,
            input_strides,
            use_contig,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    rendered = addr_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} vec1{vec1_suffix}, "
            f"const {dtype.c_type} vec2{vec2_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        m=m,
        n=n,
        input_access=input_access,
        vec1_access=_emit_strided_access(
            "vec1",
            ("i",),
            vec1_strides,
            vec1_is_contiguous,
            sizes=vec1_shape,
            c_type=dtype.c_type,
        ),
        vec2_access=_emit_strided_access(
            "vec2",
            ("j",),
            vec2_strides,
            vec2_is_contiguous,
            sizes=vec2_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i", "j"),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
        skip_input=skip_input,
    )
    return rendered.strip().splitlines()


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
    embedding_template = get_template_env().get_template(
        "embedding_kernel.c.j2"
    )
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
    loop_lines, indent = emit_loops(output_shape)
    indices_rank = len(indices_shape)
    output_rank = len(output_shape)
    output_indices = [f"i{dim}" for dim in range(output_rank)]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
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
    body_lines = [f"{indent}int64_t idx = (int64_t)({index_access});"]
    if padding_idx != -1:
        zero_literal = _format_scalar_literal(0.0, dtype)
        body_lines.extend(
            [
                f"{indent}if (idx == {padding_idx}) {{",
                f"{indent}    {output_access} = {zero_literal};",
                f"{indent}}} else {{",
                f"{indent}    {output_access} = {weight_access};",
                f"{indent}}}",
            ]
        )
    else:
        body_lines.append(f"{indent}{output_access} = {weight_access};")
    footer_lines = emit_footer(output_shape, indent)
    rendered = embedding_template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


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
    embedding_template = get_template_env().get_template(
        "embedding_kernel.c.j2"
    )
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
        f"{indent}int64_t start = (int64_t)({offsets_access});",
    ]
    if include_last_offset:
        body_lines.append(
            f"{indent}int64_t end = (int64_t)({offsets_next_access});"
        )
    else:
        body_lines.append(
            f"{indent}int64_t end = (i0 + 1 < {offsets_shape[0]}) "
            f"? (int64_t)({offsets_next_access}) "
            f": {indices_shape[0]};"
        )
    body_lines.extend(
        [
            f"{indent}{dtype.c_type} acc = {zero_literal};",
            f"{indent}int64_t count = 0;",
            f"{indent}for (int64_t j = start; j < end; ++j) {{",
        ]
    )
    inner_indent = f"{indent}    "
    body_lines.append(
        f"{inner_indent}int64_t idx = (int64_t)({indices_access});"
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


def _write_gather_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    index_shape: Sequence[int],
    input_strides: Sequence[int],
    index_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    index_dtype: torch.dtype,
    gather_dim: int,
    dtype: _CodegenDType,
) -> List[str]:
    gather_template = get_template_env().get_template("gather_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    index_suffix = _format_array_suffix(index_shape)
    out_suffix = _format_array_suffix(output_shape)
    index_c_type = _dtype_to_c_type(index_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {index_c_type} index{index_suffix}, "
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
    index_access = _emit_strided_access(
        "index",
        output_indices,
        index_strides,
        _is_contiguous(index_shape, index_strides),
        sizes=index_shape,
        c_type=index_c_type,
    )
    input_indices = [
        "idx" if dim == gather_dim else f"i{dim}"
        for dim in range(len(input_shape))
    ]
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    body_lines = [
        f"{indent}int64_t idx = (int64_t)({index_access});",
        f"{indent}{output_access} = {input_access};",
    ]
    footer_lines = emit_footer(output_shape, indent)
    rendered = gather_template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


def _write_conv2d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    output_shape: Sequence[int],
    transposed: bool,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    dtype: _CodegenDType,
    has_bias: bool,
) -> List[str]:
    template_name = (
        "conv2d_transpose_kernel.c.j2" if transposed else "conv2d_kernel.c.j2"
    )
    conv2d_template = get_template_env().get_template(template_name)
    if len(input_shape) == 4:
        has_batch = True
        batch, in_channels, in_h, in_w = input_shape
    elif len(input_shape) == 3:
        has_batch = False
        batch = 1
        in_channels, in_h, in_w = input_shape
    else:
        raise RefBackendError("codegen conv2d requires 3D or 4D input tensors")
    if transposed:
        weight_in_channels, weight_out_channels, k_h, k_w = weight_shape
        out_channels = weight_out_channels * groups
    else:
        out_channels, _, k_h, k_w = weight_shape
    if has_batch:
        out_h, out_w = output_shape[2], output_shape[3]
    else:
        out_h, out_w = output_shape[1], output_shape[2]
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    weight_suffix = _format_array_suffix(weight_shape)
    output_suffix = _format_array_suffix(output_shape)
    bias_arg = (
        f"const {dtype.c_type} bias[{out_channels}], " if has_bias else ""
    )
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} weight{weight_suffix}, "
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = conv2d_template.render(
        signature=signature,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        groups=groups,
        c_type=dtype.c_type,
        has_bias=has_bias,
        has_batch=has_batch,
    )
    return rendered.strip().splitlines()


def _write_pool2d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    dtype: _CodegenDType,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> List[str]:
    pool2d_template = get_template_env().get_template("pool2d_kernel.c.j2")
    batch, channels, in_h, in_w = input_shape
    out_h, out_w = output_shape[2], output_shape[3]
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pool2d_template.render(
        signature=signature,
        pool_kind=op_spec.name,
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        c_type=dtype.c_type,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        has_divisor_override=divisor_override is not None,
    )
    return rendered.strip().splitlines()


def _write_pool3d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    dtype: _CodegenDType,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> List[str]:
    pool3d_template = get_template_env().get_template("pool3d_kernel.c.j2")
    batch, channels, in_d, in_h, in_w = input_shape
    out_d, out_h, out_w = output_shape[2], output_shape[3], output_shape[4]
    k_d, k_h, k_w = kernel_size
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pool3d_template.render(
        signature=signature,
        pool_kind=op_spec.name,
        batch=batch,
        channels=channels,
        in_d=in_d,
        in_h=in_h,
        in_w=in_w,
        out_d=out_d,
        out_h=out_h,
        out_w=out_w,
        k_d=k_d,
        k_h=k_h,
        k_w=k_w,
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_d=pad_d,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_d=dil_d,
        dil_h=dil_h,
        dil_w=dil_w,
        c_type=dtype.c_type,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        has_divisor_override=divisor_override is not None,
    )
    return rendered.strip().splitlines()


def _write_adaptive_avg_pool2d_backward_kernel(
    node_index: int,
    op_spec: _OpSpec,
    grad_output_shape: Sequence[int],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dtype: _CodegenDType,
) -> List[str]:
    pool2d_template = get_template_env().get_template(
        "adaptive_avg_pool2d_backward_kernel.c.j2"
    )
    batch, channels, in_h, in_w = input_shape
    out_h, out_w = grad_output_shape[2], grad_output_shape[3]
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    grad_output_suffix = _format_array_suffix(grad_output_shape)
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} grad_output{grad_output_suffix}, "
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pool2d_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        c_type=dtype.c_type,
    )
    return rendered.strip().splitlines()


def _write_pool1d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: _CodegenDType,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> List[str]:
    pool1d_template = get_template_env().get_template("pool1d_kernel.c.j2")
    batch, channels, in_l = input_shape
    out_l = output_shape[2]
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pool1d_template.render(
        signature=signature,
        pool_kind=op_spec.name,
        batch=batch,
        channels=channels,
        in_l=in_l,
        out_l=out_l,
        k_l=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        c_type=dtype.c_type,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        has_divisor_override=divisor_override is not None,
    )
    return rendered.strip().splitlines()


def _write_col2im_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    output_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    padding: Tuple[int, int],
    stride: Tuple[int, int],
    dtype: _CodegenDType,
    out_blocks_h: int,
    out_blocks_w: int,
) -> List[str]:
    col2im_template = get_template_env().get_template("col2im_kernel.c.j2")
    if len(output_shape) == 4:
        batch, channels, out_h, out_w = output_shape
        has_batch = True
    else:
        channels, out_h, out_w = output_shape
        batch = 1
        has_batch = False
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = col2im_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        c_type=dtype.c_type,
        out_blocks_h=out_blocks_h,
        out_blocks_w=out_blocks_w,
        has_batch=has_batch,
    )
    return rendered.strip().splitlines()


def _write_batch_norm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
    eps: float,
    has_weight: bool,
    has_bias: bool,
) -> List[str]:
    batch_norm_template = get_template_env().get_template(
        "batch_norm_kernel.c.j2"
    )
    batch = input_shape[0]
    channels = input_shape[1]
    inner_size = 1
    for dim in input_shape[2:]:
        inner_size *= dim
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    weight_arg = (
        f"const {dtype.c_type} weight[{channels}], " if has_weight else ""
    )
    bias_arg = (
        f"const {dtype.c_type} bias[{channels}], " if has_bias else ""
    )
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} running_mean[{channels}], "
        f"const {dtype.c_type} running_var[{channels}], "
        f"{weight_arg}"
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = batch_norm_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        inner_size=inner_size,
        c_type=dtype.c_type,
        eps=_format_scalar_literal(eps, dtype),
        has_weight=has_weight,
        has_bias=has_bias,
        one_literal=_format_scalar_literal(1.0, dtype),
        zero_literal=_format_scalar_literal(0.0, dtype),
    )
    return rendered.strip().splitlines()


def _write_pdist_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    pdist_template = get_template_env().get_template("pdist_kernel.c.j2")
    n, m = input_shape
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pdist_template.render(
        signature=signature,
        n=n,
        m=m,
        c_type=dtype.c_type,
    )
    return rendered.strip().splitlines()


def _write_cdist_kernel(
    node_index: int,
    op_spec: _OpSpec,
    x1_shape: Sequence[int],
    x2_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    cdist_template = get_template_env().get_template("cdist_kernel.c.j2")
    n, m = x1_shape
    r, _ = x2_shape
    x1_suffix = _format_array_suffix(x1_shape)
    x2_suffix = _format_array_suffix(x2_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} x1{x1_suffix}, "
        f"const {dtype.c_type} x2{x2_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = cdist_template.render(
        signature=signature,
        n=n,
        r=r,
        m=m,
        c_type=dtype.c_type,
    )
    return rendered.strip().splitlines()


def _write_conv1d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    output_shape: Sequence[int],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    dtype: _CodegenDType,
    has_bias: bool,
) -> List[str]:
    conv1d_template = get_template_env().get_template("conv1d_kernel.c.j2")
    batch, in_channels, in_l = input_shape
    out_channels, _, k_l = weight_shape
    out_l = output_shape[2]
    input_suffix = _format_array_suffix(input_shape)
    weight_suffix = _format_array_suffix(weight_shape)
    output_suffix = _format_array_suffix(output_shape)
    bias_arg = (
        f"const {dtype.c_type} bias[{out_channels}], " if has_bias else ""
    )
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} weight{weight_suffix}, "
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = conv1d_template.render(
        signature=signature,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_l=in_l,
        out_l=out_l,
        k_l=k_l,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        c_type=dtype.c_type,
        has_bias=has_bias,
    )
    return rendered.strip().splitlines()


def _write_softmax_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_strides: Sequence[int],
    softmax_dim: int | None,
    dtype: _CodegenDType,
) -> List[str]:
    if softmax_dim is None:
        raise RefBackendError("codegen softmax expects a reduction dimension")
    softmax_template = get_template_env().get_template("softmax_kernel.c.j2")
    rank = len(input_shape)
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(input_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    output_dims = [{"dim": dim, "size": size} for dim, size in enumerate(input_shape)]
    input_contig = _is_contiguous(input_shape, input_strides)
    current_indices = [f"i{dim}" for dim in range(rank)]
    r_indices = current_indices.copy()
    r_indices[softmax_dim] = f"r{softmax_dim}"
    zero_indices = current_indices.copy()
    zero_indices[softmax_dim] = "0"
    input_access_r = _emit_strided_access(
        "input",
        r_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    input_access_zero = _emit_strided_access(
        "input",
        zero_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    input_access_current = _emit_strided_access(
        "input",
        current_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    output_access = _format_output_access(
        "out", input_shape, output_strides, c_type=dtype.c_type
    )
    rendered = softmax_template.render(
        signature=signature,
        output_dims=output_dims,
        softmax_dim=softmax_dim,
        softmax_size=input_shape[softmax_dim],
        c_type=dtype.c_type,
        input_access_zero=input_access_zero,
        input_access_r=input_access_r,
        input_access_current=input_access_current,
        output_access=output_access,
        is_log=op_spec.name in {"log_softmax", "_log_softmax"},
    )
    return rendered.strip().splitlines()


def _write_diagonal_kernel(
    node_index: int,
    op_node: _OpNode,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    input_dtype: torch.dtype,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_node.spec.name}_{dtype.suffix}("
        f"const {input_c_type} a{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    lines = [signature]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    input_access = _format_diagonal_access(
        "a",
        input_shape,
        input_strides,
        output_shape,
        dim1=int(op_node.p("dim1")),
        dim2=int(op_node.p("dim2")),
        offset=int(op_node.p("offset")),
        c_type=input_c_type,
    )
    lines.append(f"{indent}{output_access} = {input_access};")
    lines.extend(emit_footer(output_shape, indent))
    return lines


def _write_cumsum_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_strides: Sequence[int],
    cumsum_dim: int,
    graph_dtype: _CodegenDType,
    output_dtype: torch.dtype,
) -> List[str]:
    output_dtype_info = _CODEGEN_DTYPES.get(output_dtype)
    if output_dtype_info is None:
        raise RefBackendError(
            "codegen cumsum supports only torch.float32, torch.int8, or torch.int32"
        )
    output_c_type = output_dtype_info.c_type
    input_c_type = _input_c_type(graph_dtype.torch_dtype, graph_dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{graph_dtype.suffix}("
        f"const {input_c_type} input{_format_array_suffix(input_shape)}, "
        f"{output_c_type} out{_format_array_suffix(input_shape)}) {{"
    )
    lines = [signature]
    if not input_shape:
        lines.append(f"    out[0] = ({output_c_type})input[0];")
        lines.append("}")
        return lines
    loop_lines, indent = emit_loops(input_shape)
    lines.extend(loop_lines)
    output_access = _emit_strided_access(
        "out",
        [f"i{dim}" for dim in range(len(input_shape))],
        output_strides,
        _is_contiguous(input_shape, output_strides),
        sizes=input_shape,
        c_type=output_dtype_info.c_type,
    )
    acc_init = "0" if output_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    lines.append(f"{indent}{output_dtype_info.c_type} acc = {acc_init};")
    lines.append(
        f"{indent}for (int64_t r{cumsum_dim} = 0; r{cumsum_dim} <= i{cumsum_dim}; ++r{cumsum_dim}) {{"
    )
    inner_indent = f"{indent}    "
    input_indices = [
        f"r{cumsum_dim}" if dim == cumsum_dim else f"i{dim}"
        for dim in range(len(input_shape))
    ]
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=input_c_type,
    )
    lines.append(f"{inner_indent}acc += ({output_c_type}){input_access};")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}{output_access} = acc;")
    lines.extend(emit_footer(input_shape, indent))
    return lines


class EmptyStridedEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("empty_strided requires op spec and dtype")
        return _write_empty_strided_kernel(
            req.node_index, op_spec, req.output_shape, dtype
        )


class DiagonalEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_node = req.op_node
        dtype = req.dtype
        if op_node is None or dtype is None:
            raise RefBackendError("diagonal requires op node and dtype")
        return _write_diagonal_kernel(
            req.node_index,
            op_node,
            req.input_shapes[0],
            req.input_strides[0],
            req.input_dtypes[0],
            req.output_shape,
            req.output_strides or (),
            dtype,
        )


class SoftmaxEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("softmax requires op spec and dtype")
        return _write_softmax_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_strides or (),
            int(req.params["dim"]) if req.params.get("dim") is not None else None,
            dtype,
        )


class CumsumEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        output_dtype = req.params.get("output_dtype")
        if op_spec is None or dtype is None or output_dtype is None:
            raise RefBackendError("cumsum requires op spec, dtype, and output dtype")
        return _write_cumsum_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_strides or (),
            int(req.params["dim"]),
            dtype,
            output_dtype,
        )


class EmbeddingEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("embedding requires op spec and dtype")
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


class EmbeddingBagEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("embedding_bag requires op spec and dtype")
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


class GatherEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("gather requires op spec and dtype")
        return _write_gather_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.input_strides[0],
            req.input_strides[1],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[1],
            int(req.params["dim"]),
            dtype,
        )


class ConcatEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("concat requires op spec and dtype")
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


class Pool2dEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("pool2d requires op spec and dtype")
        return _write_pool2d_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.params["kernel_size"],
            req.params["stride"],
            req.params["padding"],
            req.params["dilation"],
            dtype,
            bool(req.params.get("ceil_mode", False)),
            bool(req.params.get("count_include_pad", False)),
            req.params.get("divisor_override"),
        )


class Pool3dEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("pool3d requires op spec and dtype")
        return _write_pool3d_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.params["kernel_size"],
            req.params["stride"],
            req.params["padding"],
            req.params["dilation"],
            dtype,
            bool(req.params.get("ceil_mode", False)),
            bool(req.params.get("count_include_pad", False)),
            req.params.get("divisor_override"),
        )


class Pool2dBackwardEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("pool2d_backward requires op spec and dtype")
        return _write_adaptive_avg_pool2d_backward_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.output_shape,
            req.params["kernel_size"],
            req.params["stride"],
            dtype,
        )


class Pool1dEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("pool1d requires op spec and dtype")
        return _write_pool1d_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.params["kernel_size"],
            req.params["stride"],
            req.params["padding"],
            req.params["dilation"],
            dtype,
            bool(req.params.get("ceil_mode", False)),
            bool(req.params.get("count_include_pad", False)),
            req.params.get("divisor_override"),
        )


class Col2imEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("col2im requires op spec and dtype")
        return _write_col2im_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.params["output_size"],
            req.params["kernel_size"],
            req.params["dilation"],
            req.params["padding"],
            req.params["stride"],
            dtype,
            int(req.params.get("out_blocks_h", 1)),
            int(req.params.get("out_blocks_w", 1)),
        )


class BatchNormEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("batch_norm requires op spec and dtype")
        return _write_batch_norm_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            dtype,
            float(req.params.get("eps", 1e-5)),
            bool(req.params.get("has_weight", False)),
            bool(req.params.get("has_bias", False)),
        )


class PdistEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("pdist requires op spec and dtype")
        return _write_pdist_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            dtype,
        )


class CdistEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("cdist requires op spec and dtype")
        return _write_cdist_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.output_shape,
            dtype,
        )


class Conv1dEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("conv1d requires op spec and dtype")
        return _write_conv1d_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.output_shape,
            int(req.params.get("stride", 1)),
            int(req.params.get("padding", 0)),
            int(req.params.get("dilation", 1)),
            int(req.params.get("groups", 1)),
            dtype,
            bool(req.params.get("has_bias", False)),
        )


class Conv2dEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("conv2d requires op spec and dtype")
        return _write_conv2d_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.output_shape,
            bool(req.params.get("transposed", False)),
            req.params.get("stride", (1, 1)),
            req.params.get("padding", (0, 0)),
            req.params.get("dilation", (1, 1)),
            int(req.params.get("groups", 1)),
            dtype,
            bool(req.params.get("has_bias", False)),
        )


class AddmmEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("addmm requires op spec and dtype")
        return _write_addmm_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.input_shapes[1],
            req.input_shapes[2],
            req.input_strides[0],
            req.input_strides[1],
            req.input_strides[2],
            req.output_strides or (),
            dtype,
            alpha=float(req.params.get("alpha", 1.0)),
            beta=float(req.params.get("beta", 1.0)),
        )


class AddbmmEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("addbmm requires op spec and dtype")
        return _write_addbmm_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.input_shapes[1],
            req.input_shapes[2],
            req.input_strides[0],
            req.input_strides[1],
            req.input_strides[2],
            req.output_strides or (),
            dtype,
            alpha=float(req.params.get("alpha", 1.0)),
            beta=float(req.params.get("beta", 1.0)),
        )


class AddmvEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("addmv requires op spec and dtype")
        return _write_addmv_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.input_shapes[1],
            req.input_shapes[2],
            req.input_strides[0],
            req.input_strides[1],
            req.input_strides[2],
            req.output_strides or (),
            dtype,
            alpha=float(req.params.get("alpha", 1.0)),
            beta=float(req.params.get("beta", 1.0)),
        )


class AddrEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("addr requires op spec and dtype")
        return _write_addr_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.input_shapes[1],
            req.input_shapes[2],
            req.input_strides[0],
            req.input_strides[1],
            req.input_strides[2],
            req.output_strides or (),
            dtype,
            alpha=float(req.params.get("alpha", 1.0)),
            beta=float(req.params.get("beta", 1.0)),
        )
