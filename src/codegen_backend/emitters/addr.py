from __future__ import annotations

import math
from typing import List, Sequence

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.c_types import _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


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
