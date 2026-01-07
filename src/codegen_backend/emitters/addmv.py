from __future__ import annotations

from typing import List, Sequence

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.c_types import _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType, _INTEGER_CODEGEN_DTYPES
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


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
