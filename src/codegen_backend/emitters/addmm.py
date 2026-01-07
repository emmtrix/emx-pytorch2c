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
