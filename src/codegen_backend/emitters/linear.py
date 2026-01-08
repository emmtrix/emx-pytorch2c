from __future__ import annotations

from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.dtypes import _CodegenDType, _INTEGER_CODEGEN_DTYPES
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.templates import get_template_env


def _write_linear_kernel(
    node_index: int,
    op_name: str,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    weight_shape: Sequence[int],
    input_strides: Sequence[int],
    weight_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    bias_shape: Sequence[int] | None = None,
    bias_strides: Sequence[int] | None = None,
) -> List[str]:
    linear_template = get_template_env().get_template("linear_kernel.c.j2")
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    weight_is_contiguous = _is_contiguous(weight_shape, weight_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    has_bias = bias_shape is not None and bias_strides is not None
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    *batch_shape, k = input_shape
    n, weight_k = weight_shape
    if weight_k != k:
        raise CodegenBackendError(
            "Linear kernel expects matching input/weight inner dims."
        )
    input_suffix = _format_array_suffix(input_shape)
    weight_suffix = _format_array_suffix(weight_shape)
    out_suffix = _format_array_suffix(output_shape)
    signature_parts = [
        f"const {dtype.c_type} input{input_suffix}",
        f"const {dtype.c_type} weight{weight_suffix}",
    ]
    if has_bias:
        bias_suffix = _format_array_suffix(bias_shape)
        signature_parts.append(f"const {dtype.c_type} bias{bias_suffix}")
    signature_parts.append(f"{dtype.c_type} out{out_suffix}")
    batch_indices = tuple(f"i{dim}" for dim in range(len(batch_shape)))
    output_indices = (*batch_indices, "j")
    bias_access = "0"
    if has_bias:
        bias_is_contiguous = _is_contiguous(bias_shape, bias_strides)
        bias_offset = len(output_indices) - len(bias_shape)
        bias_indices = output_indices[bias_offset:]
        bias_access = _emit_strided_access(
            "bias",
            bias_indices,
            bias_strides,
            bias_is_contiguous,
            sizes=bias_shape,
            c_type=dtype.c_type,
        )
    rendered = linear_template.render(
        signature=(
            f"void node{node_index}_{op_name}_{dtype.suffix}("
            f"{', '.join(signature_parts)}) {{"
        ),
        batch_shape=batch_shape,
        n=n,
        k=k,
        acc_type=acc_type,
        acc_init=acc_init,
        input_access=_emit_strided_access(
            "input",
            (*batch_indices, "t"),
            input_strides,
            input_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
        weight_access=_emit_strided_access(
            "weight",
            ("j", "t"),
            weight_strides,
            weight_is_contiguous,
            sizes=weight_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            output_indices,
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        bias_access=bias_access,
    )
    return rendered.strip().splitlines()


class LinearEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        input_shape, weight_shape = req.input_shapes[:2]
        input_strides, weight_strides = req.input_strides[:2]
        bias_shape = None
        bias_strides = None
        if req.params.get("has_bias"):
            bias_shape = req.input_shapes[2]
            bias_strides = req.input_strides[2]
        return _write_linear_kernel(
            req.node_index,
            req.op_spec.name,
            input_shape,
            req.output_shape,
            weight_shape,
            input_strides,
            weight_strides,
            req.output_strides or (),
            req.dtype,
            bias_shape=bias_shape,
            bias_strides=bias_strides,
        )
