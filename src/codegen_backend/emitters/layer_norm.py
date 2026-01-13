from __future__ import annotations

from typing import List, Sequence

from codegen_backend.c_types import _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.errors import CodegenBackendError
from codegen_backend.kinds import KernelEmitRequest
from shared.scalar_functions import ScalarFunction, ScalarFunctionKey
from shared.scalar_types import ScalarType
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_layer_norm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    normalized_shape: Sequence[int],
    weight_shape: Sequence[int] | None,
    bias_shape: Sequence[int] | None,
    dtype: _CodegenDType,
    sqrt_fn: str,
    eps: float,
    has_weight: bool,
    has_bias: bool,
) -> List[str]:
    layer_norm_template = get_template_env().get_template("layer_norm_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    weight_suffix = _format_array_suffix(weight_shape or ())
    bias_suffix = _format_array_suffix(bias_shape or ())
    weight_arg = f"const {dtype.c_type} weight{weight_suffix}, " if has_weight else ""
    bias_arg = f"const {dtype.c_type} bias{bias_suffix}, " if has_bias else ""
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{weight_arg}"
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    input_zero_indices = "".join("[0]" for _ in input_shape) or "[0]"
    output_zero_indices = "".join("[0]" for _ in output_shape) or "[0]"
    weight_zero_indices = "".join("[0]" for _ in (weight_shape or ())) or "[0]"
    bias_zero_indices = "".join("[0]" for _ in (bias_shape or ())) or "[0]"
    outer_size = 1
    inner_size = 1
    normalized_rank = len(normalized_shape)
    for dim in input_shape[: len(input_shape) - normalized_rank]:
        outer_size *= dim
    for dim in input_shape[len(input_shape) - normalized_rank :]:
        inner_size *= dim
    rendered = layer_norm_template.render(
        signature=signature,
        outer_size=outer_size,
        inner_size=inner_size,
        c_type=dtype.c_type,
        sqrt_fn=sqrt_fn,
        eps=_format_scalar_literal(eps, dtype),
        has_weight=has_weight,
        has_bias=has_bias,
        input_zero_indices=input_zero_indices,
        output_zero_indices=output_zero_indices,
        weight_zero_indices=weight_zero_indices,
        bias_zero_indices=bias_zero_indices,
    )
    return rendered.strip().splitlines()


class LayerNormEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("layer_norm requires op spec and dtype")
        sqrt_fn = f"{dtype.scalar_prefix}sqrt"
        if req.scalar_registry is not None:
            sqrt_fn = req.scalar_registry.request(
                ScalarFunctionKey(
                    ScalarFunction.SQRT,
                    ScalarType.from_torch_dtype(dtype.torch_dtype),
                )
            )
        has_weight = bool(req.params.get("has_weight", False))
        has_bias = bool(req.params.get("has_bias", False))
        weight_shape = req.input_shapes[1] if has_weight else None
        bias_shape = req.input_shapes[2 if has_weight else 1] if has_bias else None
        return _write_layer_norm_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            tuple(req.params.get("normalized_shape", ())),
            weight_shape,
            bias_shape,
            dtype,
            sqrt_fn,
            float(req.params.get("eps", 1e-5)),
            has_weight,
            has_bias,
        )
