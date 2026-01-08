from __future__ import annotations

from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_batch_norm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
    eps: float,
    momentum: float,
    training: bool,
    has_weight: bool,
    has_bias: bool,
) -> List[str]:
    batch_norm_template = get_template_env().get_template("batch_norm_kernel.c.j2")
    batch = input_shape[0]
    channels = input_shape[1]
    inner_size = 1
    for dim in input_shape[2:]:
        inner_size *= dim
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    weight_arg = f"const {dtype.c_type} weight[{channels}], " if has_weight else ""
    bias_arg = f"const {dtype.c_type} bias[{channels}], " if has_bias else ""
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} running_mean[{channels}], "
        f"{dtype.c_type} running_var[{channels}], "
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
        momentum=_format_scalar_literal(momentum, dtype),
        training=training,
        has_weight=has_weight,
        has_bias=has_bias,
        one_literal=_format_scalar_literal(1.0, dtype),
        zero_literal=_format_scalar_literal(0.0, dtype),
    )
    return rendered.strip().splitlines()


class BatchNormEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("batch_norm requires op spec and dtype")
        return _write_batch_norm_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            dtype,
            float(req.params.get("eps", 1e-5)),
            float(req.params.get("momentum", 0.1)),
            bool(req.params.get("training", False)),
            bool(req.params.get("has_weight", False)),
            bool(req.params.get("has_bias", False)),
        )
