from __future__ import annotations

from typing import List, Sequence

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


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
    bias_arg = f"const {dtype.c_type} bias[{out_channels}], " if has_bias else ""
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
