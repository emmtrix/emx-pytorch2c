from __future__ import annotations

from typing import List, Sequence, Tuple

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


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
    bias_arg = f"const {dtype.c_type} bias[{out_channels}], " if has_bias else ""
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
