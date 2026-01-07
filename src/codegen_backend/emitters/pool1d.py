from __future__ import annotations

from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


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


class Pool1dEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("pool1d requires op spec and dtype")
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
