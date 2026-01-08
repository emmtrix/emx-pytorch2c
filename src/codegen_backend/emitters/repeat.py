from __future__ import annotations

from typing import List, Sequence

from codegen_backend.c_types import _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
    emit_output_access,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_repeat_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    input_dtype: object,
    dtype: _CodegenDType,
) -> List[str]:
    repeat_template = get_template_env().get_template("repeat_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if not input_shape:
        input_access = f"(({input_c_type}*)input)[0]"
    else:
        pad = len(output_shape) - len(input_shape)
        input_indices = []
        for dim, size in enumerate(input_shape):
            output_dim = dim + pad
            if size <= 1:
                input_indices.append("0")
            else:
                input_indices.append(f"(i{output_dim} % {size})")
        input_access = _emit_strided_access(
            "input",
            input_indices,
            input_strides,
            _is_contiguous(input_shape, input_strides),
            sizes=input_shape,
            c_type=input_c_type,
        )
    rendered = repeat_template.render(
        signature=signature,
        output_shape=output_shape,
        output_access=output_access,
        input_access=input_access,
    )
    return rendered.splitlines()


class RepeatEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("repeat requires op spec and dtype")
        return _write_repeat_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[0],
            dtype,
        )
