from __future__ import annotations

from typing import List

from codegen_backend.errors import CodegenBackendError
from codegen_backend.emitters.base import (
    KindEmitterBase,
    emit_input_access,
    emit_output_access,
    emit_signature,
)
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.templates import get_template_env


class DropoutEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("dropout requires op spec and dtype")
        dropout_template = get_template_env().get_template("dropout_kernel.c.j2")
        signature = emit_signature(
            req.node_index,
            op_spec,
            req.output_shape,
            req.input_shapes,
            req.input_dtypes,
            dtype,
        )
        output_dims = [
            {"dim": dim, "size": size}
            for dim, size in enumerate(req.output_shape)
        ]
        input_access = emit_input_access(
            "a",
            req.input_shapes[0],
            req.input_strides[0],
            req.output_shape,
            broadcast_contiguous=False,
            c_type=dtype.c_type,
        )
        output_access = emit_output_access(
            req.output_shape, req.output_strides, c_type=dtype.c_type
        )
        rendered = dropout_template.render(
            signature=signature,
            output_dims=output_dims,
            input_access=input_access,
            output_access=output_access,
        )
        return rendered.strip().splitlines()
