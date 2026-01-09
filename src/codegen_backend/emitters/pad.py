from __future__ import annotations

from typing import List

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _format_scalar_literal, _input_c_type
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _is_contiguous,
    emit_output_access,
    emit_signature,
)
from codegen_backend.indexing import _emit_strided_access, _format_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.templates import get_template_env


class PadEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        if op_spec is None:
            raise CodegenBackendError("pad requires op spec")
        input_shape = req.input_shapes[0]
        input_strides = req.input_strides[0]
        input_dtype = req.input_dtypes[0]
        output_shape = req.output_shape
        output_strides = req.output_strides
        mode = req.params.get("mode", "constant")
        if mode == "constant":
            pad_template = get_template_env().get_template(
                "constant_pad_nd_kernel.c.j2"
            )
        else:
            pad_template = get_template_env().get_template(
                "mirror_pad_nd_kernel.c.j2"
            )
        signature = emit_signature(
            req.node_index,
            op_spec,
            output_shape,
            [input_shape],
            [input_dtype],
            req.dtype,
            signature_kind="unary",
            input_dim_names=req.input_dim_names,
            output_dim_names=req.output_dim_names,
            dim_order=req.dim_order,
        )
        output_access = emit_output_access(
            output_shape, output_strides, c_type=req.dtype.c_type
        )
        if not input_shape:
            input_access = _format_strided_access(
                "a",
                input_shape,
                input_strides,
                output_shape,
                c_type=_input_c_type(input_dtype, req.dtype),
            )
            rendered = pad_template.render(
                signature=signature,
                output_access=output_access,
                input_access=input_access,
                has_input_shape=False,
            )
            return rendered.strip().splitlines()
        pad_before = req.params.get("pad_before", ())
        pad_after = req.params.get("pad_after", ())
        input_access = _emit_strided_access(
            "a",
            [f"in_{dim}" for dim in range(len(input_shape))],
            input_strides,
            contig=_is_contiguous(input_shape, input_strides),
            sizes=input_shape,
            c_type=_input_c_type(input_dtype, req.dtype),
        )
        rendered = pad_template.render(
            signature=signature,
            output_access=output_access,
            input_access=input_access,
            has_input_shape=True,
            output_shape=output_shape,
            input_shape=input_shape,
            pad_before=pad_before,
            pad_after=pad_after,
            mode=mode,
            value=_format_scalar_literal(req.params["value"], req.dtype)
            if mode == "constant"
            else None,
        )
        return rendered.strip().splitlines()
