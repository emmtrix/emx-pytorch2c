from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.c_types import _format_scalar_literal, _input_c_type
from codegen_backend.emitters.base import (
    KindEmitterBase,
    emit_input_access,
    emit_output_access,
    emit_signature,
)
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import OpKind
from codegen_backend.templates import get_template_env

_PARAMETRIC_UNARY_OPS = {
    "gelu",
    "elu",
    "leaky_relu",
    "softplus",
    "hardtanh",
    "clamp",
}
_FLOAT_ONLY_UNARY_OPS = {
    "gelu",
    "elu",
    "leaky_relu",
    "softplus",
    "hardtanh",
}


class ElementwiseEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_node = req.op_node
        op_spec = op_node.spec
        elementwise_template = get_template_env().get_template(
            "elementwise_kernel.c.j2"
        )
        params = op_node.params
        signature = emit_signature(
            req.node_index,
            op_spec,
            req.output_shape,
            req.input_shapes,
            req.input_dtypes,
            req.dtype,
            params,
        )
        output_dims = [
            {"dim": dim, "size": size}
            for dim, size in enumerate(req.output_shape)
        ]
        output_access = emit_output_access(
            req.output_shape, req.output_strides, c_type=req.dtype.c_type
        )
        scalar_fn = f"{req.dtype.scalar_prefix}{op_spec.name}"
        context: Dict[str, object] = {
            "signature": signature,
            "output_dims": output_dims,
            "op_kind": op_spec.kind.value,
            "op_name": op_spec.name,
            "scalar_fn": scalar_fn,
            "output_access": output_access,
            "is_copy": op_spec.name == "copy",
            "is_alias": op_spec.name in {"alias", "clone", "_to_copy", "resize_"},
            "is_parametric": op_spec.name in _PARAMETRIC_UNARY_OPS,
            "has_scalar": "scalar" in params,
            "scalar_literal": None,
            "fill_value": None,
            "a_access": None,
            "b_access": None,
            "cond_access": None,
            "input_access": None,
        }
        if op_spec.kind == OpKind.BINARY:
            if "scalar" in params:
                a_shape = req.input_shapes[0]
                a_strides = req.input_strides[0]
                context["a_access"] = emit_input_access(
                    "a",
                    a_shape,
                    a_strides,
                    req.output_shape,
                    broadcast_contiguous=False,
                    c_type=_input_c_type(req.input_dtypes[0], req.dtype),
                )
                context["scalar_literal"] = _format_scalar_literal(
                    params["scalar"], req.dtype
                )
            else:
                a_shape, b_shape = req.input_shapes
                a_strides, b_strides = req.input_strides
                context["a_access"] = emit_input_access(
                    "a",
                    a_shape,
                    a_strides,
                    req.output_shape,
                    broadcast_contiguous=True,
                    c_type=_input_c_type(req.input_dtypes[0], req.dtype),
                )
                context["b_access"] = emit_input_access(
                    "b",
                    b_shape,
                    b_strides,
                    req.output_shape,
                    broadcast_contiguous=True,
                    c_type=_input_c_type(req.input_dtypes[1], req.dtype),
                )
        elif op_spec.kind == OpKind.WHERE:
            input_index = 0
            cond_shape = req.input_shapes[input_index]
            cond_strides = req.input_strides[input_index]
            context["cond_access"] = emit_input_access(
                "cond",
                cond_shape,
                cond_strides,
                req.output_shape,
                broadcast_contiguous=True,
                c_type=_input_c_type(req.input_dtypes[input_index], req.dtype),
            )
            input_index += 1
            if "a_scalar" in params:
                context["a_access"] = _format_scalar_literal(
                    params["a_scalar"], req.dtype
                )
            else:
                a_shape = req.input_shapes[input_index]
                a_strides = req.input_strides[input_index]
                context["a_access"] = emit_input_access(
                    "a",
                    a_shape,
                    a_strides,
                    req.output_shape,
                    broadcast_contiguous=True,
                    c_type=_input_c_type(req.input_dtypes[input_index], req.dtype),
                )
                input_index += 1
            if "b_scalar" in params:
                context["b_access"] = _format_scalar_literal(
                    params["b_scalar"], req.dtype
                )
            else:
                b_shape = req.input_shapes[input_index]
                b_strides = req.input_strides[input_index]
                context["b_access"] = emit_input_access(
                    "b",
                    b_shape,
                    b_strides,
                    req.output_shape,
                    broadcast_contiguous=True,
                    c_type=_input_c_type(req.input_dtypes[input_index], req.dtype),
                )
        elif op_spec.kind == OpKind.FILL:
            context["fill_value"] = _format_scalar_literal(
                op_node.p("value"), req.dtype
            )
        else:
            a_shape = req.input_shapes[0]
            a_strides = req.input_strides[0]
            context["input_access"] = emit_input_access(
                "a",
                a_shape,
                a_strides,
                req.output_shape,
                broadcast_contiguous=False,
                c_type=_input_c_type(req.input_dtypes[0], req.dtype),
            )
            if op_spec.name in _PARAMETRIC_UNARY_OPS:
                if req.dtype.torch_dtype is not torch.float32:
                    raise RefBackendError(
                        f"codegen {op_spec.name} supports only torch.float32 tensors"
                    )
                context.update(
                    {
                        "one": _format_scalar_literal(1.0, req.dtype),
                        "half": _format_scalar_literal(0.5, req.dtype),
                        "gelu_approximate": params.get("approximate", "none"),
                        "sqrt_2_over_pi": _format_scalar_literal(
                            0.7978845608028654, req.dtype
                        ),
                        "coeff": _format_scalar_literal(0.044715, req.dtype),
                        "inv_sqrt2": _format_scalar_literal(
                            0.7071067811865475, req.dtype
                        ),
                        "alpha": _format_scalar_literal(
                            params.get("alpha", 1.0), req.dtype
                        ),
                        "scale": _format_scalar_literal(
                            params.get("scale", 1.0), req.dtype
                        ),
                        "input_scale": _format_scalar_literal(
                            params.get("input_scale", 1.0), req.dtype
                        ),
                        "negative_slope": _format_scalar_literal(
                            params.get("negative_slope", 0.01), req.dtype
                        ),
                        "beta": _format_scalar_literal(
                            params.get("beta", 1.0), req.dtype
                        ),
                        "threshold": _format_scalar_literal(
                            params.get("threshold", 20.0), req.dtype
                        ),
                        "clamp_min": (
                            _format_scalar_literal(
                                params.get("min_val"), req.dtype
                            )
                            if params.get("min_val") is not None
                            else None
                        ),
                        "clamp_max": (
                            _format_scalar_literal(
                                params.get("max_val"), req.dtype
                            )
                            if params.get("max_val") is not None
                            else None
                        ),
                        "clamp_has_min": params.get("min_val") is not None,
                        "clamp_has_max": params.get("max_val") is not None,
                        "min_val": _format_scalar_literal(
                            params.get("min_val", -1.0), req.dtype
                        ),
                        "max_val": _format_scalar_literal(
                            params.get("max_val", 1.0), req.dtype
                        ),
                    }
                )
        rendered = elementwise_template.render(**context)
        return rendered.strip().splitlines()


__all__ = [
    "ElementwiseEmitter",
    "_FLOAT_ONLY_UNARY_OPS",
    "_PARAMETRIC_UNARY_OPS",
]
