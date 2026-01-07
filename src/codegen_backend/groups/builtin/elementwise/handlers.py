from __future__ import annotations

from typing import Dict, List

import torch
import torch.fx

from codegen_backend.c_types import _normalize_scalar_value
from codegen_backend.dtypes import _CodegenDType, _INTEGER_CODEGEN_DTYPES
from codegen_backend.emitters.elementwise import (
    _FLOAT_ONLY_UNARY_OPS,
    _PARAMETRIC_UNARY_OPS,
    ElementwiseEmitter,
)
from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    ElementwiseHandler,
    HandlerContext,
    OpKindHandler,
    OpNodeBuildResult,
)
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.groups.builtin.elementwise.analysis import (
    handle_fill_node,
    handle_to_copy_node,
    parse_bitwise_scalar,
    parse_parametric_unary_args,
    parse_where_inputs,
)


_BITWISE_OPS = {
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "bitwise_not",
}
_BITWISE_BOOL_OPS = {
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
}


class _BackendElementwiseHandler(ElementwiseHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if op_spec.name == "_to_copy":
            if dtype_info is None:
                return None
            op_node = handle_to_copy_node(
                node, op_spec, dtype_info, shapes, strides, dtypes
            )
            return OpNodeBuildResult(op_node)
        if op_spec.kind == OpKind.FILL:
            if dtype_info is None:
                return None
            op_node = handle_fill_node(
                node,
                op_spec,
                dtype_info,
                shapes,
                strides,
                dtypes,
                inplace_input,
                infer_output_shape=self._ctx.analysis_service.infer_output_shape,
            )
            return OpNodeBuildResult(op_node)
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        param_values: Dict[str, object] = {}
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[tuple[int, ...]] = []
        out_arg: torch.fx.Node | None = None

        if op_spec.kind == OpKind.BINARY and len(node.args) == 2:
            lhs, rhs = node.args
            if isinstance(lhs, torch.fx.Node) ^ isinstance(rhs, torch.fx.Node):
                if node.kwargs:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects positional args only"
                    )
                input_arg = lhs if isinstance(lhs, torch.fx.Node) else rhs
                scalar_arg = rhs if isinstance(lhs, torch.fx.Node) else lhs
                if input_arg not in shapes:
                    raise self._ctx.analysis_service.error_expected_tensor(
                        op_spec.name
                    )
                input_nodes = [input_arg]
                input_shapes = [shapes[input_arg]]
                if op_spec.name in _BITWISE_OPS:
                    param_values["scalar"] = parse_bitwise_scalar(
                        op_spec.name, scalar_arg, dtype_info.torch_dtype
                    )
                else:
                    param_values["scalar"] = _normalize_scalar_value(
                        op_spec.name, scalar_arg
                    )
            elif isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node):
                lhs_in_shapes = lhs in shapes
                rhs_in_shapes = rhs in shapes
                if lhs_in_shapes ^ rhs_in_shapes:
                    if node.kwargs:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects positional args only"
                        )
                    input_arg = lhs if lhs_in_shapes else rhs
                    scalar_arg = rhs if lhs_in_shapes else lhs
                    input_nodes = [input_arg]
                    input_shapes = [shapes[input_arg]]
                    if op_spec.name in _BITWISE_OPS:
                        param_values["scalar"] = parse_bitwise_scalar(
                            op_spec.name, scalar_arg, dtype_info.torch_dtype
                        )
                    else:
                        param_values["scalar"] = self._ctx.analysis_service.resolve_scalar_arg(
                            op_spec.name, scalar_arg, scalar_values
                        )

        if not input_nodes:
            if (
                op_spec.kind == OpKind.UNARY
                and op_spec.name in _PARAMETRIC_UNARY_OPS
            ):
                input_node, param_values = parse_parametric_unary_args(
                    op_spec.name, node
                )
                args_to_check = (input_node,)
            else:
                allowed_kwargs = set()
                is_out_overload = self._ctx.analysis_service.is_out_overload(
                    node.target
                )
                if op_spec.name == "div":
                    allowed_kwargs = {"rounding_mode"}
                elif op_spec.name == "copy":
                    allowed_kwargs = {"non_blocking"}
                elif op_spec.name == "relu":
                    allowed_kwargs = {"inplace"}
                    if node.kwargs.get("inplace"):
                        raise CodegenBackendError(
                            "codegen relu expects inplace to be False"
                        )
                if is_out_overload:
                    allowed_kwargs.add("out")
                if node.kwargs and set(node.kwargs) - allowed_kwargs:
                    raise CodegenBackendError(
                        "codegen backend expects positional args only"
                    )
                if op_spec.kind == OpKind.UNARY:
                    expected_arity = 1
                elif op_spec.kind == OpKind.BINARY:
                    expected_arity = 2
                elif op_spec.kind == OpKind.WHERE:
                    expected_arity = 3
                else:
                    expected_arity = 2
                if is_out_overload:
                    if inplace_input is None:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects out to be provided"
                        )
                    if "out" in node.kwargs:
                        if len(node.args) > expected_arity:
                            raise self._ctx.analysis_service.error_kwarg_specified_once(
                                op_spec.name, "out"
                            )
                        out_arg = node.kwargs["out"]
                    elif len(node.args) == expected_arity + 1:
                        out_arg = node.args[inplace_input]
                    elif len(node.args) != expected_arity:
                        if expected_arity == 1:
                            raise CodegenBackendError(
                                f"codegen {op_spec.name} expects one input"
                            )
                        if expected_arity == 2:
                            raise CodegenBackendError(
                                f"codegen {op_spec.name} expects exactly two inputs"
                            )
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects exactly three inputs"
                        )
                    if out_arg is None:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects out to be provided"
                        )
                elif op_spec.name == "copy":
                    if len(node.args) not in {2, 3}:
                        raise CodegenBackendError(
                            "codegen copy expects two inputs and optional non_blocking"
                        )
                elif len(node.args) != expected_arity:
                    if expected_arity == 1:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects one input"
                        )
                    if expected_arity == 2:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects exactly two inputs"
                        )
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects exactly three inputs"
                    )
                if op_spec.name == "div":
                    rounding_mode = node.kwargs.get("rounding_mode")
                    if rounding_mode is not None:
                        raise CodegenBackendError(
                            "codegen div expects rounding_mode to be None"
                        )
                if op_spec.name == "copy":
                    non_blocking = None
                    if len(node.args) > 2:
                        non_blocking = node.args[2]
                    if "non_blocking" in node.kwargs:
                        if len(node.args) > 2:
                            raise self._ctx.analysis_service.error_kwarg_specified_once(
                                op_spec.name, "non_blocking"
                            )
                        non_blocking = node.kwargs["non_blocking"]
                    if non_blocking not in (None, False, 0):
                        raise CodegenBackendError(
                            "codegen copy expects non_blocking to be False"
                        )
                if op_spec.name == "copy":
                    args_to_check = node.args[:2]
                else:
                    args_to_check = node.args
            if op_spec.kind == OpKind.WHERE:
                (
                    input_nodes,
                    input_shapes,
                    where_params,
                ) = parse_where_inputs(op_spec, node, shapes, scalar_values)
                param_values.update(where_params)
            else:
                for arg in args_to_check:
                    if not isinstance(arg, torch.fx.Node):
                        raise self._ctx.analysis_service.error_expected_tensor(
                            op_spec.name
                        )
                    if arg not in shapes:
                        raise self._ctx.analysis_service.error_expected_tensor(
                            op_spec.name
                        )
                    input_nodes.append(arg)
                    input_shapes.append(shapes[arg])
            if out_arg is not None and out_arg not in input_nodes:
                if not isinstance(out_arg, torch.fx.Node):
                    raise self._ctx.analysis_service.error_expected_tensor(
                        op_spec.name
                    )
                if out_arg not in shapes:
                    raise self._ctx.analysis_service.error_expected_tensor(
                        op_spec.name
                    )
                input_nodes.append(out_arg)
                input_shapes.append(shapes[out_arg])

        shape_input_shapes = [
            shape
            for arg, shape in zip(input_nodes, input_shapes)
            if out_arg is None or arg is not out_arg
        ]
        if op_spec.kind == OpKind.WHERE:
            if "a_scalar" in param_values:
                shape_input_shapes.append(())
            if "b_scalar" in param_values:
                shape_input_shapes.append(())
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if op_spec.name in _BITWISE_OPS:
            if dtype_info.torch_dtype in _INTEGER_CODEGEN_DTYPES:
                pass
            elif (
                dtype_info.torch_dtype is torch.bool
                and op_spec.name in _BITWISE_BOOL_OPS
            ):
                pass
            else:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects integer tensors"
                )
        if op_spec.name in _FLOAT_ONLY_UNARY_OPS:
            if dtype_info.torch_dtype is not torch.float32:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32 tensors"
                )
            if any(dtype is not torch.float32 for dtype in input_dtypes):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32 tensors"
                )
        if op_spec.name == "clamp" and dtype_info.torch_dtype is torch.bool:
            raise CodegenBackendError("codegen clamp supports only numeric tensors")
        if (
            op_spec.name == "clamp"
            and dtype_info.torch_dtype in _INTEGER_CODEGEN_DTYPES
        ):
            for name in ("min_val", "max_val"):
                value = param_values.get(name)
                if value is None:
                    continue
                if not float(value).is_integer():
                    raise CodegenBackendError(
                        "codegen clamp expects integer min/max for integer tensors"
                    )
        if op_spec.kind == OpKind.WHERE:
            if input_dtypes[0] is not torch.bool:
                raise CodegenBackendError(
                    "codegen where expects condition to be a boolean tensor"
                )
            if any(
                dtype is not dtype_info.torch_dtype for dtype in input_dtypes[1:]
            ):
                raise CodegenBackendError(
                    "codegen where expects self and other to match the graph dtype"
                )
        elif any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
            params=param_values,
        )
        self.validate(op_node, shape_input_shapes, input_dtypes, dtype_info)
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, shape_input_shapes
        )
        op_node.output_shape = output_shape
        if out_arg is not None and shapes[out_arg] != output_shape:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects out to match output shape"
            )
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        if inplace_input is not None:
            strides[node] = strides[input_nodes[inplace_input]]
        else:
            strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


def build_handlers(context: HandlerContext) -> Dict[OpKind, OpKindHandler]:
    elementwise_emitter = ElementwiseEmitter()
    return {
        OpKind.BINARY: _BackendElementwiseHandler(
            context, elementwise_emitter, "binary"
        ),
        OpKind.UNARY: _BackendElementwiseHandler(
            context, elementwise_emitter, "unary"
        ),
        OpKind.WHERE: _BackendElementwiseHandler(
            context, elementwise_emitter, "where"
        ),
        OpKind.FILL: _BackendElementwiseHandler(
            context, elementwise_emitter, "fill"
        ),
    }


def build_kind_handler_registrations() -> Dict[OpKind, KindHandlerRegistration]:
    return {
        OpKind.BINARY: KindHandlerRegistration(
            _BackendElementwiseHandler, ElementwiseEmitter
        ),
        OpKind.UNARY: KindHandlerRegistration(
            _BackendElementwiseHandler, ElementwiseEmitter
        ),
        OpKind.WHERE: KindHandlerRegistration(
            _BackendElementwiseHandler, ElementwiseEmitter
        ),
        OpKind.FILL: KindHandlerRegistration(
            _BackendElementwiseHandler, ElementwiseEmitter
        ),
    }


__all__ = ["build_handlers", "build_kind_handler_registrations"]
