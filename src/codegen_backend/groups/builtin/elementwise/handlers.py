from __future__ import annotations

from typing import Callable, Dict, List, Sequence, TYPE_CHECKING, Tuple

import torch
import torch.fx

from codegen_backend import shape_utils
from codegen_backend.c_types import _normalize_scalar_value
from codegen_backend.dtypes import _CodegenDType, _INTEGER_CODEGEN_DTYPES
from codegen_backend.emitters.elementwise import (
    _FLOAT_ONLY_UNARY_OPS,
    _PARAMETRIC_UNARY_OPS,
    ElementwiseEmitter,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _GenericGraph, _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    ElementwiseContext,
    HandlerContext,
    HandlerContextProvider,
    OpKindHandler,
    OpKindHandlerFactory,
    OpNodeBuildResult,
)
from codegen_backend.specs import OpKind, _OpSpec

if TYPE_CHECKING:
    from codegen_backend.emitters.base import KindEmitter
    from codegen_backend.emitters.registry import KindHandlerRegistration


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


class ElementwiseHandler(OpKindHandler):
    def __init__(
        self,
        context: HandlerContext,
        emitter: "KindEmitter | None",
        elementwise_kind: str,
        builder: Callable[
            [
                "torch.fx.Node",
                "_OpSpec",
                "_CodegenDType | None",
                Dict["torch.fx.Node", Tuple[int, ...]],
                Dict["torch.fx.Node", Tuple[int, ...]],
                Dict["torch.fx.Node", "torch.dtype"],
                Dict["torch.fx.Node", object],
                int | None,
            ],
            OpNodeBuildResult | None,
        ]
        | None = None,
    ) -> None:
        super().__init__(context, emitter, builder)
        self._elementwise_kind = elementwise_kind

    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        signature_kind = "unary"
        elementwise_kind = self._elementwise_kind
        if op_node.spec.name == "clamp_tensor":
            elementwise_kind = "clamp_tensor"
            signature_kind = "clamp_tensor"
        elif self._elementwise_kind == "binary":
            signature_kind = (
                "binary_scalar"
                if "scalar" in op_node.params
                else "binary"
            )
        elif self._elementwise_kind == "where":
            signature_kind = "where"
        params = {
            "elementwise_kind": elementwise_kind,
            "signature_kind": signature_kind,
        }
        return self._emit_standard(
            node_index, op_node, graph, params=params
        )

    def infer_shapes(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        op_spec = op_node.spec
        if op_spec.kind == OpKind.BINARY:
            if op_spec.name == "copy":
                output_shape = input_shapes[0]
                broadcast_shape = shape_utils.broadcast_output_shape(
                    op_spec.name, *input_shapes
                )
                if broadcast_shape != output_shape:
                    raise CodegenBackendError(
                        "codegen copy expects source to be broadcastable to the destination"
                    )
                return output_shape
            return shape_utils.broadcast_output_shape(
                op_spec.name, *input_shapes
            )
        if op_spec.kind == OpKind.WHERE:
            return shape_utils.broadcast_output_shape(
                op_spec.name, *input_shapes
            )
        if op_spec.kind in {OpKind.UNARY, OpKind.FILL}:
            return input_shapes[0]
        raise NotImplementedError(
            "Shape inference not implemented for kind "
            f"'{op_spec.kind.value}'."
        )


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
        parser = self._ctx.arg_parser
        if op_spec.name == "_to_copy":
            if dtype_info is None:
                return None
            op_node = parser.handle_to_copy_node(
                node, op_spec, dtype_info, shapes, strides, dtypes
            )
            return OpNodeBuildResult(op_node)
        if op_spec.kind == OpKind.FILL:
            if dtype_info is None:
                return None
            op_node = parser.handle_fill_node(
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

        if op_spec.name == "clamp_tensor":
            if not node.args:
                raise CodegenBackendError(
                    "codegen clamp_tensor expects one input"
                )
            if len(node.args) > 3:
                raise CodegenBackendError(
                    "codegen clamp_tensor expects one input"
                )
            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                raise self._ctx.analysis_service.error_expected_tensor(
                    op_spec.name
                )
            if input_node not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor(
                    op_spec.name
                )
            input_nodes.append(input_node)
            input_shapes.append(shapes[input_node])
            min_arg = node.args[1] if len(node.args) > 1 else None
            max_arg = node.args[2] if len(node.args) > 2 else None
            if "min" in node.kwargs:
                if len(node.args) > 1:
                    raise CodegenBackendError(
                        "codegen clamp_tensor expects min as a keyword"
                    )
                min_arg = node.kwargs["min"]
            if "max" in node.kwargs:
                if len(node.args) > 2:
                    raise CodegenBackendError(
                        "codegen clamp_tensor expects max as a keyword"
                    )
                max_arg = node.kwargs["max"]
            extra = set(node.kwargs) - {"min", "max"}
            if extra:
                raise CodegenBackendError(
                    "codegen clamp_tensor got unexpected kwargs: "
                    f"{sorted(extra)}"
                )
            if min_arg is not None:
                if not isinstance(min_arg, torch.fx.Node):
                    raise self._ctx.analysis_service.error_expected_tensor(
                        op_spec.name
                    )
                if min_arg not in shapes:
                    raise self._ctx.analysis_service.error_expected_tensor(
                        op_spec.name
                    )
                input_nodes.append(min_arg)
                input_shapes.append(shapes[min_arg])
            if max_arg is not None:
                if not isinstance(max_arg, torch.fx.Node):
                    raise self._ctx.analysis_service.error_expected_tensor(
                        op_spec.name
                    )
                if max_arg not in shapes:
                    raise self._ctx.analysis_service.error_expected_tensor(
                        op_spec.name
                    )
                input_nodes.append(max_arg)
                input_shapes.append(shapes[max_arg])
            param_values["has_min"] = min_arg is not None
            param_values["has_max"] = max_arg is not None
        elif op_spec.kind == OpKind.BINARY and len(node.args) == 2:
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
                    param_values["scalar"] = parser.parse_bitwise_scalar(
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
                        param_values["scalar"] = parser.parse_bitwise_scalar(
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
                input_node, param_values = parser.parse_parametric_unary_args(
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
                ) = parser.parse_where_inputs(op_spec, node, shapes, scalar_values)
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
        if op_spec.name in {"clamp", "clamp_tensor"} and dtype_info.torch_dtype is torch.bool:
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


def build_handlers(context: ElementwiseContext) -> Dict[OpKind, OpKindHandler]:
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


class ElementwiseKindHandlerFactory:
    def build_handlers(
        self, context_provider: HandlerContextProvider
    ) -> Dict[OpKind, OpKindHandler]:
        return build_handlers(context_provider.elementwise)


def build_kind_handler_registrations() -> Dict[OpKind, "KindHandlerRegistration"]:
    from codegen_backend.emitters.registry import KindHandlerRegistration

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


__all__ = [
    "ElementwiseKindHandlerFactory",
    "build_handlers",
    "build_kind_handler_registrations",
]
