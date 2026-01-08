from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.fx

from codegen_backend.analysis_helpers import (
    error_expected_tensor,
    parse_bitwise_scalar as _parse_bitwise_scalar,
    parse_constant_float,
)
from codegen_backend.c_types import _normalize_scalar_value
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.specs import _OpSpec


class ElementwiseArgParser:
    def parse_parametric_unary_args(
        self, op_name: str, node: torch.fx.Node
    ) -> Tuple[torch.fx.Node, Dict[str, object]]:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_name} expects one input")
        input_node = node.args[0]
        params: Dict[str, object] = {}
        if op_name == "gelu":
            if len(node.args) > 2:
                raise CodegenBackendError("codegen gelu expects one input")
            if len(node.args) > 1:
                params["approximate"] = node.args[1]
            if "approximate" in node.kwargs:
                if len(node.args) > 1:
                    raise CodegenBackendError(
                        "codegen gelu expects approximate as a keyword"
                    )
                params["approximate"] = node.kwargs["approximate"]
            extra = set(node.kwargs) - {"approximate"}
            if extra:
                raise CodegenBackendError(
                    f"codegen gelu got unexpected kwargs: {sorted(extra)}"
                )
            approximate = params.get("approximate", "none")
            if isinstance(approximate, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen gelu expects approximate to be constant"
                )
            if approximate is None:
                approximate = "none"
            if approximate not in {"none", "tanh"}:
                raise CodegenBackendError(
                    "codegen gelu expects approximate to be 'none' or 'tanh'"
                )
            params["approximate"] = approximate
            return input_node, params
        if op_name == "elu":
            if len(node.args) > 4:
                raise CodegenBackendError("codegen elu expects one input")
            args = list(node.args[1:])
            kwargs = dict(node.kwargs)
            for name in ("alpha", "scale", "input_scale"):
                if name in kwargs and args:
                    raise CodegenBackendError(
                        f"codegen elu got multiple values for {name}"
                    )
                if args:
                    params[name] = args.pop(0)
                elif name in kwargs:
                    params[name] = kwargs[name]
            extra = set(kwargs) - {"alpha", "scale", "input_scale"}
            if extra:
                raise CodegenBackendError(
                    f"codegen elu got unexpected kwargs: {sorted(extra)}"
                )
            params["alpha"] = parse_constant_float(
                op_name, "alpha", params.get("alpha", 1.0)
            )
            params["scale"] = parse_constant_float(
                op_name, "scale", params.get("scale", 1.0)
            )
            params["input_scale"] = parse_constant_float(
                op_name, "input_scale", params.get("input_scale", 1.0)
            )
            return input_node, params
        if op_name == "leaky_relu":
            if len(node.args) > 2:
                raise CodegenBackendError("codegen leaky_relu expects one input")
            if len(node.args) > 1:
                params["negative_slope"] = node.args[1]
            if "negative_slope" in node.kwargs:
                if len(node.args) > 1:
                    raise CodegenBackendError(
                        "codegen leaky_relu expects negative_slope as a keyword"
                    )
                params["negative_slope"] = node.kwargs["negative_slope"]
            extra = set(node.kwargs) - {"negative_slope"}
            if extra:
                raise CodegenBackendError(
                    f"codegen leaky_relu got unexpected kwargs: {sorted(extra)}"
                )
            params["negative_slope"] = parse_constant_float(
                op_name, "negative_slope", params.get("negative_slope", 0.01)
            )
            return input_node, params
        if op_name == "softplus":
            if len(node.args) > 3:
                raise CodegenBackendError("codegen softplus expects one input")
            if len(node.args) > 1:
                params["beta"] = node.args[1]
            if len(node.args) > 2:
                params["threshold"] = node.args[2]
            if "beta" in node.kwargs:
                if len(node.args) > 1:
                    raise CodegenBackendError(
                        "codegen softplus expects beta as a keyword"
                    )
                params["beta"] = node.kwargs["beta"]
            if "threshold" in node.kwargs:
                if len(node.args) > 2:
                    raise CodegenBackendError(
                        "codegen softplus expects threshold as a keyword"
                    )
                params["threshold"] = node.kwargs["threshold"]
            extra = set(node.kwargs) - {"beta", "threshold"}
            if extra:
                raise CodegenBackendError(
                    f"codegen softplus got unexpected kwargs: {sorted(extra)}"
                )
            params["beta"] = parse_constant_float(
                op_name, "beta", params.get("beta", 1.0)
            )
            params["threshold"] = parse_constant_float(
                op_name, "threshold", params.get("threshold", 20.0)
            )
            return input_node, params
        if op_name == "clamp":
            if len(node.args) > 3:
                raise CodegenBackendError("codegen clamp expects one input")
            if len(node.args) > 1:
                params["min_val"] = node.args[1]
            if len(node.args) > 2:
                params["max_val"] = node.args[2]
            if "min" in node.kwargs:
                if len(node.args) > 1:
                    raise CodegenBackendError("codegen clamp expects min as a keyword")
                params["min_val"] = node.kwargs["min"]
            if "max" in node.kwargs:
                if len(node.args) > 2:
                    raise CodegenBackendError("codegen clamp expects max as a keyword")
                params["max_val"] = node.kwargs["max"]
            extra = set(node.kwargs) - {"min", "max"}
            if extra:
                raise CodegenBackendError(
                    f"codegen clamp got unexpected kwargs: {sorted(extra)}"
                )
            if params.get("min_val") is not None:
                params["min_val"] = parse_constant_float(
                    op_name, "min", params["min_val"]
                )
            if params.get("max_val") is not None:
                params["max_val"] = parse_constant_float(
                    op_name, "max", params["max_val"]
                )
            return input_node, params
        if op_name == "hardtanh":
            if len(node.args) > 3:
                raise CodegenBackendError("codegen hardtanh expects one input")
            if len(node.args) > 1:
                params["min_val"] = node.args[1]
            if len(node.args) > 2:
                params["max_val"] = node.args[2]
            if "min_val" in node.kwargs:
                if len(node.args) > 1:
                    raise CodegenBackendError(
                        "codegen hardtanh expects min_val as a keyword"
                    )
                params["min_val"] = node.kwargs["min_val"]
            if "max_val" in node.kwargs:
                if len(node.args) > 2:
                    raise CodegenBackendError(
                        "codegen hardtanh expects max_val as a keyword"
                    )
                params["max_val"] = node.kwargs["max_val"]
            extra = set(node.kwargs) - {"min_val", "max_val"}
            if extra:
                raise CodegenBackendError(
                    f"codegen hardtanh got unexpected kwargs: {sorted(extra)}"
                )
            params["min_val"] = parse_constant_float(
                op_name, "min_val", params.get("min_val", -1.0)
            )
            params["max_val"] = parse_constant_float(
                op_name, "max_val", params.get("max_val", 1.0)
            )
            return input_node, params
        raise CodegenBackendError(f"Unsupported parametric op: {op_name}")

    def parse_bitwise_scalar(
        self, op_name: str, value: object, dtype: torch.dtype
    ) -> object:
        return _parse_bitwise_scalar(op_name, value, dtype)

    def parse_where_inputs(
        self,
        op_spec: _OpSpec,
        node: torch.fx.Node,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        scalar_values: Dict[torch.fx.Node, object],
    ) -> Tuple[List[torch.fx.Node], List[Tuple[int, ...]], Dict[str, object]]:
        if len(node.args) < 3:
            raise CodegenBackendError(f"codegen {op_spec.name} expects three inputs")
        cond_arg, a_arg, b_arg = node.args[:3]
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[Tuple[int, ...]] = []
        params: Dict[str, object] = {}

        def add_tensor_arg(arg: object) -> None:
            if not isinstance(arg, torch.fx.Node):
                raise error_expected_tensor(op_spec.name)
            if arg not in shapes:
                raise error_expected_tensor(op_spec.name)
            input_nodes.append(arg)
            input_shapes.append(shapes[arg])

        def add_where_value(arg: object, scalar_key: str) -> None:
            if isinstance(arg, torch.fx.Node):
                if arg in shapes:
                    input_nodes.append(arg)
                    input_shapes.append(shapes[arg])
                    return
                if arg in scalar_values:
                    params[scalar_key] = _normalize_scalar_value(
                        op_spec.name, scalar_values[arg]
                    )
                    return
                raise error_expected_tensor(op_spec.name)
            params[scalar_key] = _normalize_scalar_value(op_spec.name, arg)

        add_tensor_arg(cond_arg)
        add_where_value(a_arg, "a_scalar")
        add_where_value(b_arg, "b_scalar")
        return input_nodes, input_shapes, params

    def handle_fill_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        inplace_input: int | None,
        *,
        infer_output_shape,
    ) -> _OpNode:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_spec.name} expects inputs")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
            raise error_expected_tensor(op_spec.name)
        value = None
        if len(node.args) > 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects a tensor and scalar value"
            )
        if len(node.args) > 1:
            value = node.args[1]
        if "value" in node.kwargs:
            if value is not None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects a single scalar value"
                )
            value = node.kwargs["value"]
        if op_spec.name == "full_like":
            allowed = {
                "fill_value",
                "value",
                "dtype",
                "layout",
                "device",
                "pin_memory",
                "memory_format",
            }
            extra = set(node.kwargs) - allowed
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
            if "fill_value" in node.kwargs:
                if value is not None:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects a single scalar value"
                    )
                value = node.kwargs["fill_value"]
            if "value" in node.kwargs:
                if value is not None:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects a single scalar value"
                    )
                value = node.kwargs["value"]
            dtype = node.kwargs.get("dtype")
            if isinstance(dtype, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects dtype to be a constant"
                )
            if dtype is not None and dtype is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects dtype to match the graph dtype"
                )
            layout = node.kwargs.get("layout")
            if isinstance(layout, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects layout to be a constant"
                )
            if layout is not None and layout is not torch.strided:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects layout to be None or torch.strided"
                )
            device = node.kwargs.get("device")
            if isinstance(device, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects device to be a constant"
                )
            if device is not None and device != "cpu" and device != torch.device("cpu"):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects device to be None or cpu"
                )
            pin_memory = node.kwargs.get("pin_memory")
            if isinstance(pin_memory, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects pin_memory to be a constant"
                )
            if pin_memory not in (None, False):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects pin_memory to be False"
                )
            memory_format = node.kwargs.get("memory_format")
            if isinstance(memory_format, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects memory_format to be a constant"
                )
            if memory_format not in (
                None,
                torch.contiguous_format,
                torch.preserve_format,
            ):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects memory_format to be None, "
                    "torch.contiguous_format, or torch.preserve_format"
                )
            if memory_format in (None, torch.preserve_format):
                if strides[input_arg] != _contiguous_strides(shapes[input_arg]):
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects contiguous input when memory_format preserves format"
                    )
        elif node.kwargs and set(node.kwargs) != {"value"}:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects only 'value' as a keyword argument"
            )
        if value is None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects a scalar value"
            )
        if dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        scalar_value = _normalize_scalar_value(op_spec.name, value)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=inplace_input,
            params={"value": scalar_value},
        )
        output_shape = infer_output_shape(op_node, [shapes[input_arg]])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        if inplace_input is not None:
            strides[node] = strides[input_arg]
        else:
            strides[node] = _contiguous_strides(output_shape)
        return op_node

    def handle_to_copy_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
            raise error_expected_tensor(op_spec.name)
        if dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        if node.kwargs:
            dtype = node.kwargs.get("dtype")
            if isinstance(dtype, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects dtype to be a constant"
                )
            if dtype is not None and dtype is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects dtype to match the graph dtype"
                )
            device = node.kwargs.get("device")
            if isinstance(device, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects device to be a constant"
                )
            if device not in (None, "cpu", torch.device("cpu")):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects device to be None or cpu"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=shapes[input_arg],
            inplace_input=None,
            params={},
        )
        shapes[node] = shapes[input_arg]
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = strides[input_arg]
        return op_node


_DEFAULT_PARSER = ElementwiseArgParser()


def parse_parametric_unary_args(
    op_name: str, node: torch.fx.Node
) -> Tuple[torch.fx.Node, Dict[str, object]]:
    return _DEFAULT_PARSER.parse_parametric_unary_args(op_name, node)


def parse_bitwise_scalar(
    op_name: str, value: object, dtype: torch.dtype
) -> object:
    return _DEFAULT_PARSER.parse_bitwise_scalar(op_name, value, dtype)


def parse_where_inputs(
    op_spec: _OpSpec,
    node: torch.fx.Node,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    scalar_values: Dict[torch.fx.Node, object],
) -> Tuple[List[torch.fx.Node], List[Tuple[int, ...]], Dict[str, object]]:
    return _DEFAULT_PARSER.parse_where_inputs(op_spec, node, shapes, scalar_values)


def handle_fill_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    inplace_input: int | None,
    *,
    infer_output_shape,
) -> _OpNode:
    return _DEFAULT_PARSER.handle_fill_node(
        node,
        op_spec,
        dtype_info,
        shapes,
        strides,
        dtypes,
        inplace_input,
        infer_output_shape=infer_output_shape,
    )


def handle_to_copy_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    return _DEFAULT_PARSER.handle_to_copy_node(
        node, op_spec, dtype_info, shapes, strides, dtypes
    )


__all__ = [
    "ElementwiseArgParser",
    "handle_fill_node",
    "handle_to_copy_node",
    "parse_bitwise_scalar",
    "parse_parametric_unary_args",
    "parse_where_inputs",
]
