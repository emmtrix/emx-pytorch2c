import math
import numbers
import operator
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.fx

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _dtype_to_c_type, _input_c_type
from codegen_backend.dtypes import (
    _CODEGEN_DTYPES,
    _CodegenDType,
    _EMBEDDING_INDEX_DTYPES,
    _INTEGER_CODEGEN_DTYPES,
)
from codegen_backend.graph import _GenericGraph, _OpNode
from codegen_backend.emitters.base import _is_contiguous
from codegen_backend.indexing import (
    _contiguous_strides,
    _emit_strided_access,
    _format_strided_access,
)
from codegen_backend.groups.builtin.reductions.parsing import ReductionsArgParser
from codegen_backend.groups.builtin.tensor.parsing import (
    parse_addmm_like_args,
    parse_cumsum_args,
    parse_diagonal_args,
    parse_gather_args,
    parse_linear_args,
    parse_resize_size,
)
from codegen_backend.groups.registry import get_group_registry
from codegen_backend.kinds import OpNodeBuildResult
from codegen_backend.analysis_helpers import (
    channels_last_3d_strides,
    channels_last_strides,
    normalize_as_strided_sequence,
    resolve_scalar_arg,
)
from codegen_backend.compiler import Compiler
from codegen_backend.emitter import Emitter
from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.graph_builder import GraphBuilder
from codegen_backend.ops_parsing import _parse_where_inputs
from codegen_backend.parser import Parser
from codegen_backend.parsing.common import (
    parse_constant_bool,
    parse_constant_float,
    parse_constant_int,
)
from codegen_backend.services import GraphAnalysisService
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.templates import get_template_env


def _infer_output_shape(
    op_node: _OpNode,
    input_shapes: Sequence[Tuple[int, ...]],
    *,
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> Tuple[int, ...]:
    handler = kind_handlers.get(op_node.spec.kind)
    if handler is None:
        raise CodegenBackendError(
            f"codegen backend does not support kind '{op_node.spec.kind.value}'"
        )
    return handler.infer_shapes(op_node, input_shapes)


def _normalize_flip_dims(
    op_name: str, dims: object, rank: int
) -> Tuple[int, ...]:
    if dims is None:
        raise CodegenBackendError(f"codegen {op_name} expects dims to be provided")
    if isinstance(dims, torch.fx.Node):
        raise CodegenBackendError(
            f"codegen {op_name} expects dims to be an int or tuple of ints"
        )
    if isinstance(dims, (tuple, list)):
        dims_list = list(dims)
    else:
        dims_list = [dims]
    if not dims_list:
        return ()
    if rank == 0:
        raise CodegenBackendError(
            f"codegen {op_name} expects dims to be within the input rank"
        )
    normalized = []
    seen = set()
    for item in dims_list:
        if isinstance(item, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be an int or tuple of ints"
            )
        try:
            dim = operator.index(item)
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be an int or tuple of ints"
            ) from exc
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be within the input rank"
            )
        if dim in seen:
            raise CodegenBackendError(
                f"codegen {op_name} expects dims to be unique"
            )
        seen.add(dim)
        normalized.append(dim)
    return tuple(normalized)
def _error_expected_tensor(op_name: str) -> CodegenBackendError:
    return CodegenBackendError(f"codegen {op_name} expects tensor inputs only")


def _error_kwarg_specified_once(op_name: str, kwarg: str) -> CodegenBackendError:
    return CodegenBackendError(f"codegen {op_name} expects {kwarg} to be specified once")


def _normalize_param(normalizer: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return normalizer(*args, **kwargs)
    except ValueError as exc:
        raise CodegenBackendError(str(exc)) from exc


def _parse_parametric_unary_args(
    op_name: str, node: torch.fx.Node
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
                raise CodegenBackendError("codegen gelu expects approximate as a keyword")
            params["approximate"] = node.kwargs["approximate"]
        extra = set(node.kwargs) - {"approximate"}
        if extra:
            raise CodegenBackendError(
                f"codegen gelu got unexpected kwargs: {sorted(extra)}"
            )
        approximate = params.get("approximate", "none")
        if isinstance(approximate, torch.fx.Node):
            raise CodegenBackendError("codegen gelu expects approximate to be constant")
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
                raise CodegenBackendError(f"codegen elu got multiple values for {name}")
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
                raise CodegenBackendError("codegen softplus expects beta as a keyword")
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
                raise CodegenBackendError("codegen hardtanh expects min_val as a keyword")
            params["min_val"] = node.kwargs["min_val"]
        if "max_val" in node.kwargs:
            if len(node.args) > 2:
                raise CodegenBackendError("codegen hardtanh expects max_val as a keyword")
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


def _validate_addmm_like_scalars(
    op_name: str, dtype: torch.dtype, alpha: float, beta: float
) -> None:
    if dtype in _INTEGER_CODEGEN_DTYPES or dtype is torch.bool:
        for name, value in (("alpha", alpha), ("beta", beta)):
            if not float(value).is_integer():
                raise CodegenBackendError(
                    f"codegen {op_name} expects {name} to be an integer for integral tensors"
                )


def _handle_flip_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
    input_node = node.args[0]
    if not isinstance(input_node, torch.fx.Node) or input_node not in shapes:
        raise _error_expected_tensor(op_spec.name)
    dims = None
    if len(node.args) >= 2:
        dims = node.args[1]
        if len(node.args) > 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects one input and dims"
            )
    if "dims" in node.kwargs:
        if dims is not None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects dims only once"
            )
        dims = node.kwargs["dims"]
    if node.kwargs and "dims" not in node.kwargs:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects dims as the only keyword argument"
        )
    input_shape = shapes[input_node]
    if dtypes[input_node] is not dtype_info.torch_dtype:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    normalized_dims = _normalize_flip_dims(op_spec.name, dims, len(input_shape))
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_node],
        output_shape=(),
        inplace_input=None,
        params={"dims": normalized_dims},
    )
    output_shape = _infer_output_shape(
        op_node, [input_shape], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node


def _handle_batch_norm_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    scalar_values: Dict[torch.fx.Node, object],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    has_training_flag = op_spec.name == "_native_batch_norm_legit"
    expected_inputs = 8 if has_training_flag else 7
    if len(node.args) < expected_inputs:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects {expected_inputs} inputs"
        )
    if node.kwargs:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects positional args only"
        )
    if has_training_flag:
        (
            input_arg,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
        ) = node.args[:8]
        if isinstance(training, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant training flag"
            )
        if training not in (False, 0):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only training=False"
            )
    else:
        (
            input_arg,
            weight,
            bias,
            running_mean,
            running_var,
            momentum,
            eps,
        ) = node.args[:7]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if not isinstance(running_mean, torch.fx.Node) or running_mean not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if not isinstance(running_var, torch.fx.Node) or running_var not in shapes:
        raise _error_expected_tensor(op_spec.name)
    weight_node = None
    bias_node = None
    if weight is not None:
        if not isinstance(weight, torch.fx.Node) or weight not in shapes:
            raise _error_expected_tensor(op_spec.name)
        weight_node = weight
    if bias is not None:
        if not isinstance(bias, torch.fx.Node) or bias not in shapes:
            raise _error_expected_tensor(op_spec.name)
        bias_node = bias
    if dtype_info.torch_dtype is not torch.float32:
        raise CodegenBackendError(f"codegen {op_spec.name} supports only torch.float32")
    if dtypes[input_arg] is not torch.float32:
        raise CodegenBackendError(f"codegen {op_spec.name} supports only torch.float32")
    input_shape = shapes[input_arg]
    if len(input_shape) < 2:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects at least 2D inputs"
        )
    if not _is_contiguous(input_shape, strides[input_arg]):
        raise CodegenBackendError(
            f"codegen {op_spec.name} requires contiguous input"
        )
    channels = input_shape[1]
    for stat_arg, name in (
        (running_mean, "running_mean"),
        (running_var, "running_var"),
    ):
        stat_shape = shapes[stat_arg]
        if stat_shape != (channels,):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects {name} shape to match channels"
            )
        if not _is_contiguous(stat_shape, strides[stat_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous stats"
            )
        if dtypes[stat_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
            )
    if weight_node is not None:
        if shapes[weight_node] != (channels,):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects weight shape to match channels"
            )
        if dtypes[weight_node] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
            )
    if bias_node is not None:
        if shapes[bias_node] != (channels,):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects bias shape to match channels"
            )
        if dtypes[bias_node] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
            )
    try:
        _ = float(resolve_scalar_arg(op_spec.name, momentum, scalar_values))
        eps_value = float(resolve_scalar_arg(op_spec.name, eps, scalar_values))
    except (TypeError, ValueError) as exc:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects eps to be a float"
        ) from exc
    inputs = [input_arg, running_mean, running_var]
    if weight_node is not None:
        inputs.append(weight_node)
    if bias_node is not None:
        inputs.append(bias_node)
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=inputs,
        output_shape=(),
        inplace_input=None,
        params={
            "eps": eps_value,
            "has_weight": weight_node is not None,
            "has_bias": bias_node is not None,
        },
    )
    output_shape = _infer_output_shape(
        op_node, [input_shape], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node



def _handle_pdist_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if not node.args:
        raise CodegenBackendError("codegen _pdist_forward expects one input")
    if len(node.args) > 2:
        raise CodegenBackendError("codegen _pdist_forward expects at most two inputs")
    if node.kwargs:
        raise CodegenBackendError("codegen _pdist_forward expects positional args only")
    input_arg = node.args[0]
    p_value = node.args[1] if len(node.args) > 1 else 2.0
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtype_info.torch_dtype is not torch.float32:
        raise CodegenBackendError("codegen _pdist_forward supports only torch.float32")
    if dtypes[input_arg] is not torch.float32:
        raise CodegenBackendError("codegen _pdist_forward supports only torch.float32")
    if isinstance(p_value, torch.fx.Node):
        raise CodegenBackendError("codegen _pdist_forward expects constant p value")
    try:
        p_value = float(p_value)
    except (TypeError, ValueError) as exc:
        raise CodegenBackendError("codegen _pdist_forward expects p to be a float") from exc
    if p_value != 2.0:
        raise CodegenBackendError("codegen _pdist_forward supports only p=2")
    input_shape = shapes[input_arg]
    if len(input_shape) != 2:
        raise CodegenBackendError("codegen _pdist_forward expects a 2D input tensor")
    if not _is_contiguous(input_shape, strides[input_arg]):
        raise CodegenBackendError("codegen _pdist_forward requires contiguous input")
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=(),
        inplace_input=None,
        params={},
    )
    output_shape = _infer_output_shape(
        op_node, [input_shape], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node



def _handle_cdist_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if len(node.args) < 3:
        raise CodegenBackendError(
            "codegen _cdist_forward expects two inputs and a p value"
        )
    if len(node.args) > 4:
        raise CodegenBackendError(
            "codegen _cdist_forward expects at most four inputs"
        )
    if node.kwargs:
        raise CodegenBackendError(
            "codegen _cdist_forward expects positional args only"
        )
    x1_arg = node.args[0]
    x2_arg = node.args[1]
    p_value = node.args[2]
    compute_mode = node.args[3] if len(node.args) > 3 else None
    if not isinstance(x1_arg, torch.fx.Node) or x1_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if not isinstance(x2_arg, torch.fx.Node) or x2_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtype_info.torch_dtype is not torch.float32:
        raise CodegenBackendError("codegen _cdist_forward supports only torch.float32")
    if dtypes[x1_arg] is not torch.float32 or dtypes[x2_arg] is not torch.float32:
        raise CodegenBackendError("codegen _cdist_forward supports only torch.float32")
    if isinstance(p_value, torch.fx.Node):
        raise CodegenBackendError("codegen _cdist_forward expects constant p value")
    try:
        p_value = float(p_value)
    except (TypeError, ValueError) as exc:
        raise CodegenBackendError(
            "codegen _cdist_forward expects p to be a float"
        ) from exc
    if p_value != 2.0:
        raise CodegenBackendError("codegen _cdist_forward supports only p=2")
    if isinstance(compute_mode, torch.fx.Node):
        raise CodegenBackendError(
            "codegen _cdist_forward expects constant compute_mode value"
        )
    if compute_mode not in (None, 0):
        raise CodegenBackendError(
            "codegen _cdist_forward supports only compute_mode=None or 0"
        )
    x1_shape = shapes[x1_arg]
    x2_shape = shapes[x2_arg]
    if len(x1_shape) != 2 or len(x2_shape) != 2:
        raise CodegenBackendError(
            "codegen _cdist_forward expects 2D input tensors"
        )
    if x1_shape[1] != x2_shape[1]:
        raise CodegenBackendError(
            "codegen _cdist_forward expects matching feature dimensions"
        )
    if not _is_contiguous(x1_shape, strides[x1_arg]) or not _is_contiguous(
        x2_shape, strides[x2_arg]
    ):
        raise CodegenBackendError(
            "codegen _cdist_forward requires contiguous inputs"
        )
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[x1_arg, x2_arg],
        output_shape=(),
        inplace_input=None,
        params={},
    )
    output_shape = _infer_output_shape(
        op_node, [x1_shape, x2_shape], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node



def _handle_addmm_like_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    inplace_input: int | None,
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    op_name = op_spec.name
    input_nodes, alpha, beta = parse_addmm_like_args(op_name, node)
    input_shapes = []
    for arg in input_nodes:
        if not isinstance(arg, torch.fx.Node):
            raise _error_expected_tensor(op_name)
        if arg not in shapes:
            raise _error_expected_tensor(op_name)
        input_shapes.append(shapes[arg])
    input_dtypes = [dtypes[arg] for arg in input_nodes]
    if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
        raise CodegenBackendError(
            f"codegen {op_name} expects inputs to share the graph dtype"
        )
    _validate_addmm_like_scalars(op_name, dtype_info.torch_dtype, alpha, beta)
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=list(input_nodes),
        output_shape=(),
        inplace_input=inplace_input,
        params={"alpha": alpha, "beta": beta},
    )
    output_shape = _infer_output_shape(
        op_node, input_shapes, kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    if inplace_input is not None:
        strides[node] = strides[input_nodes[inplace_input]]
    else:
        strides[node] = _contiguous_strides(output_shape)
    return op_node


def _handle_linear_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    input_arg, weight_arg, bias_arg = parse_linear_args(op_spec.name, node)
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if not isinstance(weight_arg, torch.fx.Node) or weight_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    input_nodes = [input_arg, weight_arg]
    input_shapes = [shapes[input_arg], shapes[weight_arg]]
    if bias_arg is not None:
        if not isinstance(bias_arg, torch.fx.Node) or bias_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        input_nodes.append(bias_arg)
        input_shapes.append(shapes[bias_arg])
    input_dtypes = [dtypes[arg] for arg in input_nodes]
    if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=input_nodes,
        output_shape=(),
        inplace_input=None,
        params={"has_bias": bias_arg is not None},
    )
    output_shape = _infer_output_shape(
        op_node, input_shapes, kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node


def _handle_diagonal_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    offset, dim1, dim2 = parse_diagonal_args(
        op_spec.name, node, shapes[input_arg]
    )
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=(),
        params={"offset": offset, "dim1": dim1, "dim2": dim2},
    )
    output_shape = _infer_output_shape(
        op_node, [shapes[input_arg]], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node

def _handle_cumsum_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtype_info.torch_dtype is torch.bool:
        raise CodegenBackendError("codegen cumsum does not support torch.bool tensors")
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    dim, dtype_override = parse_cumsum_args(
        op_spec.name, node, shapes[input_arg]
    )
    output_dtype = dtype_override or dtype_info.torch_dtype
    if output_dtype not in _CODEGEN_DTYPES:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects dtype to be torch.float32, torch.int8, or torch.int32"
        )
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=(),
        inplace_input=None,
        params={"dim": dim},
    )
    output_shape = _infer_output_shape(
        op_node, [shapes[input_arg]], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = output_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node


def _handle_constant_pad_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if not node.args:
        raise CodegenBackendError(
            "codegen constant_pad_nd expects at least one argument"
        )
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise CodegenBackendError(
            "codegen constant_pad_nd expects inputs to share the graph dtype"
        )
    pad = None
    value = 0
    if len(node.args) > 1:
        pad = node.args[1]
    if len(node.args) > 2:
        value = node.args[2]
    if len(node.args) > 3:
        raise CodegenBackendError(
            "codegen constant_pad_nd expects at most three positional arguments"
        )
    extra_kwargs = set(node.kwargs) - {"pad", "value"}
    if extra_kwargs:
        raise CodegenBackendError(
            "codegen constant_pad_nd expects only pad and value kwargs"
        )
    if "pad" in node.kwargs:
        if pad is not None:
            raise _error_kwarg_specified_once(op_spec.name, "pad")
        pad = node.kwargs["pad"]
    if "value" in node.kwargs:
        if len(node.args) > 2:
            raise _error_kwarg_specified_once(op_spec.name, "value")
        value = node.kwargs["value"]
    if pad is None:
        raise CodegenBackendError("codegen constant_pad_nd expects a pad argument")
    if isinstance(pad, torch.fx.Node):
        raise CodegenBackendError(
            "codegen constant_pad_nd expects pad to be a constant list"
        )
    if not isinstance(pad, (tuple, list)):
        raise CodegenBackendError(
            "codegen constant_pad_nd expects pad to be a list or tuple"
        )
    if len(pad) % 2 != 0:
        raise CodegenBackendError(
            "codegen constant_pad_nd expects pad to have an even number of values"
        )
    pad_values = []
    for item in pad:
        if isinstance(item, numbers.Real) and not float(item).is_integer():
            raise CodegenBackendError(
                "codegen constant_pad_nd expects pad values to be integers"
            )
        try:
            pad_values.append(int(operator.index(item)))
        except TypeError:
            try:
                pad_values.append(int(item))
            except (TypeError, ValueError) as exc:
                raise CodegenBackendError(
                    "codegen constant_pad_nd expects pad values to be integers"
                ) from exc
    input_shape = shapes[input_arg]
    rank = len(input_shape)
    if len(pad_values) > 2 * rank:
        raise CodegenBackendError(
            "codegen constant_pad_nd expects pad to have at most 2 * input rank values"
        )
    pad_before = [0] * rank
    pad_after = [0] * rank
    for idx in range(len(pad_values) // 2):
        dim = rank - 1 - idx
        pad_before[dim] = pad_values[2 * idx]
        pad_after[dim] = pad_values[2 * idx + 1]
    if isinstance(value, torch.fx.Node):
        raise CodegenBackendError(
            "codegen constant_pad_nd expects a constant padding value"
        )
    value = _normalize_scalar_value(op_spec.name, value)
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=(),
        params={
            "pad_before": tuple(pad_before),
            "pad_after": tuple(pad_after),
            "value": value,
        },
    )
    output_shape = _infer_output_shape(
        op_node, [input_shape], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node


def _handle_gather_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    input_arg, dim, index, sparse_grad = parse_gather_args(node)
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if not isinstance(index, torch.fx.Node) or index not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise CodegenBackendError(
            "codegen gather expects input to match the graph dtype"
        )
    if dtypes[index] not in _EMBEDDING_INDEX_DTYPES:
        raise CodegenBackendError(
            "codegen gather expects index dtype to be torch.int32 or torch.int64"
        )
    if isinstance(sparse_grad, torch.fx.Node):
        raise CodegenBackendError("codegen gather expects sparse_grad to be False")
    if sparse_grad not in (False, 0, None):
        raise CodegenBackendError("codegen gather supports only sparse_grad=False")
    input_shape = shapes[input_arg]
    index_shape = shapes[index]
    if not input_shape:
        raise CodegenBackendError("codegen gather expects input to have at least 1 dimension")
    if len(index_shape) != len(input_shape):
        raise CodegenBackendError(
            "codegen gather expects index to have the same rank as input"
        )
    dim_value = parse_constant_int(op_spec.name, "dim", dim)
    if dim_value < 0:
        dim_value += len(input_shape)
    if dim_value < 0 or dim_value >= len(input_shape):
        raise CodegenBackendError("codegen gather dim is out of range")
    for idx, (input_dim, index_dim) in enumerate(
        zip(input_shape, index_shape)
    ):
        if idx == dim_value:
            continue
        if input_dim != index_dim:
            raise CodegenBackendError(
                "codegen gather expects index shape to match input shape"
            )
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg, index],
        output_shape=(),
        inplace_input=None,
        params={"dim": dim_value},
    )
    output_shape = _infer_output_shape(
        op_node, [input_shape, index_shape], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return op_node


def _handle_fill_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    inplace_input: int | None,
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_spec.name} expects inputs")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
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
    output_shape = _infer_output_shape(
        op_node, [shapes[input_arg]], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    if inplace_input is not None:
        strides[node] = strides[input_arg]
    else:
        strides[node] = _contiguous_strides(output_shape)
    return op_node


def _handle_view_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise CodegenBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    if op_spec.name == "as_strided":
        if len(node.args) > 4:
            raise CodegenBackendError(
                "codegen as_strided expects at most four inputs"
            )
        size = node.args[1] if len(node.args) > 1 else None
        stride = node.args[2] if len(node.args) > 2 else None
        storage_offset = node.args[3] if len(node.args) > 3 else None
        if node.kwargs:
            if "size" in node.kwargs:
                if size is not None:
                    raise _error_kwarg_specified_once(op_spec.name, "size")
                size = node.kwargs["size"]
            if "stride" in node.kwargs:
                if stride is not None:
                    raise _error_kwarg_specified_once(op_spec.name, "stride")
                stride = node.kwargs["stride"]
            if "storage_offset" in node.kwargs:
                if storage_offset is not None:
                    raise _error_kwarg_specified_once(
                        op_spec.name, "storage_offset"
                    )
                storage_offset = node.kwargs["storage_offset"]
            extra = set(node.kwargs) - {"size", "stride", "storage_offset"}
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        if size is None or stride is None:
            raise CodegenBackendError("codegen as_strided expects size and stride")
        if isinstance(size, torch.fx.Node) or isinstance(stride, torch.fx.Node):
            raise CodegenBackendError(
                "codegen as_strided expects size/stride to be constants"
            )
        if storage_offset is None:
            storage_offset = 0
        if isinstance(storage_offset, torch.fx.Node):
            raise CodegenBackendError(
                "codegen as_strided expects storage_offset to be an int"
            )
        size_tuple = normalize_as_strided_sequence(op_spec.name, size, "size")
        stride_tuple = normalize_as_strided_sequence(
            op_spec.name, stride, "stride"
        )
        if len(size_tuple) != len(stride_tuple):
            raise CodegenBackendError(
                "codegen as_strided expects size and stride to match length"
            )
        storage_offset_value = int(operator.index(storage_offset))
        if storage_offset_value < 0:
            raise CodegenBackendError(
                "codegen as_strided expects storage_offset to be non-negative"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            params={
                "size": size_tuple,
                "view_strides": stride_tuple,
                "storage_offset": storage_offset_value,
            },
        )
        output_shape = _infer_output_shape(
            op_node, [shapes[input_arg]], kind_handlers=kind_handlers
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return op_node
    if op_spec.name == "reshape":
        shape_values: List[object] | None = None
        shape_arg = None
        if len(node.args) > 2:
            shape_values = list(node.args[1:])
        else:
            if len(node.args) > 1:
                shape_arg = node.args[1]
            if node.kwargs:
                if "shape" in node.kwargs:
                    if shape_arg is not None:
                        raise _error_kwarg_specified_once(op_spec.name, "shape")
                    shape_arg = node.kwargs["shape"]
                extra = set(node.kwargs) - {"shape"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if shape_arg is None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects a shape argument"
                )
            if isinstance(shape_arg, torch.Size):
                shape_values = list(shape_arg)
            elif isinstance(shape_arg, (tuple, list)):
                shape_values = list(shape_arg)
            else:
                shape_values = [shape_arg]
        if any(isinstance(value, torch.fx.Node) for value in shape_values):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects shape to be constant"
            )
        if not _is_contiguous(shapes[input_arg], strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects contiguous input"
            )
        output_shape: List[int] = []
        unknown_dim = None
        known_product = 1
        for dim in shape_values:
            dim_value = parse_constant_int(op_spec.name, "shape", dim)
            if dim_value == -1:
                if unknown_dim is not None:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects at most one -1 dim"
                    )
                unknown_dim = len(output_shape)
                output_shape.append(-1)
                continue
            if dim_value < -1:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects shape dims >= -1"
                )
            output_shape.append(dim_value)
            known_product *= dim_value
        input_numel = math.prod(shapes[input_arg])
        if unknown_dim is not None:
            if known_product == 0:
                if input_numel != 0:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects shape to match input numel"
                    )
                inferred = 0
            else:
                if input_numel % known_product != 0:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects shape to match input numel"
                    )
                inferred = input_numel // known_product
            output_shape[unknown_dim] = inferred
        elif known_product != input_numel:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects shape to match input numel"
            )
        output_shape_tuple = tuple(output_shape)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            params={
                "size": output_shape_tuple,
                "view_strides": _contiguous_strides(output_shape_tuple),
                "storage_offset": 0,
            },
        )
        output_shape = _infer_output_shape(
            op_node, [shapes[input_arg]], kind_handlers=kind_handlers
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return op_node
    if op_spec.name == "squeeze":
        input_shape = shapes[input_arg]
        input_strides = strides[input_arg]
        if node.target is torch.ops.aten.squeeze.dim:
            if len(node.args) > 2:
                raise CodegenBackendError(
                    "codegen squeeze expects at most two inputs"
                )
            dim = node.args[1] if len(node.args) > 1 else None
            if node.kwargs:
                if "dim" in node.kwargs:
                    if dim is not None:
                        raise _error_kwarg_specified_once(op_spec.name, "dim")
                    dim = node.kwargs["dim"]
                extra = set(node.kwargs) - {"dim"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if dim is None:
                raise CodegenBackendError("codegen squeeze expects dim to be an int")
            dim_value = parse_constant_int(op_spec.name, "dim", dim)
            if dim_value < 0:
                dim_value += len(input_shape)
            if dim_value < 0 or dim_value >= len(input_shape):
                raise CodegenBackendError("codegen squeeze dim is out of range")
            remove_dims = {
                dim_value
            } if input_shape[dim_value] == 1 else set()
        else:
            if len(node.args) > 2:
                raise CodegenBackendError(
                    "codegen squeeze expects at most two inputs"
                )
            dims = node.args[1] if len(node.args) > 1 else None
            if node.kwargs:
                if "dim" in node.kwargs:
                    if dims is not None:
                        raise _error_kwarg_specified_once(op_spec.name, "dim")
                    dims = node.kwargs["dim"]
                extra = set(node.kwargs) - {"dim"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if dims is None:
                raise CodegenBackendError("codegen squeeze expects dim to be a list")
            if isinstance(dims, torch.fx.Node) or not isinstance(dims, (tuple, list)):
                raise CodegenBackendError("codegen squeeze expects dim to be a list")
            dim_values = []
            for dim in dims:
                dim_value = parse_constant_int(op_spec.name, "dim", dim)
                if dim_value < 0:
                    dim_value += len(input_shape)
                if dim_value < 0 or dim_value >= len(input_shape):
                    raise CodegenBackendError("codegen squeeze dim is out of range")
                dim_values.append(dim_value)
            remove_dims = {
                dim for dim in set(dim_values) if input_shape[dim] == 1
            }
        view_strides = tuple(
            stride
            for dim, stride in enumerate(input_strides)
            if dim not in remove_dims
        )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            params={
                "squeeze_dims": tuple(sorted(remove_dims)),
                "view_strides": view_strides,
                "storage_offset": 0,
            },
        )
        output_shape = _infer_output_shape(
            op_node, [input_shape], kind_handlers=kind_handlers
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return op_node
    raise CodegenBackendError(f"Unsupported view op: {op_spec.name}")


def _handle_resize_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    inplace_input: int | None,
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _OpNode:
    memory_format = None
    if node.kwargs:
        if set(node.kwargs) != {"memory_format"}:
            raise CodegenBackendError(
                "codegen resize_ expects only memory_format as a keyword argument"
            )
        memory_format = node.kwargs.get("memory_format")
    if len(node.args) != 2:
        raise CodegenBackendError("codegen resize_ expects input and size arguments")
    input_arg, size_arg = node.args
    if not isinstance(input_arg, torch.fx.Node):
        raise _error_expected_tensor(op_spec.name)
    if input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise CodegenBackendError(
            "codegen resize_ expects inputs to share the graph dtype"
        )
    size = parse_resize_size(op_spec.name, size_arg)
    if any(dim < 0 for dim in size):
        raise CodegenBackendError(
            "codegen resize_ expects size values to be non-negative"
        )
    input_shape = shapes[input_arg]
    if math.prod(size) != math.prod(input_shape):
        raise CodegenBackendError(
            "codegen resize_ expects size to match the input numel"
        )
    op_node = _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=(),
        inplace_input=None,
        params={"size": size},
    )
    output_shape = _infer_output_shape(
        op_node, [input_shape], kind_handlers=kind_handlers
    )
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    if memory_format is None or memory_format is torch.contiguous_format:
        output_strides = _contiguous_strides(output_shape)
    elif memory_format is torch.channels_last:
        output_strides = channels_last_strides(output_shape)
    elif memory_format is torch.channels_last_3d:
        output_strides = channels_last_3d_strides(output_shape)
    else:
        raise CodegenBackendError("Unsupported memory formatPreserve")
    strides[node] = output_strides
    return op_node
class CodegenBackend:
    def __init__(
        self,
        *,
        group_registry: object | None = None,
        analysis_service: GraphAnalysisService | None = None,
        templates_env: object | None = None,
    ) -> None:
        self.group_registry = (
            group_registry if group_registry is not None else get_group_registry()
        )
        self.templates_env = (
            templates_env if templates_env is not None else get_template_env()
        )
        self._supported_ops: Dict[str, _OpSpec] | None = None
        self._target_registry: Dict[object, "_TargetInfo"] | None = None
        self._kind_handlers: Dict[OpKind, "OpKindHandler"] | None = None
        self._kind_handler_registrations: Dict[
            OpKind, KindHandlerRegistration
        ] | None = None
        self._analysis_service = (
            analysis_service
            if analysis_service is not None
            else GraphAnalysisService(lambda: self.kind_handlers)
        )
        self._parser = Parser(
            kind_handlers=lambda: self.kind_handlers,
            target_registry=lambda: self.target_registry,
        )
        self._graph_builder = GraphBuilder(
            group_registry=lambda: self.group_registry,
            kind_handlers=lambda: self.kind_handlers,
            parser=self._parser,
        )
        self._emitter = Emitter(
            templates_env=lambda: self.templates_env,
            kind_handlers=lambda: self.kind_handlers,
            kind_handler_registrations=lambda: self.kind_handler_registrations,
        )
        self._compiler = Compiler(self._graph_builder, self._emitter)
        self._context_provider = self.group_registry.build_context_provider(self)

    @property
    def supported_ops(self) -> Dict[str, _OpSpec]:
        if self._supported_ops is None:
            self._supported_ops = self.group_registry.merged_supported_ops()
        return self._supported_ops

    @property
    def target_registry(self) -> Dict[object, "_TargetInfo"]:
        if self._target_registry is None:
            self._target_registry = self.group_registry.merged_target_registry()
        return self._target_registry

    @property
    def kind_handlers(self) -> Dict[OpKind, "OpKindHandler"]:
        if self._kind_handlers is None:
            self._kind_handlers = self.group_registry.build_kind_handlers(
                self._context_provider
            )
        return self._kind_handlers

    @property
    def kind_handler_registrations(self) -> Dict[OpKind, KindHandlerRegistration]:
        if self._kind_handler_registrations is None:
            self._kind_handler_registrations = (
                self.group_registry.merged_kind_handler_registrations()
            )
        return self._kind_handler_registrations

    @property
    def analysis_service(self) -> GraphAnalysisService:
        return self._analysis_service

    def get_generic_source(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> str:
        return self._compiler.get_source(gm, example_inputs)

    def codegen_generic_backend(
        self, gm: torch.fx.GraphModule, example_inputs: List[object]
    ) -> Callable[..., torch.Tensor]:
        return self._compiler.compile_graph(gm, example_inputs)

    def _analyze_generic_graph(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> _GenericGraph:
        return self._graph_builder.build(gm, example_inputs)

    def _write_generic_source(self, graph: _GenericGraph) -> str:
        return self._emitter.emit(graph)


_DEFAULT_BACKEND = CodegenBackend()


def get_generic_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> str:
    return _DEFAULT_BACKEND.get_generic_source(gm, example_inputs)


def codegen_generic_backend(
    gm: torch.fx.GraphModule, example_inputs: List[object]
) -> Callable[..., torch.Tensor]:
    return _DEFAULT_BACKEND.codegen_generic_backend(gm, example_inputs)
