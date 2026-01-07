import hashlib
import math
import numbers
import operator
from collections.abc import Sequence as ABCSequence
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.fx
from torch.fx.immutable_collections import immutable_list

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import (
    _dtype_to_c_type,
    _input_c_type,
    _normalize_scalar_value,
)
from codegen_backend.compile import compile_or_load
from codegen_backend.dtypes import (
    _CODEGEN_DTYPES,
    _CodegenDType,
    _EMBEDDING_INDEX_DTYPES,
    _INTEGER_CODEGEN_DTYPES,
)
from codegen_backend.graph import _GenericGraph, _GenericLibrary, _OpNode
from codegen_backend.emitters.base import _format_array_suffix, _is_contiguous
from codegen_backend.indexing import (
    _contiguous_strides,
    _emit_strided_access,
    _format_strided_access,
)
from codegen_backend.groups.registry import get_group_registry
from codegen_backend.kinds import HandlerContext, OpNodeBuildResult
from codegen_backend.param_normalize import normalize_int_or_tuple
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.templates import get_template_env


def _is_out_overload(target: object) -> bool:
    schema = getattr(target, "_schema", None)
    return schema is not None and schema.overload_name == "out"


_C_SRC_DIR = Path(__file__).resolve().parents[2] / "csrc"



def _channels_last_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    if len(shape) != 4:
        raise CodegenBackendError("required rank 4 tensor to use channels_last format")
    batch, channels, height, width = (
        max(shape[0], 1),
        max(shape[1], 1),
        max(shape[2], 1),
        max(shape[3], 1),
    )
    return (
        height * width * channels,
        1,
        width * channels,
        channels,
    )


def _channels_last_3d_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    if len(shape) != 5:
        raise CodegenBackendError(
            "required rank 5 tensor to use channels_last_3d format"
        )
    batch, channels, depth, height, width = (
        max(shape[0], 1),
        max(shape[1], 1),
        max(shape[2], 1),
        max(shape[3], 1),
        max(shape[4], 1),
    )
    return (
        depth * height * width * channels,
        1,
        height * width * channels,
        width * channels,
        channels,
    )


def _normalize_col2im_output_size(
    op_name: str, value: object
) -> Tuple[int, int]:
    if isinstance(value, torch.Size):
        value = tuple(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CodegenBackendError(
            f"codegen {op_name} expects output_size to be a tuple of two ints"
        )
    try:
        return normalize_int_or_tuple("output_size", value, 2)
    except ValueError as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects output_size to be a tuple of ints"
        ) from exc


def _resolve_scalar_arg(
    op_name: str,
    value: object,
    scalar_values: Dict[torch.fx.Node, object],
) -> float | int | bool:
    if isinstance(value, torch.fx.Node):
        if value in scalar_values:
            return _normalize_scalar_value(op_name, scalar_values[value])
        meta_value = value.meta.get("val")
        if meta_value is None:
            meta_value = value.meta.get("example_value")
        if meta_value is not None:
            return _normalize_scalar_value(op_name, meta_value)
        raise CodegenBackendError(f"codegen {op_name} expects a scalar value")
    return _normalize_scalar_value(op_name, value)


def _normalize_as_strided_sequence(
    op_name: str, value: object, arg_name: str
) -> Tuple[int, ...]:
    if isinstance(value, torch.Size):
        seq = tuple(value)
    elif isinstance(value, ABCSequence) and not isinstance(value, (str, bytes)):
        seq = value
    else:
        raise CodegenBackendError(
            f"codegen {op_name} expects {arg_name} to be a sequence"
        )
    try:
        return tuple(int(operator.index(item)) for item in seq)
    except TypeError as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects {arg_name} entries to be int-like"
        ) from exc


def _resolve_alias(
    node: torch.fx.Node, alias_map: Dict[torch.fx.Node, torch.fx.Node]
) -> torch.fx.Node:
    while node in alias_map:
        node = alias_map[node]
    return node


def _kernel_inputs(op_node: _OpNode) -> List[torch.fx.Node]:
    if _is_out_overload(op_node.node.target) and op_node.inplace_input is not None:
        return [
            arg
            for index, arg in enumerate(op_node.inputs)
            if index != op_node.inplace_input
        ]
    return list(op_node.inputs)


def _write_generic_source(
    graph: _GenericGraph,
    *,
    kind_handlers: Dict[OpKind, "OpKindHandler"],
    templates_env: Any,
) -> str:
    placeholders = graph.tensor_placeholders
    op_nodes = graph.op_nodes
    headers = [
        "#include <stdint.h>",
        "#include <stdbool.h>",
        f"#include \"{graph.dtype.scalar_header}\"",
    ]
    kernels: List[str] = []
    for index, op_node in enumerate(op_nodes, start=1):
        handler = kind_handlers.get(op_node.spec.kind)
        if handler is None:
            raise CodegenBackendError(
                "codegen backend does not support kind "
                f"'{op_node.spec.kind.value}'"
            )
        handler.postprocess(op_node, graph)
        kernel_lines = handler.emit(index, op_node, graph)
        kernels.append("\n".join(kernel_lines))
    input_args = ", ".join(
        [
            (
                f"const {_input_c_type(graph.dtypes[node], graph.dtype)} "
                f"input_{idx}{_format_array_suffix(graph.shapes[node])}"
            )
            for idx, node in enumerate(placeholders)
        ]
    )
    input_args = f"{input_args}, " if input_args else ""
    output_dtype = graph.dtypes[graph.output_value]
    output_c_type = _dtype_to_c_type(output_dtype, graph.dtype)
    signature = (
        f"void ref_codegen_main_{graph.dtype.suffix}("
        f"{input_args}{output_c_type} out{_format_array_suffix(graph.shapes[graph.output_value])}) {{"
    )
    name_map: Dict[torch.fx.Node, str] = {}
    for idx, placeholder in enumerate(placeholders):
        name_map[placeholder] = f"input_{idx}"
    temp_index = 0
    temp_decls: List[str] = []
    for op_node in op_nodes:
        if op_node.node is graph.output_value:
            if op_node.inplace_input is not None:
                name_map[op_node.node] = name_map[op_node.inputs[op_node.inplace_input]]
            else:
                name_map[op_node.node] = "out"
            continue
        if op_node.inplace_input is not None:
            name_map[op_node.node] = name_map[op_node.inputs[op_node.inplace_input]]
            continue
        temp_name = f"tmp_{temp_index}"
        temp_index += 1
        name_map[op_node.node] = temp_name
        temp_dtype = graph.dtypes[op_node.node]
        temp_c_type = _dtype_to_c_type(temp_dtype, graph.dtype)
        temp_decls.append(
            f"{temp_c_type} {temp_name}{_format_array_suffix(op_node.output_shape)};"
        )
    call_lines: List[str] = []
    for index, op_node in enumerate(op_nodes, start=1):
        input_names = [
            name_map[_resolve_alias(arg, graph.alias_map)]
            for arg in _kernel_inputs(op_node)
        ]
        output_name = name_map[op_node.node]
        args = ", ".join([*input_names, output_name])
        call_lines.append(
            f"node{index}_{op_node.spec.name}_{graph.dtype.suffix}({args});"
        )
    template = templates_env.get_template("generic_source.c.j2")
    return (
        template.render(
            headers=headers,
            kernels=kernels,
            signature=signature,
            temp_decls=temp_decls,
            call_lines=call_lines,
        )
        + "\n"
    )


def _iter_example_tensors(example_inputs: Sequence[object]) -> Iterable[torch.Tensor]:
    for example in example_inputs:
        if isinstance(example, torch.Tensor):
            yield example
            continue
        if isinstance(example, (list, tuple)):
            for item in example:
                if isinstance(item, torch.Tensor):
                    yield item
                elif isinstance(item, (list, tuple)):
                    yield from _iter_example_tensors(item)


def _validate_example_inputs(
    example_inputs: Sequence[object],
) -> _CodegenDType | None:
    all_tensor_examples = list(_iter_example_tensors(example_inputs))
    tensor_examples = [
        example
        for example in all_tensor_examples
        if example.dtype in _CODEGEN_DTYPES
    ]
    if not tensor_examples:
        if all_tensor_examples:
            raise CodegenBackendError(
                "codegen backend supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
            )
        return None
    for example in _iter_example_tensors(example_inputs):
        if example.device.type != "cpu":
            raise CodegenBackendError("codegen backend supports only CPU tensors")
    non_bool_examples = [
        example for example in tensor_examples if example.dtype is not torch.bool
    ]
    if non_bool_examples:
        non_bool_dtypes = {example.dtype for example in non_bool_examples}
        non_index_dtypes = {
            dtype
            for dtype in non_bool_dtypes
            if dtype not in _EMBEDDING_INDEX_DTYPES
        }
        if len(non_index_dtypes) > 1:
            raise CodegenBackendError(
                "codegen backend expects all tensors to share a dtype"
            )
        if non_index_dtypes:
            first_dtype = next(iter(non_index_dtypes))
        else:
            first_dtype = next(iter(non_bool_dtypes))
    else:
        first_dtype = torch.bool
    dtype_info = _CODEGEN_DTYPES.get(first_dtype)
    if dtype_info is None:
        raise CodegenBackendError(
            "codegen backend supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
        )
    for example in tensor_examples:
        if example.dtype is torch.bool:
            continue
        if (
            example.dtype is not first_dtype
            and example.dtype not in _EMBEDDING_INDEX_DTYPES
        ):
            raise CodegenBackendError(
                "codegen backend expects all tensors to share a dtype"
            )
    return dtype_info


def _infer_empty_strided_dtype(
    gm: torch.fx.GraphModule,
    *,
    target_registry: Dict[object, "_TargetInfo"],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _CodegenDType | None:
    dtype_value = None
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target_info = target_registry.get(node.target)
        if target_info is None:
            continue
        handler = kind_handlers.get(target_info.op_spec.kind)
        if handler is None:
            continue
        node_dtype = handler.infer_graph_dtype(node, target_info.op_spec)
        if node_dtype is None:
            continue
        if dtype_value is None:
            dtype_value = node_dtype
        elif dtype_value is not node_dtype:
            raise CodegenBackendError(
                "codegen empty_strided requires a consistent dtype"
            )
    if dtype_value is None:
        return None
    dtype_info = _CODEGEN_DTYPES.get(dtype_value)
    if dtype_info is None:
        raise CodegenBackendError(
            "codegen empty_strided supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
        )
    return dtype_info


def _unwrap_output_node(output_node: torch.fx.Node) -> Tuple[torch.fx.Node, object]:
    output_value = output_node.args[0]
    output_structure = output_value
    if isinstance(output_value, (tuple, list, immutable_list)):
        if not output_value:
            raise CodegenBackendError("codegen backend expects a non-empty output list")
        if not all(isinstance(item, torch.fx.Node) for item in output_value):
            raise CodegenBackendError("codegen backend expects output nodes only")
        output_value = output_value[0]
    if not isinstance(output_value, torch.fx.Node):
        raise CodegenBackendError("codegen backend expects a single output node")
    return output_value, output_structure


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
def _normalize_reduction_dims(
    op_name: str, dim: object | None, rank: int
) -> Tuple[int, ...]:
    if dim is None:
        return tuple(range(rank))
    if isinstance(dim, torch.fx.Node):
        raise CodegenBackendError(
            f"codegen {op_name} expects dim to be an int or tuple of ints"
        )
    if rank == 0:
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        for item in dims:
            if isinstance(item, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_name} expects dim to be an int or tuple of ints"
                )
            try:
                dim_value = operator.index(item)
            except TypeError as exc:
                raise CodegenBackendError(
                    f"codegen {op_name} expects dim to be an int or tuple of ints"
                ) from exc
            if dim_value not in (-1, 0):
                raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        return ()
    if isinstance(dim, (tuple, list)):
        dims = dim
    else:
        dims = (dim,)
    normalized: List[int] = []
    seen: set[int] = set()
    for item in dims:
        if isinstance(item, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects dim to be an int or tuple of ints"
            )
        try:
            dim_value = operator.index(item)
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects dim to be an int or tuple of ints"
            ) from exc
        if dim_value < 0:
            dim_value += rank
        if dim_value < 0 or dim_value >= rank:
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        if dim_value in seen:
            continue
        seen.add(dim_value)
        normalized.append(dim_value)
    return tuple(sorted(normalized))


def _error_expected_tensor(op_name: str) -> CodegenBackendError:
    return CodegenBackendError(f"codegen {op_name} expects tensor inputs only")


def _error_kwarg_specified_once(op_name: str, kwarg: str) -> CodegenBackendError:
    return CodegenBackendError(f"codegen {op_name} expects {kwarg} to be specified once")


def _normalize_param(normalizer: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return normalizer(*args, **kwargs)
    except ValueError as exc:
        raise CodegenBackendError(str(exc)) from exc


def _parse_reduction_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[Tuple[int, ...], bool, bool, bool | None]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 4:
        raise CodegenBackendError(f"codegen {op_name} expects at most four inputs")
    if op_name == "std":
        unbiased = True
        if len(node.args) > 2:
            raise CodegenBackendError(
                "codegen std expects at most two inputs (self, unbiased)"
            )
        if len(node.args) > 1:
            unbiased = node.args[1]
        if node.kwargs:
            if "unbiased" in node.kwargs:
                if len(node.args) > 1:
                    raise _error_kwarg_specified_once(op_name, "unbiased")
                unbiased = node.kwargs["unbiased"]
            extra = set(node.kwargs) - {"unbiased"}
            if extra:
                raise CodegenBackendError(
                    f"codegen std got unexpected kwargs: {sorted(extra)}"
                )
        if isinstance(unbiased, torch.fx.Node):
            raise CodegenBackendError("codegen std expects unbiased to be a bool")
        if not isinstance(unbiased, bool):
            raise CodegenBackendError("codegen std expects unbiased to be a bool")
        reduction_dims = tuple(range(len(input_shape)))
        keepdim = False
        reduce_all = True
        return reduction_dims, keepdim, reduce_all, unbiased
    if op_name == "var":
        if len(node.args) > 4:
            raise CodegenBackendError(
                "codegen var expects at most four inputs (self, dim, unbiased, keepdim)"
            )
        dim = node.args[1] if len(node.args) > 1 else None
        unbiased = node.args[2] if len(node.args) > 2 else True
        keepdim = node.args[3] if len(node.args) > 3 else False
        correction = None
        if node.kwargs:
            if "dim" in node.kwargs:
                if dim is not None:
                    raise _error_kwarg_specified_once(op_name, "dim")
                dim = node.kwargs["dim"]
            if "unbiased" in node.kwargs:
                if len(node.args) > 2:
                    raise _error_kwarg_specified_once(op_name, "unbiased")
                unbiased = node.kwargs["unbiased"]
            if "keepdim" in node.kwargs:
                if len(node.args) > 3:
                    raise _error_kwarg_specified_once(op_name, "keepdim")
                keepdim = node.kwargs["keepdim"]
            if "correction" in node.kwargs:
                correction = node.kwargs["correction"]
            extra = set(node.kwargs) - {"dim", "unbiased", "keepdim", "correction"}
            if extra:
                raise CodegenBackendError(
                    f"codegen var got unexpected kwargs: {sorted(extra)}"
                )
        if isinstance(unbiased, torch.fx.Node):
            raise CodegenBackendError("codegen var expects unbiased to be a bool")
        if not isinstance(unbiased, bool):
            raise CodegenBackendError("codegen var expects unbiased to be a bool")
        if isinstance(keepdim, torch.fx.Node):
            raise CodegenBackendError("codegen var expects keepdim to be a bool")
        if not isinstance(keepdim, bool):
            raise CodegenBackendError("codegen var expects keepdim to be a bool")
        if correction is not None:
            if isinstance(correction, torch.fx.Node):
                raise CodegenBackendError("codegen var expects correction to be 0 or 1")
            if not isinstance(correction, numbers.Number):
                raise CodegenBackendError("codegen var expects correction to be 0 or 1")
            correction_value = float(correction)
            if correction_value not in (0.0, 1.0):
                raise CodegenBackendError("codegen var expects correction to be 0 or 1")
            unbiased = bool(int(correction_value))
        reduction_dims = _normalize_reduction_dims(op_name, dim, len(input_shape))
        reduce_all = dim is None
        return reduction_dims, keepdim, reduce_all, unbiased
    dim = node.args[1] if len(node.args) > 1 else None
    keepdim = node.args[2] if len(node.args) > 2 else False
    dtype = node.args[3] if len(node.args) > 3 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once(op_name, "dim")
            dim = node.kwargs["dim"]
        if "keepdim" in node.kwargs:
            if len(node.args) > 2:
                raise _error_kwarg_specified_once(op_name, "keepdim")
            keepdim = node.kwargs["keepdim"]
        if "dtype" in node.kwargs:
            if dtype is not None:
                raise _error_kwarg_specified_once(op_name, "dtype")
            dtype = node.kwargs["dtype"]
        extra = set(node.kwargs) - {"dim", "keepdim", "dtype"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(keepdim, torch.fx.Node):
        raise CodegenBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if not isinstance(keepdim, bool):
        raise CodegenBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if dtype is not None:
        if isinstance(dtype, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.int8, torch.int32, or torch.bool"
            )
        if dtype not in (torch.float32, torch.int8, torch.int32, torch.bool):
            raise CodegenBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.int8, torch.int32, or torch.bool"
            )
    reduction_dims = _normalize_reduction_dims(op_name, dim, len(input_shape))
    reduce_all = dim is None
    return reduction_dims, keepdim, reduce_all, None


def _parse_norm_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[Tuple[int, ...], bool, bool, float]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 4:
        raise CodegenBackendError(
            f"codegen {op_name} expects at most four inputs (self, p, dim, keepdim)"
        )
    p = node.args[1] if len(node.args) > 1 else None
    dim = node.args[2] if len(node.args) > 2 else None
    keepdim = node.args[3] if len(node.args) > 3 else False
    if node.kwargs:
        if "p" in node.kwargs:
            if len(node.args) > 1:
                raise _error_kwarg_specified_once(op_name, "p")
            p = node.kwargs["p"]
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once(op_name, "dim")
            dim = node.kwargs["dim"]
        if "keepdim" in node.kwargs:
            if len(node.args) > 3:
                raise _error_kwarg_specified_once(op_name, "keepdim")
            keepdim = node.kwargs["keepdim"]
        extra = set(node.kwargs) - {"p", "dim", "keepdim"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(keepdim, torch.fx.Node):
        raise CodegenBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if not isinstance(keepdim, bool):
        raise CodegenBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if isinstance(p, torch.fx.Node):
        raise CodegenBackendError(f"codegen {op_name} expects p to be a number")
    if p is None:
        p_value = 2.0
    elif isinstance(p, bool):
        p_value = float(p)
    elif isinstance(p, (int, float)):
        p_value = float(p)
    else:
        raise CodegenBackendError(f"codegen {op_name} expects p to be a number")
    if math.isinf(p_value) or math.isnan(p_value):
        raise CodegenBackendError(f"codegen {op_name} expects p to be finite")
    reduction_dims = _normalize_reduction_dims(op_name, dim, len(input_shape))
    reduce_all = dim is None
    return reduction_dims, keepdim, reduce_all, p_value


def _parse_argminmax_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[Tuple[int, ...], bool, bool]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 3:
        raise CodegenBackendError(f"codegen {op_name} expects at most three inputs")
    dim = node.args[1] if len(node.args) > 1 else None
    keepdim = node.args[2] if len(node.args) > 2 else False
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once(op_name, "dim")
            dim = node.kwargs["dim"]
        if "keepdim" in node.kwargs:
            if len(node.args) > 2:
                raise _error_kwarg_specified_once(op_name, "keepdim")
            keepdim = node.kwargs["keepdim"]
        extra = set(node.kwargs) - {"dim", "keepdim"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(keepdim, torch.fx.Node):
        raise CodegenBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if not isinstance(keepdim, bool):
        raise CodegenBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if dim is None:
        reduction_dims = tuple(range(len(input_shape)))
        reduce_all = True
        return reduction_dims, keepdim, reduce_all
    if isinstance(dim, (tuple, list)):
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    dim_value = _parse_constant_int(op_name, "dim", dim)
    rank = len(input_shape)
    if rank == 0:
        if dim_value not in (-1, 0):
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        return (), keepdim, True
    if dim_value < 0:
        dim_value += rank
    if dim_value < 0 or dim_value >= rank:
        raise CodegenBackendError(f"codegen {op_name} dim is out of range")
    return (dim_value,), keepdim, False


def _constant_error_message(
    op_name: str, name: str, expected_type: type, *, node_error: bool = False
) -> str:
    if expected_type is int:
        return f"codegen {op_name} expects {name} to be an int"
    if expected_type is bool:
        return f"codegen {op_name} expects {name} to be a bool"
    if expected_type is float:
        if node_error:
            return f"codegen {op_name} expects {name} to be constant"
        return f"codegen {op_name} expects {name} to be numeric"
    raise ValueError(f"Unsupported expected type: {expected_type}")


def resolve_node_constant(
    value: object,
    expected_type: type,
    *,
    fallback_meta_keys: Tuple[str, ...] = ("val", "example_value"),
    allow_scalar_tensor: bool = False,
    op_name: str,
    name: str,
) -> object:
    type_error_message = _constant_error_message(op_name, name, expected_type)
    node_error_message = _constant_error_message(
        op_name, name, expected_type, node_error=True
    )
    if isinstance(value, torch.fx.Node):
        for key in fallback_meta_keys:
            if key in value.meta:
                value = value.meta[key]
                break
        else:
            keys_list = ", ".join(fallback_meta_keys)
            raise CodegenBackendError(
                f"{node_error_message}; missing node.meta for keys: {keys_list}"
            )
    if allow_scalar_tensor and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(type_error_message)
        value = value.item()
    if expected_type is int:
        if isinstance(value, bool):
            raise CodegenBackendError(type_error_message)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError as exc:
                raise CodegenBackendError(type_error_message) from exc
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise CodegenBackendError(type_error_message)
        try:
            return operator.index(value)
        except TypeError as exc:
            raise CodegenBackendError(type_error_message) from exc
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        raise CodegenBackendError(type_error_message)
    if expected_type is float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        raise CodegenBackendError(type_error_message)
    raise ValueError(f"Unsupported expected type: {expected_type}")


def _parse_constant_float(op_name: str, name: str, value: object) -> float:
    return resolve_node_constant(value, float, op_name=op_name, name=name)


def _parse_constant_int(op_name: str, name: str, value: object) -> int:
    return resolve_node_constant(
        value,
        int,
        allow_scalar_tensor=True,
        op_name=op_name,
        name=name,
    )


def _parse_constant_bool(op_name: str, name: str, value: object) -> bool:
    return resolve_node_constant(
        value,
        bool,
        allow_scalar_tensor=True,
        op_name=op_name,
        name=name,
    )


def _parse_bitwise_scalar(
    op_name: str, value: object, dtype: torch.dtype
) -> object:
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise CodegenBackendError(
                f"codegen {op_name} expects scalar to be constant"
            )
        return _parse_bitwise_scalar(op_name, meta_value, dtype)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(
                f"codegen {op_name} expects scalar to be a single value"
            )
        return _parse_bitwise_scalar(op_name, value.item(), dtype)
    if dtype is torch.bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise CodegenBackendError(
                    f"codegen {op_name} expects scalar to be a boolean value"
                )
            return bool(int(value))
        try:
            return bool(operator.index(value))
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects scalar to be a boolean value"
            ) from exc
    if isinstance(value, bool):
        return int(value)
    return _parse_constant_int(op_name, "scalar", value)


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
        params["alpha"] = _parse_constant_float(
            op_name, "alpha", params.get("alpha", 1.0)
        )
        params["scale"] = _parse_constant_float(
            op_name, "scale", params.get("scale", 1.0)
        )
        params["input_scale"] = _parse_constant_float(
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
        params["negative_slope"] = _parse_constant_float(
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
        params["beta"] = _parse_constant_float(
            op_name, "beta", params.get("beta", 1.0)
        )
        params["threshold"] = _parse_constant_float(
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
            params["min_val"] = _parse_constant_float(
                op_name, "min", params["min_val"]
            )
        if params.get("max_val") is not None:
            params["max_val"] = _parse_constant_float(
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
        params["min_val"] = _parse_constant_float(
            op_name, "min_val", params.get("min_val", -1.0)
        )
        params["max_val"] = _parse_constant_float(
            op_name, "max_val", params.get("max_val", 1.0)
        )
        return input_node, params
    raise CodegenBackendError(f"Unsupported parametric op: {op_name}")


def _parse_softmax_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, object | None]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    is_internal = op_name in {"_softmax", "_log_softmax"}
    if len(node.args) > 3:
        raise CodegenBackendError(f"codegen {op_name} expects at most three inputs")
    dim = node.args[1] if len(node.args) > 1 else None
    dtype = None
    half_to_float = None
    if len(node.args) > 2:
        if is_internal:
            half_to_float = node.args[2]
        else:
            dtype = node.args[2]
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once(op_name, "dim")
            dim = node.kwargs["dim"]
        if is_internal:
            if "half_to_float" in node.kwargs:
                if half_to_float is not None:
                    raise _error_kwarg_specified_once(op_name, "half_to_float")
                half_to_float = node.kwargs["half_to_float"]
            extra = set(node.kwargs) - {"dim", "half_to_float"}
        else:
            if "dtype" in node.kwargs:
                if dtype is not None:
                    raise _error_kwarg_specified_once(op_name, "dtype")
                dtype = node.kwargs["dtype"]
            extra = set(node.kwargs) - {"dim", "dtype"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None:
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, torch.fx.Node):
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, (tuple, list)):
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    try:
        dim_value = operator.index(dim)
    except TypeError as exc:
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int") from exc
    rank = len(input_shape)
    if dim_value < 0:
        dim_value += rank
    if dim_value < 0 or dim_value >= rank:
        raise CodegenBackendError(f"codegen {op_name} dim is out of range")
    if is_internal:
        if half_to_float is None:
            half_to_float = False
        if isinstance(half_to_float, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects half_to_float to be a bool"
            )
        if half_to_float not in (False, 0):
            raise CodegenBackendError(
                f"codegen {op_name} expects half_to_float to be False"
            )
    else:
        if dtype is not None:
            if isinstance(dtype, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_name} expects dtype to be torch.float32 or None"
                )
            if dtype is not torch.float32:
                raise CodegenBackendError(
                    f"codegen {op_name} expects dtype to be torch.float32 or None"
                )
    return dim_value, dtype


def _parse_diagonal_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, int, int]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 4:
        raise CodegenBackendError(f"codegen {op_name} expects at most four inputs")
    offset = node.args[1] if len(node.args) > 1 else 0
    dim1 = node.args[2] if len(node.args) > 2 else 0
    dim2 = node.args[3] if len(node.args) > 3 else 1
    if node.kwargs:
        if "offset" in node.kwargs:
            if len(node.args) > 1:
                raise _error_kwarg_specified_once(op_name, "offset")
            offset = node.kwargs["offset"]
        if "dim1" in node.kwargs:
            if len(node.args) > 2:
                raise _error_kwarg_specified_once(op_name, "dim1")
            dim1 = node.kwargs["dim1"]
        if "dim2" in node.kwargs:
            if len(node.args) > 3:
                raise _error_kwarg_specified_once(op_name, "dim2")
            dim2 = node.kwargs["dim2"]
        extra = set(node.kwargs) - {"offset", "dim1", "dim2"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    offset_value = _parse_constant_int(op_name, "offset", offset)
    dim1_value = _parse_constant_int(op_name, "dim1", dim1)
    dim2_value = _parse_constant_int(op_name, "dim2", dim2)
    rank = len(input_shape)
    if rank < 2:
        raise CodegenBackendError(f"codegen {op_name} expects input rank >= 2")
    if dim1_value < 0:
        dim1_value += rank
    if dim2_value < 0:
        dim2_value += rank
    if dim1_value < 0 or dim1_value >= rank:
        raise CodegenBackendError(f"codegen {op_name} dim1 is out of range")
    if dim2_value < 0 or dim2_value >= rank:
        raise CodegenBackendError(f"codegen {op_name} dim2 is out of range")
    if dim1_value == dim2_value:
        raise CodegenBackendError(f"codegen {op_name} expects dim1 != dim2")
    return offset_value, dim1_value, dim2_value


def _parse_cumsum_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, torch.dtype | None]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 3:
        raise CodegenBackendError(
            f"codegen {op_name} expects at most three inputs (self, dim, dtype)"
        )
    dim = node.args[1] if len(node.args) > 1 else None
    dtype = node.args[2] if len(node.args) > 2 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once(op_name, "dim")
            dim = node.kwargs["dim"]
        if "dtype" in node.kwargs:
            if dtype is not None:
                raise _error_kwarg_specified_once(op_name, "dtype")
            dtype = node.kwargs["dtype"]
        extra = set(node.kwargs) - {"dim", "dtype"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None:
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, (tuple, list)):
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    dim_value = _parse_constant_int(op_name, "dim", dim)
    rank = len(input_shape)
    if rank == 0:
        if dim_value not in (-1, 0):
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        dim_value = 0
    else:
        if dim_value < 0:
            dim_value += rank
        if dim_value < 0 or dim_value >= rank:
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
    if dtype is not None:
        if isinstance(dtype, torch.fx.Node):
            meta_value = dtype.meta.get("val") or dtype.meta.get("example_value")
            if meta_value is None:
                raise CodegenBackendError(
                    f"codegen {op_name} expects dtype to be torch.float32, torch.int8, or torch.int32"
                )
            dtype = meta_value
        if dtype not in (torch.float32, torch.int8, torch.int32):
            raise CodegenBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.int8, or torch.int32"
            )
    return dim_value, dtype


def _parse_gather_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object, object]:
    if len(node.args) > 4:
        raise CodegenBackendError("codegen gather expects at most four inputs")
    if not node.args:
        raise CodegenBackendError("codegen gather expects input, dim, and index")
    input_arg = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else None
    index = node.args[2] if len(node.args) > 2 else None
    sparse_grad = node.args[3] if len(node.args) > 3 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once("gather", "dim")
            dim = node.kwargs["dim"]
        if "index" in node.kwargs:
            if index is not None:
                raise _error_kwarg_specified_once("gather", "index")
            index = node.kwargs["index"]
        if "sparse_grad" in node.kwargs:
            if sparse_grad is not None:
                raise _error_kwarg_specified_once("gather", "sparse_grad")
            sparse_grad = node.kwargs["sparse_grad"]
        extra = set(node.kwargs) - {"dim", "index", "sparse_grad"}
        if extra:
            raise CodegenBackendError(
                f"codegen gather got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None or index is None:
        raise CodegenBackendError("codegen gather expects dim and index arguments")
    if sparse_grad is None:
        sparse_grad = False
    return input_arg, dim, index, sparse_grad


def _parse_embedding_args(
    node: torch.fx.Node,
) -> Tuple[object, object, int, bool, bool]:
    op_name = "embedding"
    if len(node.args) < 2:
        raise CodegenBackendError(f"codegen {op_name} expects weight and indices")
    if len(node.args) > 5:
        raise CodegenBackendError(
            f"codegen {op_name} expects at most five arguments"
        )
    weight = node.args[0]
    indices = node.args[1]
    padding_idx = node.args[2] if len(node.args) > 2 else -1
    scale_grad_by_freq = node.args[3] if len(node.args) > 3 else False
    sparse = node.args[4] if len(node.args) > 4 else False
    if node.kwargs:
        if "padding_idx" in node.kwargs:
            if len(node.args) > 2:
                raise _error_kwarg_specified_once(op_name, "padding_idx")
            padding_idx = node.kwargs["padding_idx"]
        if "scale_grad_by_freq" in node.kwargs:
            if len(node.args) > 3:
                raise _error_kwarg_specified_once(op_name, "scale_grad_by_freq")
            scale_grad_by_freq = node.kwargs["scale_grad_by_freq"]
        if "sparse" in node.kwargs:
            if len(node.args) > 4:
                raise _error_kwarg_specified_once(op_name, "sparse")
            sparse = node.kwargs["sparse"]
        extra = set(node.kwargs) - {
            "padding_idx",
            "scale_grad_by_freq",
            "sparse",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    padding_idx_value = _parse_constant_int(op_name, "padding_idx", padding_idx)
    scale_grad_value = _parse_constant_bool(
        op_name, "scale_grad_by_freq", scale_grad_by_freq
    )
    sparse_value = _parse_constant_bool(op_name, "sparse", sparse)
    return weight, indices, padding_idx_value, scale_grad_value, sparse_value


def _parse_embedding_bag_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object, bool, int, bool, object, bool, int]:
    op_name = "_embedding_bag"
    if len(node.args) < 3:
        raise CodegenBackendError(
            f"codegen {op_name} expects weight, indices, and offsets"
        )
    if len(node.args) > 9:
        raise CodegenBackendError(
            f"codegen {op_name} expects at most nine arguments"
        )
    weight = node.args[0]
    indices = node.args[1]
    offsets = node.args[2]
    scale_grad_by_freq = node.args[3] if len(node.args) > 3 else False
    mode = node.args[4] if len(node.args) > 4 else 0
    sparse = node.args[5] if len(node.args) > 5 else False
    per_sample_weights = node.args[6] if len(node.args) > 6 else None
    include_last_offset = node.args[7] if len(node.args) > 7 else False
    padding_idx = node.args[8] if len(node.args) > 8 else -1
    if node.kwargs:
        if "scale_grad_by_freq" in node.kwargs:
            if len(node.args) > 3:
                raise _error_kwarg_specified_once(op_name, "scale_grad_by_freq")
            scale_grad_by_freq = node.kwargs["scale_grad_by_freq"]
        if "mode" in node.kwargs:
            if len(node.args) > 4:
                raise _error_kwarg_specified_once(op_name, "mode")
            mode = node.kwargs["mode"]
        if "sparse" in node.kwargs:
            if len(node.args) > 5:
                raise _error_kwarg_specified_once(op_name, "sparse")
            sparse = node.kwargs["sparse"]
        if "per_sample_weights" in node.kwargs:
            if len(node.args) > 6:
                raise _error_kwarg_specified_once(op_name, "per_sample_weights")
            per_sample_weights = node.kwargs["per_sample_weights"]
        if "include_last_offset" in node.kwargs:
            if len(node.args) > 7:
                raise _error_kwarg_specified_once(op_name, "include_last_offset")
            include_last_offset = node.kwargs["include_last_offset"]
        if "padding_idx" in node.kwargs:
            if len(node.args) > 8:
                raise _error_kwarg_specified_once(op_name, "padding_idx")
            padding_idx = node.kwargs["padding_idx"]
        extra = set(node.kwargs) - {
            "scale_grad_by_freq",
            "mode",
            "sparse",
            "per_sample_weights",
            "include_last_offset",
            "padding_idx",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    scale_grad_value = _parse_constant_bool(
        op_name, "scale_grad_by_freq", scale_grad_by_freq
    )
    mode_value = _parse_constant_int(op_name, "mode", mode)
    sparse_value = _parse_constant_bool(op_name, "sparse", sparse)
    include_last_offset_value = _parse_constant_bool(
        op_name, "include_last_offset", include_last_offset
    )
    padding_idx_value = _parse_constant_int(op_name, "padding_idx", padding_idx)
    return (
        weight,
        indices,
        offsets,
        scale_grad_value,
        mode_value,
        sparse_value,
        per_sample_weights,
        include_last_offset_value,
        padding_idx_value,
    )


def _parse_addmm_like_scalar(
    op_name: str, name: str, value: object | None
) -> float:
    if value is None:
        return 1.0
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise CodegenBackendError(f"codegen {op_name} expects {name} to be a number")
        return _parse_addmm_like_scalar(op_name, name, meta_value)
    if isinstance(value, bool):
        raise CodegenBackendError(f"codegen {op_name} expects {name} to be a number")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(f"codegen {op_name} expects {name} to be a number")
        return float(value.item())
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects {name} to be a number"
            ) from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects {name} to be a number"
        ) from exc


def _validate_addmm_like_scalars(
    op_name: str, dtype: torch.dtype, alpha: float, beta: float
) -> None:
    if dtype in _INTEGER_CODEGEN_DTYPES or dtype is torch.bool:
        for name, value in (("alpha", alpha), ("beta", beta)):
            if not float(value).is_integer():
                raise CodegenBackendError(
                    f"codegen {op_name} expects {name} to be an integer for integral tensors"
                )


def _parse_addmm_like_args(
    op_name: str, node: torch.fx.Node
) -> Tuple[Sequence[torch.fx.Node], float, float]:
    if len(node.args) < 3:
        raise CodegenBackendError(f"codegen {op_name} expects at least three inputs")
    if len(node.args) > 5:
        raise CodegenBackendError(f"codegen {op_name} expects at most five inputs")
    input_node, mat1_node, mat2_node = node.args[:3]
    beta = node.args[3] if len(node.args) > 3 else None
    alpha = node.args[4] if len(node.args) > 4 else None
    if node.kwargs:
        if "beta" in node.kwargs:
            if beta is not None:
                raise _error_kwarg_specified_once(op_name, "beta")
            beta = node.kwargs["beta"]
        if "alpha" in node.kwargs:
            if alpha is not None:
                raise _error_kwarg_specified_once(op_name, "alpha")
            alpha = node.kwargs["alpha"]
        extra = set(node.kwargs) - {"beta", "alpha"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    return (
        (input_node, mat1_node, mat2_node),
        _parse_addmm_like_scalar(op_name, "alpha", alpha),
        _parse_addmm_like_scalar(op_name, "beta", beta),
    )


def _parse_linear_args(
    op_name: str, node: torch.fx.Node
) -> Tuple[object, object, object | None]:
    if len(node.args) < 2:
        raise CodegenBackendError(f"codegen {op_name} expects at least two inputs")
    if len(node.args) > 3:
        raise CodegenBackendError(f"codegen {op_name} expects at most three inputs")
    input_node, weight_node = node.args[:2]
    bias = node.args[2] if len(node.args) > 2 else None
    if node.kwargs:
        if "bias" in node.kwargs:
            if bias is not None:
                raise _error_kwarg_specified_once(op_name, "bias")
            bias = node.kwargs["bias"]
        extra = set(node.kwargs) - {"bias"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    return input_node, weight_node, bias


def _parse_concat_args(
    node: torch.fx.Node,
) -> Tuple[Sequence[torch.fx.Node], int]:
    if not node.args:
        raise CodegenBackendError("codegen cat expects a tensor list input")
    if len(node.args) > 2:
        raise CodegenBackendError("codegen cat expects at most two inputs")
    tensors_arg = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once("cat", "dim")
            dim = node.kwargs["dim"]
        extra = set(node.kwargs) - {"dim"}
        if extra:
            raise CodegenBackendError(
                f"codegen cat got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(dim, torch.fx.Node):
        raise CodegenBackendError("codegen cat expects dim to be an int")
    if dim is None:
        dim_value = 0
    else:
        try:
            dim_value = operator.index(dim)
        except TypeError as exc:
            raise CodegenBackendError("codegen cat expects dim to be an int") from exc
    if not isinstance(tensors_arg, (list, tuple)) or not tensors_arg:
        raise CodegenBackendError("codegen cat expects a non-empty tensor list input")
    for item in tensors_arg:
        if not isinstance(item, torch.fx.Node):
            raise _error_expected_tensor("cat")
    return list(tensors_arg), dim_value


def _parse_conv2d_args(
    node: torch.fx.Node,
) -> Tuple[
    torch.fx.Node,
    torch.fx.Node,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 9:
        raise CodegenBackendError("codegen conv2d expects convolution arguments")
    input_arg = args[0]
    weight_arg = args[1]
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    transposed = False
    output_padding: object = (0, 0)
    remaining = args[2:]
    if len(args) <= 7:
        if len(remaining) >= 1:
            bias = remaining[0]
        if len(remaining) >= 2:
            stride = remaining[1]
        if len(remaining) >= 3:
            padding = remaining[2]
        if len(remaining) >= 4:
            dilation = remaining[3]
        if len(remaining) >= 5:
            groups = remaining[4]
    elif len(args) in {8, 9}:
        if len(args) == 8:
            (
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            ) = remaining
        else:
            (
                bias,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            ) = remaining

    if kwargs:
        extra = set(kwargs) - {
            "bias",
            "stride",
            "padding",
            "dilation",
            "groups",
            "transposed",
            "output_padding",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen conv2d got unexpected kwargs: {sorted(extra)}"
            )
        if "bias" in kwargs:
            bias = kwargs["bias"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "groups" in kwargs:
            groups = kwargs["groups"]
        if "transposed" in kwargs:
            transposed = kwargs["transposed"]
        if "output_padding" in kwargs:
            output_padding = kwargs["output_padding"]

    return (
        input_arg,
        weight_arg,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


def _parse_conv1d_args(
    node: torch.fx.Node,
) -> Tuple[
    torch.fx.Node,
    torch.fx.Node,
    object,
    object,
    object,
    object,
    object,
]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 7:
        raise CodegenBackendError("codegen conv1d expects convolution arguments")
    input_arg = args[0]
    weight_arg = args[1]
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    remaining = args[2:]
    if len(remaining) >= 1:
        bias = remaining[0]
    if len(remaining) >= 2:
        stride = remaining[1]
    if len(remaining) >= 3:
        padding = remaining[2]
    if len(remaining) >= 4:
        dilation = remaining[3]
    if len(remaining) >= 5:
        groups = remaining[4]

    if kwargs:
        extra = set(kwargs) - {"bias", "stride", "padding", "dilation", "groups"}
        if extra:
            raise CodegenBackendError(
                f"codegen conv1d got unexpected kwargs: {sorted(extra)}"
            )
        if "bias" in kwargs:
            bias = kwargs["bias"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "groups" in kwargs:
            groups = kwargs["groups"]

    return (input_arg, weight_arg, bias, stride, padding, dilation, groups)


def _parse_col2im_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 1 or len(args) > 6:
        raise CodegenBackendError(
            "codegen col2im expects input, output_size, kernel_size, dilation, padding, and stride"
        )
    input_arg = args[0] if len(args) >= 1 else None
    output_size = args[1] if len(args) >= 2 else None
    kernel_size = args[2] if len(args) >= 3 else None
    dilation = args[3] if len(args) >= 4 else None
    padding = args[4] if len(args) >= 5 else None
    stride = args[5] if len(args) >= 6 else None
    if kwargs:
        extra = set(kwargs) - {
            "output_size",
            "kernel_size",
            "dilation",
            "padding",
            "stride",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen col2im got unexpected kwargs: {sorted(extra)}"
            )
        if "output_size" in kwargs:
            output_size = kwargs["output_size"]
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
    if (
        input_arg is None
        or output_size is None
        or kernel_size is None
        or dilation is None
        or padding is None
        or stride is None
    ):
        raise CodegenBackendError(
            "codegen col2im expects input, output_size, kernel_size, dilation, padding, and stride"
        )
    return input_arg, output_size, kernel_size, dilation, padding, stride


def _parse_max_pool1d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 6:
        raise CodegenBackendError("codegen max_pool1d expects pooling arguments")
    input_arg = args[0]
    kernel_size = args[1]
    stride = None
    padding = 0
    dilation = 1
    ceil_mode = False
    remaining = args[2:]
    if len(remaining) >= 1:
        stride = remaining[0]
    if len(remaining) >= 2:
        padding = remaining[1]
    if len(remaining) >= 3:
        dilation = remaining[2]
    if len(remaining) >= 4:
        ceil_mode = remaining[3]
    if kwargs:
        extra = set(kwargs) - {
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "ceil_mode",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen max_pool1d got unexpected kwargs: {sorted(extra)}"
            )
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "ceil_mode" in kwargs:
            ceil_mode = kwargs["ceil_mode"]
    return input_arg, kernel_size, stride, padding, dilation, ceil_mode


def _parse_avg_pool1d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 7:
        raise CodegenBackendError("codegen avg_pool1d expects pooling arguments")
    input_arg = args[0]
    kernel_size = args[1]
    stride = None
    padding = 0
    ceil_mode = False
    count_include_pad = False
    divisor_override = None
    remaining = args[2:]
    if len(remaining) >= 1:
        stride = remaining[0]
    if len(remaining) >= 2:
        padding = remaining[1]
    if len(remaining) >= 3:
        ceil_mode = remaining[2]
    if len(remaining) >= 4:
        count_include_pad = remaining[3]
    if len(remaining) >= 5:
        divisor_override = remaining[4]
    if kwargs:
        extra = set(kwargs) - {
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen avg_pool1d got unexpected kwargs: {sorted(extra)}"
            )
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "ceil_mode" in kwargs:
            ceil_mode = kwargs["ceil_mode"]
        if "count_include_pad" in kwargs:
            count_include_pad = kwargs["count_include_pad"]
        if "divisor_override" in kwargs:
            divisor_override = kwargs["divisor_override"]
    return (
        input_arg,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def _parse_adaptive_avg_pool1d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool1d expects input and output_size"
        )
    input_arg = args[0]
    output_size = args[1]
    if kwargs:
        if "output_size" in kwargs:
            if len(args) > 1:
                raise _error_kwarg_specified_once(
                    "adaptive_avg_pool1d", "output_size"
                )
            output_size = kwargs["output_size"]
        extra = set(kwargs) - {"output_size"}
        if extra:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool1d got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    return input_arg, output_size


def _parse_adaptive_avg_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool2d expects input and output_size"
        )
    input_arg = args[0]
    output_size = args[1]
    if kwargs:
        if "output_size" in kwargs:
            if len(args) > 1:
                raise _error_kwarg_specified_once(
                    "adaptive_avg_pool2d", "output_size"
                )
            output_size = kwargs["output_size"]
        extra = set(kwargs) - {"output_size"}
        if extra:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    return input_arg, output_size


def _parse_adaptive_avg_pool3d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool3d expects input and output_size"
        )
    input_arg = args[0]
    output_size = args[1]
    if kwargs:
        if "output_size" in kwargs:
            if len(args) > 1:
                raise _error_kwarg_specified_once(
                    "adaptive_avg_pool3d", "output_size"
                )
            output_size = kwargs["output_size"]
        extra = set(kwargs) - {"output_size"}
        if extra:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    return input_arg, output_size


def _parse_adaptive_avg_pool2d_backward_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) != 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool2d_backward expects grad_output and input"
        )
    if kwargs:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool2d_backward expects no keyword arguments"
        )
    grad_output = args[0]
    input_arg = args[1]
    return grad_output, input_arg


def _parse_max_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) > 6:
        raise CodegenBackendError("codegen max_pool2d expects pooling arguments")
    input_arg = args[0] if len(args) > 0 else None
    kernel_size = None
    stride = None
    padding = 0
    dilation = 1
    ceil_mode = False
    remaining = args[1:]
    has_kernel_size = len(remaining) >= 1
    has_stride = len(remaining) >= 2
    has_padding = len(remaining) >= 3
    has_dilation = len(remaining) >= 4
    has_ceil_mode = len(remaining) >= 5
    if has_kernel_size:
        kernel_size = remaining[0]
    if has_stride:
        stride = remaining[1]
    if has_padding:
        padding = remaining[2]
    if has_dilation:
        dilation = remaining[3]
    if has_ceil_mode:
        ceil_mode = remaining[4]
    if kwargs:
        extra = set(kwargs) - {
            "input",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "ceil_mode",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen max_pool2d got unexpected kwargs: {sorted(extra)}"
            )
        if "input" in kwargs:
            if input_arg is not None:
                raise _error_kwarg_specified_once("max_pool2d", "input")
            input_arg = kwargs["input"]
        if "kernel_size" in kwargs:
            if has_kernel_size:
                raise _error_kwarg_specified_once("max_pool2d", "kernel_size")
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            if has_stride:
                raise _error_kwarg_specified_once("max_pool2d", "stride")
            stride = kwargs["stride"]
        if "padding" in kwargs:
            if has_padding:
                raise _error_kwarg_specified_once("max_pool2d", "padding")
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            if has_dilation:
                raise _error_kwarg_specified_once("max_pool2d", "dilation")
            dilation = kwargs["dilation"]
        if "ceil_mode" in kwargs:
            if has_ceil_mode:
                raise _error_kwarg_specified_once("max_pool2d", "ceil_mode")
            ceil_mode = kwargs["ceil_mode"]
    if input_arg is None or kernel_size is None:
        raise CodegenBackendError("codegen max_pool2d expects pooling arguments")
    return input_arg, kernel_size, stride, padding, dilation, ceil_mode


def _parse_avg_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 7:
        raise CodegenBackendError("codegen avg_pool2d expects pooling arguments")
    input_arg = args[0]
    kernel_size = args[1]
    stride = None
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    remaining = args[2:]
    if len(remaining) >= 1:
        stride = remaining[0]
    if len(remaining) >= 2:
        padding = remaining[1]
    if len(remaining) >= 3:
        ceil_mode = remaining[2]
    if len(remaining) >= 4:
        count_include_pad = remaining[3]
    if len(remaining) >= 5:
        divisor_override = remaining[4]
    if kwargs:
        extra = set(kwargs) - {
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen avg_pool2d got unexpected kwargs: {sorted(extra)}"
            )
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "ceil_mode" in kwargs:
            ceil_mode = kwargs["ceil_mode"]
        if "count_include_pad" in kwargs:
            count_include_pad = kwargs["count_include_pad"]
        if "divisor_override" in kwargs:
            divisor_override = kwargs["divisor_override"]
    return (
        input_arg,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
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
        _ = float(_resolve_scalar_arg(op_spec.name, momentum, scalar_values))
        eps_value = float(_resolve_scalar_arg(op_spec.name, eps, scalar_values))
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
    input_nodes, alpha, beta = _parse_addmm_like_args(op_name, node)
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
    input_arg, weight_arg, bias_arg = _parse_linear_args(op_spec.name, node)
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
    offset, dim1, dim2 = _parse_diagonal_args(
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
    dim, dtype_override = _parse_cumsum_args(
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
    input_arg, dim, index, sparse_grad = _parse_gather_args(node)
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
    dim_value = _parse_constant_int(op_spec.name, "dim", dim)
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


def _parse_arange_dtype(
    op_name: str,
    dtype: torch.dtype | None,
    dtype_info: _CodegenDType | None,
    start: float | int | bool,
    end: float | int | bool,
    step: float | int | bool,
) -> _CodegenDType:
    if dtype is None:
        if any(
            not isinstance(value, numbers.Integral)
            for value in (start, end, step)
        ):
            dtype = torch.get_default_dtype()
        else:
            dtype = torch.int32
    if dtype is torch.bool:
        raise CodegenBackendError(
            f"codegen {op_name} supports only numeric dtypes"
        )
    dtype_spec = _CODEGEN_DTYPES.get(dtype)
    if dtype_spec is None:
        supported = ", ".join(
            f"torch.{supported_dtype.name}"
            for supported_dtype in _CODEGEN_DTYPES
            if supported_dtype is not torch.bool
        )
        raise CodegenBackendError(
            f"codegen {op_name} supports only {supported}"
        )
    if (
        dtype_info is not None
        and dtype_spec.torch_dtype is not dtype_info.torch_dtype
    ):
        raise CodegenBackendError(
            f"codegen {op_name} expects dtype to match the graph dtype"
        )
    return dtype_spec


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
        size_tuple = _normalize_as_strided_sequence(op_spec.name, size, "size")
        stride_tuple = _normalize_as_strided_sequence(
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
            dim_value = _parse_constant_int(op_spec.name, "shape", dim)
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
            dim_value = _parse_constant_int(op_spec.name, "dim", dim)
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
                dim_value = _parse_constant_int(op_spec.name, "dim", dim)
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


def _parse_resize_size(op_name: str, size_value: object) -> Tuple[int, ...]:
    if isinstance(size_value, torch.Size):
        size_value = tuple(size_value)
    if isinstance(size_value, (list, tuple)):
        try:
            return tuple(int(operator.index(item)) for item in size_value)
        except TypeError:
            try:
                return tuple(int(item) for item in size_value)
            except TypeError as exc:
                raise CodegenBackendError(
                    f"codegen {op_name} expects size values to be integers"
                ) from exc
    raise CodegenBackendError(f"codegen {op_name} expects size to be a sequence")


def _parse_empty_strided_stride(
    op_name: str, stride_value: object
) -> Tuple[int, ...]:
    if isinstance(stride_value, torch.Size):
        stride_value = tuple(stride_value)
    if isinstance(stride_value, (list, tuple)):
        try:
            return tuple(int(operator.index(item)) for item in stride_value)
        except TypeError:
            try:
                return tuple(int(item) for item in stride_value)
            except TypeError as exc:
                raise CodegenBackendError(
                    f"codegen {op_name} expects stride values to be integers"
                ) from exc
    raise CodegenBackendError(f"codegen {op_name} expects stride to be a sequence")


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
    size = _parse_resize_size(op_spec.name, size_arg)
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
        output_strides = _channels_last_strides(output_shape)
    elif memory_format is torch.channels_last_3d:
        output_strides = _channels_last_3d_strides(output_shape)
    else:
        raise CodegenBackendError("Unsupported memory formatPreserve")
    strides[node] = output_strides
    return op_node


def _parse_where_inputs(
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
            raise _error_expected_tensor(op_spec.name)
        if arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
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
            raise _error_expected_tensor(op_spec.name)
        params[scalar_key] = _normalize_scalar_value(op_spec.name, arg)

    add_tensor_arg(cond_arg)
    add_where_value(a_arg, "a_scalar")
    add_where_value(b_arg, "b_scalar")
    return input_nodes, input_shapes, params


def _analyze_generic_graph(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object],
    *,
    supported_ops: Dict[str, _OpSpec],
    target_registry: Dict[object, "_TargetInfo"],
    kind_handlers: Dict[OpKind, "OpKindHandler"],
) -> _GenericGraph:
    tensor_examples = list(_iter_example_tensors(example_inputs))
    if tensor_examples:
        dtype_info = _validate_example_inputs(example_inputs)
    else:
        dtype_info = _infer_empty_strided_dtype(
            gm,
            target_registry=target_registry,
            kind_handlers=kind_handlers,
        )
    output_node = None
    placeholders: List[torch.fx.Node] = []
    tensor_placeholders: List[torch.fx.Node] = []
    op_nodes: List[_OpNode] = []
    shapes: Dict[torch.fx.Node, Tuple[int, ...]] = {}
    strides: Dict[torch.fx.Node, Tuple[int, ...]] = {}
    dtypes: Dict[torch.fx.Node, torch.dtype] = {}
    scalar_values: Dict[torch.fx.Node, object] = {}
    alias_map: Dict[torch.fx.Node, torch.fx.Node] = {}
    empty_outputs: set[torch.fx.Node] = set()
    input_iter = iter(example_inputs)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            try:
                example = next(input_iter)
            except StopIteration as exc:
                raise CodegenBackendError(
                    "codegen backend expects example inputs to match placeholder count"
                ) from exc
            placeholders.append(node)
            if isinstance(example, torch.Tensor):
                if example.dtype not in _CODEGEN_DTYPES:
                    if example.dtype in _EMBEDDING_INDEX_DTYPES:
                        shapes[node] = tuple(example.shape)
                        strides[node] = tuple(example.stride())
                        dtypes[node] = example.dtype
                        tensor_placeholders.append(node)
                    elif example.numel() == 1:
                        continue
                    continue
                shapes[node] = tuple(example.shape)
                strides[node] = tuple(example.stride())
                dtypes[node] = example.dtype
                tensor_placeholders.append(node)
            else:
                if isinstance(example, numbers.Number):
                    scalar_values[node] = example
                else:
                    try:
                        scalar_values[node] = operator.index(example)
                    except TypeError:
                        pass
            continue
        if node.op in {"call_function", "call_method"}:
            if node.op == "call_function" and node.target is operator.getitem:
                if node.kwargs:
                    raise CodegenBackendError(
                        "codegen backend expects getitem to use positional args"
                    )
                if len(node.args) != 2:
                    raise CodegenBackendError(
                        "codegen backend expects getitem to have two inputs"
                    )
                source, index = node.args
                if not isinstance(source, torch.fx.Node):
                    raise CodegenBackendError(
                        "codegen backend expects getitem source to be a tensor op"
                    )
                if isinstance(index, torch.fx.Node):
                    raise CodegenBackendError(
                        "codegen backend supports only constant getitem indices"
                    )
                if index not in (0, 0.0, 1, 1.0, 2, 2.0):
                    raise CodegenBackendError(
                        "codegen backend supports only getitem[0], getitem[1], or getitem[2]"
                    )
                if source not in shapes:
                    raise CodegenBackendError(
                        "codegen backend expects getitem source to be analyzed"
                    )
                if source.target not in {
                    torch.ops.aten._native_batch_norm_legit,
                    torch.ops.aten._native_batch_norm_legit.default,
                    torch.ops.aten._native_batch_norm_legit_no_training,
                    torch.ops.aten._native_batch_norm_legit_no_training.default,
                    torch.ops.aten._embedding_bag,
                    torch.ops.aten._embedding_bag.default,
                }:
                    raise CodegenBackendError(
                        "codegen backend supports getitem only for _native_batch_norm_legit* ops"
                    )
                if index in (0, 0.0):
                    alias_map[node] = source
                    shapes[node] = shapes[source]
                    strides[node] = strides[source]
                    dtypes[node] = dtypes[source]
                else:
                    shapes[node] = (0,)
                    strides[node] = _contiguous_strides(shapes[node])
                    dtypes[node] = dtypes[source]
                    empty_outputs.add(node)
                continue
            if node.op == "call_method":
                if node.target == "item":
                    continue
                if node.target not in {
                    "sum",
                    "prod",
                    "mean",
                    "std",
                    "any",
                    "all",
                    "argmax",
                    "argmin",
                    "softmax",
                    "log_softmax",
                    "diagonal",
                    "cumsum",
                    "clone",
                }:
                    raise CodegenBackendError(f"Unsupported call_method: {node.target}")
                op_spec = supported_ops[node.target]
                inplace_input = None
            else:
                target_info = target_registry.get(node.target)
                if target_info is None:
                    raise CodegenBackendError(f"Unsupported call_function: {node.target}")
                op_spec = target_info.op_spec
                inplace_input = target_info.inplace_arg_index
            handler = kind_handlers.get(op_spec.kind)
            if handler is None:
                raise CodegenBackendError(
                    "codegen backend does not support kind "
                    f"'{op_spec.kind.value}'"
                )
            build_result = handler.build_op_node(
                node,
                op_spec,
                dtype_info,
                shapes,
                strides,
                dtypes,
                scalar_values,
                inplace_input,
            )
            if build_result is None:
                if dtype_info is None:
                    raise CodegenBackendError(
                        "codegen backend requires at least one tensor input or a factory op dtype"
                    )
                raise CodegenBackendError(
                    "codegen backend does not support building kind "
                    f"'{op_spec.kind.value}'"
                )
            op_nodes.append(build_result.op_node)
            if build_result.dtype_info is not None:
                dtype_info = build_result.dtype_info
            continue
        if node.op == "output":
            output_node = node
            continue
        raise CodegenBackendError(f"Unsupported node op: {node.op}")

    try:
        next(input_iter)
    except StopIteration:
        pass
    else:
        raise CodegenBackendError(
            "codegen backend expects example inputs to match placeholder count"
        )

    if not op_nodes:
        raise CodegenBackendError("codegen backend requires at least one operation")
    if output_node is None:
        raise CodegenBackendError("codegen backend requires an output node")
    if not tensor_placeholders and dtype_info is None:
        raise CodegenBackendError(
            "codegen backend requires at least one tensor input or a factory op dtype"
        )
    if dtype_info is None:
        raise CodegenBackendError("codegen backend could not infer a graph dtype")
    output_value, output_structure = _unwrap_output_node(output_node)
    while output_value in alias_map:
        output_value = alias_map[output_value]
    if output_value not in shapes:
        raise CodegenBackendError("codegen backend expects a single output node")
    if output_value not in {op.node for op in op_nodes}:
        raise CodegenBackendError("codegen backend output must be an operator result")

    output_op = next(op for op in op_nodes if op.node is output_value)
    for op_node in op_nodes:
        if (
            op_node.spec.kind == OpKind.EMPTY_STRIDED
            and op_node.node is not output_value
            and not _is_contiguous(op_node.output_shape, strides[op_node.node])
        ):
            raise CodegenBackendError(
                "codegen empty_strided supports non-contiguous strides only for outputs"
            )

    output_inplace_input = None
    for op_node in op_nodes:
        if op_node.node is output_value and op_node.inplace_input is not None:
            candidate = op_node.inputs[op_node.inplace_input]
            if candidate in tensor_placeholders:
                output_inplace_input = candidate
            break

    return _GenericGraph(
        placeholders=placeholders,
        tensor_placeholders=tensor_placeholders,
        op_nodes=op_nodes,
        output_node=output_node,
        output_value=output_value,
        output_op=output_op,
        output_inplace_input=output_inplace_input,
        output_structure=output_structure,
        shapes=shapes,
        strides=strides,
        dtypes=dtypes,
        dtype=dtype_info,
        alias_map=alias_map,
        empty_outputs=empty_outputs,
    )


def _compile_generic_library(
    graph: _GenericGraph,
    *,
    write_generic_source: Callable[[_GenericGraph], str],
) -> _GenericLibrary:
    source = write_generic_source(graph)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    entry_name = f"ref_codegen_main_{graph.dtype.suffix}"
    input_shapes = tuple(graph.shapes[node] for node in graph.tensor_placeholders)
    input_strides = tuple(graph.strides[node] for node in graph.tensor_placeholders)
    return compile_or_load(
        source,
        digest,
        entry_name=entry_name,
        include_dirs=[_C_SRC_DIR],
        input_shapes=input_shapes,
        input_strides=input_strides,
        output_shape=graph.shapes[graph.output_value],
        dtype=graph.dtype,
    )


def _validate_runtime_inputs(
    inputs: Iterable[torch.Tensor],
    expected_dtypes: Sequence[torch.dtype],
    graph_dtype: _CodegenDType,
) -> None:
    for tensor, expected_dtype in zip(inputs, expected_dtypes):
        if expected_dtype is torch.bool:
            if tensor.dtype is not torch.bool:
                raise CodegenBackendError(
                    "codegen backend expects boolean condition tensors"
                )
        elif expected_dtype in _EMBEDDING_INDEX_DTYPES:
            if tensor.dtype is not expected_dtype:
                raise CodegenBackendError(
                    "codegen backend expects int32 or int64 index tensors"
                )
        elif tensor.dtype is not graph_dtype.torch_dtype:
            raise CodegenBackendError(
                f"codegen backend supports only {graph_dtype.torch_dtype} tensors"
            )
        if tensor.device.type != "cpu":
            raise CodegenBackendError("codegen backend supports only CPU tensors")


def _compile_graph(
    gm: torch.fx.GraphModule,
    example_inputs: List[object],
    *,
    analyze_generic_graph: Callable[
        [torch.fx.GraphModule, Sequence[object]], _GenericGraph
    ],
    compile_generic_library: Callable[[_GenericGraph], _GenericLibrary],
) -> Callable[..., torch.Tensor]:
    graph = analyze_generic_graph(gm, example_inputs)
    conv_contiguous_indices = tuple(
        sorted(
            {
                graph.tensor_placeholders.index(input_node)
                for op_node in graph.op_nodes
                if op_node.spec.kind in {OpKind.CONV1D, OpKind.CONV2D}
                for input_node in op_node.inputs
                if input_node in graph.tensor_placeholders
            }
        )
    )

    def _normalize_conv_inputs(
        inputs: Sequence[object],
    ) -> List[object]:
        normalized = list(inputs)
        for index in conv_contiguous_indices:
            placeholder = graph.tensor_placeholders[index]
            placeholder_index = graph.placeholders.index(placeholder)
            value = normalized[placeholder_index]
            if isinstance(value, torch.Tensor) and not value.is_contiguous():
                normalized[placeholder_index] = value.contiguous()
        return normalized

    normalized_example_inputs = (
        _normalize_conv_inputs(example_inputs)
        if conv_contiguous_indices
        else list(example_inputs)
    )
    graph = analyze_generic_graph(gm, normalized_example_inputs)
    lib = compile_generic_library(graph)
    output_structure = graph.output_structure
    output_value = graph.output_value
    output_inplace_input = graph.output_inplace_input
    library_cache: Dict[
        Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]],
        _GenericLibrary,
    ] = {
        (lib.input_shapes, lib.input_strides): lib,
    }

    def _recompile(new_inputs: Sequence[object]) -> None:
        nonlocal graph, lib, output_inplace_input
        graph = analyze_generic_graph(gm, _normalize_conv_inputs(new_inputs))
        lib = compile_generic_library(graph)
        output_inplace_input = graph.output_inplace_input

    def resolve_output(value: object, env: Dict[torch.fx.Node, object]) -> object:
        if isinstance(value, torch.fx.Node):
            return env[value]
        if isinstance(value, (list, tuple, immutable_list)):
            resolved = [resolve_output(item, env) for item in value]
            return type(value)(resolved)
        return value

    def compiled(*args: object, **kwargs: object) -> object:
        if kwargs:
            placeholder_targets = [node.target for node in graph.placeholders]
            normalized_args = list(args)
            for name in placeholder_targets[len(normalized_args) :]:
                if name in kwargs:
                    normalized_args.append(kwargs[name])
        else:
            normalized_args = list(args)
        if len(normalized_args) != len(graph.placeholders):
            raise CodegenBackendError(
                f"codegen backend expects {len(graph.placeholders)} inputs, got {len(normalized_args)}"
            )
        env: Dict[torch.fx.Node, object] = {}
        input_tensors = []
        for node, value in zip(graph.placeholders, normalized_args):
            env[node] = value
            if node in graph.tensor_placeholders:
                if not isinstance(value, torch.Tensor):
                    raise CodegenBackendError("codegen backend expects tensor inputs only")
                input_tensors.append(value)
        expected_dtypes = [graph.dtypes[node] for node in graph.tensor_placeholders]
        _validate_runtime_inputs(input_tensors, expected_dtypes, graph.dtype)

        contiguous_inputs = list(input_tensors)
        if conv_contiguous_indices:
            for index in conv_contiguous_indices:
                if not contiguous_inputs[index].is_contiguous():
                    contiguous_inputs[index] = contiguous_inputs[
                        index
                    ].contiguous()

        input_shapes = tuple(tuple(tensor.shape) for tensor in contiguous_inputs)
        input_strides = tuple(tuple(tensor.stride()) for tensor in contiguous_inputs)
        cache_key = (input_shapes, input_strides)
        cached_lib = library_cache.get(cache_key)
        if cached_lib is None:
            analysis_inputs = list(normalized_args)
            if conv_contiguous_indices:
                for index in conv_contiguous_indices:
                    placeholder = graph.tensor_placeholders[index]
                    placeholder_index = graph.placeholders.index(placeholder)
                    analysis_inputs[placeholder_index] = contiguous_inputs[
                        index
                    ]
            updated_graph = analyze_generic_graph(gm, analysis_inputs)
            cached_lib = compile_generic_library(updated_graph)
            library_cache[cache_key] = cached_lib
        lib = cached_lib
        if output_inplace_input is not None:
            original_input = env[output_inplace_input]
            if not isinstance(original_input, torch.Tensor):
                raise CodegenBackendError("codegen backend expects tensor inputs only")
            inplace_index = graph.tensor_placeholders.index(output_inplace_input)
            inplace_out = contiguous_inputs[inplace_index]
            lib.run(contiguous_inputs, inplace_out)
            if inplace_out is not original_input:
                original_input.copy_(inplace_out)
            env[output_value] = original_input
        else:
            output_dtype = graph.dtypes[output_value]
            device = (
                contiguous_inputs[0].device
                if contiguous_inputs
                else torch.device("cpu")
            )
            if graph.output_op.spec.kind == OpKind.EMPTY_STRIDED:
                out = torch.empty_strided(
                    graph.shapes[output_value],
                    graph.strides[output_value],
                    dtype=output_dtype,
                    device=device,
                )
            else:
                out = torch.empty(
                    lib.output_shape,
                    dtype=output_dtype,
                    device=device,
                )
            lib.run(contiguous_inputs, out)
            env[output_value] = out
        if graph.alias_map:
            for alias, source in graph.alias_map.items():
                resolved = _resolve_alias(source, graph.alias_map)
                if resolved in env:
                    env[alias] = env[resolved]
        if graph.empty_outputs:
            device = (
                contiguous_inputs[0].device
                if contiguous_inputs
                else torch.device("cpu")
            )
            for node in graph.empty_outputs:
                if node not in env:
                    env[node] = torch.empty(
                        graph.shapes[node],
                        dtype=graph.dtypes[node],
                        device=device,
                    )
        return resolve_output(output_structure, env)

    return compiled


class BackendContext(HandlerContext):
    def __init__(self, backend: "CodegenBackend") -> None:
        self._backend = backend

    @property
    def kind_handlers(self) -> Dict[OpKind, "OpKindHandler"]:
        return self._backend.kind_handlers

    def handle_col2im_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_col2im_node(
            node, op_spec, dtype_info, shapes, strides, dtypes
        )

    def handle_batch_norm_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
    ) -> _OpNode:
        return _handle_batch_norm_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            scalar_values,
            self._backend.kind_handlers,
        )

    def handle_pdist_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_pdist_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_cdist_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_cdist_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_diagonal_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_diagonal_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_addmm_like_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        inplace_input: int | None,
    ) -> _OpNode:
        return _handle_addmm_like_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            inplace_input,
            self._backend.kind_handlers,
        )

    def handle_linear_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
    ) -> _OpNode:
        return _handle_linear_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_flip_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_flip_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_cumsum_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_cumsum_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_pad_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_constant_pad_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_gather_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_gather_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_view_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_view_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            self._backend.kind_handlers,
        )

    def handle_fill_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        inplace_input: int | None,
    ) -> _OpNode:
        return _handle_fill_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            inplace_input,
            self._backend.kind_handlers,
        )

    def handle_resize_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        inplace_input: int | None,
    ) -> _OpNode:
        return _handle_resize_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            inplace_input,
            self._backend.kind_handlers,
        )

    def handle_to_copy_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
    ) -> _OpNode:
        return _handle_to_copy_node(
            node, op_spec, dtype_info, shapes, strides, dtypes
        )

    def kernel_inputs(self, op_node: _OpNode) -> List[torch.fx.Node]:
        return _kernel_inputs(op_node)


class CodegenBackend:
    def __init__(
        self,
        *,
        group_registry: object | None = None,
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
        self._handler_context = BackendContext(self)

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
                self._handler_context
            )
        return self._kind_handlers

    def get_generic_source(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> str:
        graph = self._analyze_generic_graph(gm, example_inputs)
        return self._write_generic_source(graph)

    def codegen_generic_backend(
        self, gm: torch.fx.GraphModule, example_inputs: List[object]
    ) -> Callable[..., torch.Tensor]:
        return self._compile_graph(gm, example_inputs)

    def _write_generic_source(self, graph: _GenericGraph) -> str:
        return _write_generic_source(
            graph,
            kind_handlers=self.kind_handlers,
            templates_env=self.templates_env,
        )

    def _analyze_generic_graph(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> _GenericGraph:
        return _analyze_generic_graph(
            gm,
            example_inputs,
            supported_ops=self.supported_ops,
            target_registry=self.target_registry,
            kind_handlers=self.kind_handlers,
        )

    def _compile_generic_library(self, graph: _GenericGraph) -> _GenericLibrary:
        return _compile_generic_library(
            graph, write_generic_source=self._write_generic_source
        )

    def _compile_graph(
        self, gm: torch.fx.GraphModule, example_inputs: List[object]
    ) -> Callable[..., torch.Tensor]:
        return _compile_graph(
            gm,
            example_inputs,
            analyze_generic_graph=self._analyze_generic_graph,
            compile_generic_library=self._compile_generic_library,
        )


_DEFAULT_BACKEND = CodegenBackend()


def get_generic_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> str:
    return _DEFAULT_BACKEND.get_generic_source(gm, example_inputs)


def codegen_generic_backend(
    gm: torch.fx.GraphModule, example_inputs: List[object]
) -> Callable[..., torch.Tensor]:
    return _DEFAULT_BACKEND.codegen_generic_backend(gm, example_inputs)
