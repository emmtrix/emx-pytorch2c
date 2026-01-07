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
from codegen_backend.emitters.arange import ArangeEmitter
from codegen_backend.emitters.addr import AddrEmitter
from codegen_backend.emitters.concat import ConcatEmitter
from codegen_backend.emitters.conv1d import Conv1dEmitter
from codegen_backend.emitters.conv2d import Conv2dEmitter
from codegen_backend.emitters.embedding import EmbeddingEmitter
from codegen_backend.emitters.embedding_bag import EmbeddingBagEmitter
from codegen_backend.emitters.empty_strided import EmptyStridedEmitter
from codegen_backend.emitters.elementwise import (
    _FLOAT_ONLY_UNARY_OPS,
    _PARAMETRIC_UNARY_OPS,
)
from codegen_backend.emitters.elementwise import ElementwiseEmitter
from codegen_backend.emitters.matmul import MatmulEmitter
from codegen_backend.emitters.pool1d import Pool1dEmitter
from codegen_backend.emitters.pool2d import Pool2dEmitter
from codegen_backend.emitters.pool2d_backward import Pool2dBackwardEmitter
from codegen_backend.emitters.pool3d import Pool3dEmitter
from codegen_backend.emitters.reduction import ReductionEmitter
from codegen_backend.emitters.argreduction import ArgReductionEmitter
from codegen_backend.emitters.softmax import SoftmaxEmitter
from codegen_backend.indexing import (
    _contiguous_strides,
    _emit_strided_access,
    _format_strided_access,
)
from codegen_backend.kinds import (
    ArangeHandler,
    ConcatHandler,
    Conv1dHandler,
    Conv2dHandler,
    EmbeddingBagHandler,
    EmbeddingHandler,
    EmptyStridedHandler,
    ElementwiseHandler,
    HandlerContext,
    OpNodeBuildResult,
    ArgReductionHandler,
    AddrHandler,
    MatmulHandler,
    ReductionHandler,
    Pool1dHandler,
    Pool2dBackwardHandler,
    Pool2dHandler,
    Pool3dHandler,
    SoftmaxHandler,
    build_kind_handlers,
)
from codegen_backend.ops_registry import SUPPORTED_OPS
from codegen_backend.param_normalize import (
    normalize_bool,
    normalize_int_or_pair,
    normalize_int_or_tuple,
    normalize_padding,
)
from codegen_backend.registry import TARGET_REGISTRY
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.templates import get_template_env
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


def _write_generic_source(graph: _GenericGraph) -> str:
    placeholders = graph.tensor_placeholders
    op_nodes = graph.op_nodes
    headers = [
        "#include <stdint.h>",
        "#include <stdbool.h>",
        f"#include \"{graph.dtype.scalar_header}\"",
    ]
    kernels: List[str] = []
    for index, op_node in enumerate(op_nodes, start=1):
        handler = _KIND_HANDLERS.get(op_node.spec.kind)
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
    template = get_template_env().get_template("generic_source.c.j2")
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
) -> _CodegenDType | None:
    dtype_value = None
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target_info = TARGET_REGISTRY.get(node.target)
        if target_info is None:
            continue
        handler = _KIND_HANDLERS.get(target_info.op_spec.kind)
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
    op_node: _OpNode, input_shapes: Sequence[Tuple[int, ...]]
) -> Tuple[int, ...]:
    handler = _KIND_HANDLERS.get(op_node.spec.kind)
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


def _parse_constant_float(op_name: str, name: str, value: object) -> float:
    if isinstance(value, torch.fx.Node):
        raise CodegenBackendError(f"codegen {op_name} expects {name} to be constant")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise CodegenBackendError(f"codegen {op_name} expects {name} to be numeric")


def _parse_constant_int(op_name: str, name: str, value: object) -> int:
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise CodegenBackendError(f"codegen {op_name} expects {name} to be an int")
        return _parse_constant_int(op_name, name, meta_value)
    if isinstance(value, bool):
        raise CodegenBackendError(f"codegen {op_name} expects {name} to be an int")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(f"codegen {op_name} expects {name} to be an int")
        return _parse_constant_int(op_name, name, value.item())
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects {name} to be an int"
            ) from exc
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise CodegenBackendError(f"codegen {op_name} expects {name} to be an int")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects {name} to be an int"
        ) from exc


def _parse_constant_bool(op_name: str, name: str, value: object) -> bool:
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise CodegenBackendError(
                f"codegen {op_name} expects {name} to be a bool"
            )
        return _parse_constant_bool(op_name, name, meta_value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(
                f"codegen {op_name} expects {name} to be a bool"
            )
        return _parse_constant_bool(op_name, name, value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    raise CodegenBackendError(f"codegen {op_name} expects {name} to be a bool")


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
    output_shape = _infer_output_shape(op_node, [input_shape])
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
    output_shape = _infer_output_shape(op_node, [input_shape])
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
    output_shape = _infer_output_shape(op_node, [input_shape])
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
    output_shape = _infer_output_shape(op_node, [x1_shape, x2_shape])
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
    output_shape = _infer_output_shape(op_node, input_shapes)
    op_node.output_shape = output_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    if inplace_input is not None:
        strides[node] = strides[input_nodes[inplace_input]]
    else:
        strides[node] = _contiguous_strides(output_shape)
    return op_node



def _handle_diagonal_node(
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
    output_shape = _infer_output_shape(op_node, [shapes[input_arg]])
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
    output_shape = _infer_output_shape(op_node, [shapes[input_arg]])
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
    output_shape = _infer_output_shape(op_node, [input_shape])
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
    output_shape = _infer_output_shape(op_node, [input_shape, index_shape])
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
    output_shape = _infer_output_shape(op_node, [shapes[input_arg]])
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
        output_shape = _infer_output_shape(op_node, [shapes[input_arg]])
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
        output_shape = _infer_output_shape(op_node, [input_shape])
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
    output_shape = _infer_output_shape(op_node, [input_shape])
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
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> _GenericGraph:
    tensor_examples = list(_iter_example_tensors(example_inputs))
    if tensor_examples:
        dtype_info = _validate_example_inputs(example_inputs)
    else:
        dtype_info = _infer_empty_strided_dtype(gm)
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
                op_spec = SUPPORTED_OPS[node.target]
                inplace_input = None
            else:
                target_info = TARGET_REGISTRY.get(node.target)
                if target_info is None:
                    raise CodegenBackendError(f"Unsupported call_function: {node.target}")
                op_spec = target_info.op_spec
                inplace_input = target_info.inplace_arg_index
            handler = _KIND_HANDLERS.get(op_spec.kind)
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


def _compile_generic_library(graph: _GenericGraph) -> _GenericLibrary:
    source = _write_generic_source(graph)
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
    gm: torch.fx.GraphModule, example_inputs: List[object]
) -> Callable[..., torch.Tensor]:
    graph = _analyze_generic_graph(gm, example_inputs)
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
    graph = _analyze_generic_graph(gm, normalized_example_inputs)
    lib = _compile_generic_library(graph)
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
        graph = _analyze_generic_graph(
            gm, _normalize_conv_inputs(new_inputs)
        )
        lib = _compile_generic_library(graph)
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
            updated_graph = _analyze_generic_graph(gm, analysis_inputs)
            cached_lib = _compile_generic_library(updated_graph)
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


def get_generic_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> str:
    graph = _analyze_generic_graph(gm, example_inputs)
    return _write_generic_source(graph)


class _BackendElementwiseHandler(ElementwiseHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if op_spec.name == "_to_copy":
            if dtype_info is None:
                return None
            handler = getattr(self._ctx, "handle_to_copy_node", None)
            if handler is None:
                return None
            op_node = handler(node, op_spec, dtype_info, shapes, strides, dtypes)
            return OpNodeBuildResult(op_node)
        if op_spec.kind == OpKind.FILL:
            if dtype_info is None:
                return None
            handler = getattr(self._ctx, "handle_fill_node", None)
            if handler is None:
                return None
            op_node = handler(
                node,
                op_spec,
                dtype_info,
                shapes,
                strides,
                dtypes,
                inplace_input,
            )
            return OpNodeBuildResult(op_node)
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        param_values: Dict[str, object] = {}
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[Tuple[int, ...]] = []
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
                    raise _error_expected_tensor(op_spec.name)
                input_nodes = [input_arg]
                input_shapes = [shapes[input_arg]]
                if op_spec.name in _BITWISE_OPS:
                    param_values["scalar"] = _parse_bitwise_scalar(
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
                        param_values["scalar"] = _parse_bitwise_scalar(
                            op_spec.name, scalar_arg, dtype_info.torch_dtype
                        )
                    else:
                        param_values["scalar"] = _resolve_scalar_arg(
                            op_spec.name, scalar_arg, scalar_values
                        )

        if not input_nodes:
            if (
                op_spec.kind == OpKind.UNARY
                and op_spec.name in _PARAMETRIC_UNARY_OPS
            ):
                input_node, param_values = _parse_parametric_unary_args(
                    op_spec.name, node
                )
                args_to_check = (input_node,)
            else:
                allowed_kwargs = set()
                is_out_overload = _is_out_overload(node.target)
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
                            raise _error_kwarg_specified_once(
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
                            raise _error_kwarg_specified_once(
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
                ) = _parse_where_inputs(op_spec, node, shapes, scalar_values)
                param_values.update(where_params)
            else:
                for arg in args_to_check:
                    if not isinstance(arg, torch.fx.Node):
                        raise _error_expected_tensor(op_spec.name)
                    if arg not in shapes:
                        raise _error_expected_tensor(op_spec.name)
                    input_nodes.append(arg)
                    input_shapes.append(shapes[arg])
            if out_arg is not None and out_arg not in input_nodes:
                if not isinstance(out_arg, torch.fx.Node):
                    raise _error_expected_tensor(op_spec.name)
                if out_arg not in shapes:
                    raise _error_expected_tensor(op_spec.name)
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
        output_shape = _infer_output_shape(op_node, shape_input_shapes)
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


class _BackendReductionHandler(ReductionHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        if len(node.args) < 1:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects one input"
            )
        args_to_check = node.args[:1]
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[Tuple[int, ...]] = []
        for arg in args_to_check:
            if not isinstance(arg, torch.fx.Node):
                raise _error_expected_tensor(op_spec.name)
            if arg not in shapes:
                raise _error_expected_tensor(op_spec.name)
            input_nodes.append(arg)
            input_shapes.append(shapes[arg])
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        param_values: Dict[str, object] = {}
        if op_spec.name == "norm":
            if dtype_info.torch_dtype is not torch.float32:
                raise CodegenBackendError(
                    "codegen norm supports only torch.float32 tensors"
                )
            reduction_dims, keepdim, reduce_all, norm_p = _parse_norm_args(
                op_spec.name, node, input_shapes[0]
            )
            param_values["norm_p"] = norm_p
        else:
            reduction_dims, keepdim, reduce_all, unbiased = _parse_reduction_args(
                op_spec.name, node, input_shapes[0]
            )
            if unbiased is not None:
                param_values["unbiased"] = unbiased
            if (
                op_spec.name == "var"
                and dtype_info.torch_dtype is not torch.float32
            ):
                raise CodegenBackendError(
                    "codegen var supports only torch.float32 tensors"
                )
        param_values["reduce_all"] = reduce_all
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
            reduction_dims=reduction_dims,
            keepdim=keepdim,
            params=param_values,
        )
        self.validate(op_node, input_shapes, input_dtypes, dtype_info)
        output_shape = _infer_output_shape(op_node, input_shapes)
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        if inplace_input is not None:
            strides[node] = strides[input_nodes[inplace_input]]
        else:
            strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendArgReductionHandler(ArgReductionHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        if len(node.args) < 1:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects one input"
            )
        args_to_check = node.args[:1]
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[Tuple[int, ...]] = []
        for arg in args_to_check:
            if not isinstance(arg, torch.fx.Node):
                raise _error_expected_tensor(op_spec.name)
            if arg not in shapes:
                raise _error_expected_tensor(op_spec.name)
            input_nodes.append(arg)
            input_shapes.append(shapes[arg])
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        reduction_dims, keepdim, reduce_all = _parse_argminmax_args(
            op_spec.name, node, input_shapes[0]
        )
        reduction_count = 1
        if reduce_all:
            for size in input_shapes[0]:
                reduction_count *= size
        else:
            for dim in reduction_dims:
                reduction_count *= input_shapes[0][dim]
        if reduction_count == 0:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects a non-empty reduction dimension"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
            reduction_dims=reduction_dims,
            keepdim=keepdim,
            params={"reduce_all": reduce_all},
        )
        self.validate(op_node, input_shapes, input_dtypes, dtype_info)
        output_shape = _infer_output_shape(op_node, input_shapes)
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = torch.int64
        if inplace_input is not None:
            strides[node] = strides[input_nodes[inplace_input]]
        else:
            strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendMatmulHandler(MatmulHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        allowed_kwargs = set()
        is_out_overload = _is_out_overload(node.target)
        if is_out_overload:
            allowed_kwargs.add("out")
        if node.kwargs and set(node.kwargs) - allowed_kwargs:
            raise CodegenBackendError(
                "codegen backend expects positional args only"
            )
        expected_arity = 2
        out_arg: torch.fx.Node | None = None
        if is_out_overload:
            if inplace_input is None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects out to be provided"
                )
            if "out" in node.kwargs:
                if len(node.args) > expected_arity:
                    raise _error_kwarg_specified_once(op_spec.name, "out")
                out_arg = node.kwargs["out"]
            elif len(node.args) == expected_arity + 1:
                out_arg = node.args[inplace_input]
            elif len(node.args) != expected_arity:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects exactly two inputs"
                )
            if out_arg is None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects out to be provided"
                )
        elif len(node.args) != expected_arity:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects exactly two inputs"
            )
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[Tuple[int, ...]] = []
        for arg in node.args[:expected_arity]:
            if not isinstance(arg, torch.fx.Node):
                raise _error_expected_tensor(op_spec.name)
            if arg not in shapes:
                raise _error_expected_tensor(op_spec.name)
            input_nodes.append(arg)
            input_shapes.append(shapes[arg])
        if out_arg is not None and out_arg not in input_nodes:
            if not isinstance(out_arg, torch.fx.Node):
                raise _error_expected_tensor(op_spec.name)
            if out_arg not in shapes:
                raise _error_expected_tensor(op_spec.name)
            input_nodes.append(out_arg)
            input_shapes.append(shapes[out_arg])
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
        )
        self.validate(op_node, input_shapes, input_dtypes, dtype_info)
        output_shape = _infer_output_shape(op_node, input_shapes)
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


class _BackendAddrHandler(AddrHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        handler = getattr(self._ctx, "handle_addmm_like_node", None)
        if handler is None:
            return None
        op_node = handler(
            node, op_spec, dtype_info, shapes, strides, dtypes, inplace_input
        )
        if inplace_input is not None:
            input_shape = shapes[op_node.inputs[inplace_input]]
            output_shape = op_node.output_shape
            if len(input_shape) != 2:
                raise CodegenBackendError(
                    "codegen addr expects 2D input and 1D vectors"
                )
            if tuple(output_shape) != tuple(input_shape):
                raise CodegenBackendError(
                    "codegen addr expects input shape to match outer product output"
                )
        return OpNodeBuildResult(op_node)

class _BackendArangeHandler(ArangeHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        allowed_kwargs = {
            "start",
            "end",
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "step",
        }
        extra = set(node.kwargs) - allowed_kwargs
        if extra:
            raise CodegenBackendError(
                f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
            )
        if node.kwargs.get("layout") is not None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects layout to be None"
            )
        device = node.kwargs.get("device")
        if device is not None and device != "cpu" and device != torch.device("cpu"):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects device to be None or cpu"
            )
        pin_memory = node.kwargs.get("pin_memory")
        if pin_memory not in (None, False):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects pin_memory to be False"
            )
        start_arg = None
        end_arg = None
        step_arg = None
        if node.args:
            if len(node.args) > 3:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects start and end arguments"
                )
            start_arg = node.args[0]
            if len(node.args) > 1:
                end_arg = node.args[1]
            if len(node.args) > 2:
                step_arg = node.args[2]
        if "start" in node.kwargs:
            if start_arg is not None:
                raise _error_kwarg_specified_once(op_spec.name, "start")
            start_arg = node.kwargs["start"]
        if "end" in node.kwargs:
            if end_arg is not None:
                raise _error_kwarg_specified_once(op_spec.name, "end")
            end_arg = node.kwargs["end"]
        if "step" in node.kwargs:
            if step_arg is not None:
                raise _error_kwarg_specified_once(op_spec.name, "step")
            step_arg = node.kwargs["step"]
        if start_arg is None or end_arg is None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects start and end arguments"
            )
        if step_arg is None:
            step_arg = 1
        start = _resolve_scalar_arg(op_spec.name, start_arg, scalar_values)
        end = _resolve_scalar_arg(op_spec.name, end_arg, scalar_values)
        step = _resolve_scalar_arg(op_spec.name, step_arg, scalar_values)
        for name, value in (("start", start), ("end", end), ("step", step)):
            if not isinstance(value, numbers.Real):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to be an int or float"
                )
        dtype_spec = _parse_arange_dtype(
            op_spec.name, node.kwargs.get("dtype"), dtype_info, start, end, step
        )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[],
            output_shape=(),
            params={"start": start, "end": end, "step": step},
        )
        output_shape = _infer_output_shape(op_node, [])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_spec.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node, dtype_spec)


class _BackendConcatHandler(ConcatHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        input_nodes, concat_dim = _parse_concat_args(node)
        input_shapes: List[Tuple[int, ...]] = []
        for arg in input_nodes:
            if arg not in shapes:
                raise _error_expected_tensor("cat")
            input_shapes.append(shapes[arg])
        if not input_shapes:
            raise CodegenBackendError(
                "codegen cat expects a non-empty tensor list input"
            )
        rank = len(input_shapes[0])
        if rank == 0:
            raise CodegenBackendError("codegen cat expects inputs with rank >= 1")
        if concat_dim < 0:
            concat_dim += rank
        if concat_dim < 0 or concat_dim >= rank:
            raise CodegenBackendError("codegen cat dim is out of range")
        for shape in input_shapes:
            if len(shape) != rank:
                raise CodegenBackendError(
                    "codegen cat expects inputs with the same rank"
                )
            for dim, size in enumerate(shape):
                if dim == concat_dim:
                    continue
                if size != input_shapes[0][dim]:
                    raise CodegenBackendError(
                        "codegen cat expects input shapes to match except in the concat dimension"
                    )
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                "codegen cat expects inputs to share the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=None,
            params={"dim": concat_dim},
        )
        output_shape = _infer_output_shape(op_node, input_shapes)
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendEmptyStridedHandler(EmptyStridedHandler):
    def infer_graph_dtype(
        self, node: torch.fx.Node, op_spec: _OpSpec
    ) -> torch.dtype | None:
        node_dtype = None
        if len(node.args) > 2:
            node_dtype = node.args[2]
        if "dtype" in node.kwargs:
            if node_dtype is not None:
                raise _error_kwarg_specified_once(op_spec.name, "dtype")
            node_dtype = node.kwargs["dtype"]
        if isinstance(node_dtype, torch.fx.Node):
            raise CodegenBackendError(
                "codegen empty_strided expects dtype to be a constant"
            )
        if node_dtype is None:
            raise CodegenBackendError(
                "codegen empty_strided requires dtype when no tensor inputs are provided"
            )
        return node_dtype

    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        if len(node.args) < 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects size and stride arguments"
            )
        if len(node.args) > 7:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects at most seven arguments"
            )
        size_arg, stride_arg = node.args[:2]
        if isinstance(size_arg, torch.fx.Node) or isinstance(
            stride_arg, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects size and stride to be constants"
            )
        kwargs = dict(node.kwargs)
        positional_names = [
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "requires_grad",
        ]
        for index, name in enumerate(positional_names, start=2):
            if len(node.args) > index:
                if name in kwargs:
                    raise _error_kwarg_specified_once(op_spec.name, name)
                kwargs[name] = node.args[index]
        extra = set(kwargs) - {
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "requires_grad",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
            )
        dtype_value = kwargs.get("dtype")
        if dtype_value is not None and dtype_value is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects dtype to match the graph dtype"
            )
        for name in ("layout", "device"):
            if kwargs.get(name) is not None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to be None"
                )
        for name in ("pin_memory", "requires_grad"):
            if kwargs.get(name) not in (None, False):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to be False"
                )
        output_shape = _parse_resize_size(op_spec.name, size_arg)
        output_strides = _parse_empty_strided_stride(op_spec.name, stride_arg)
        if len(output_shape) != len(output_strides):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects size and stride to match length"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[],
            output_shape=(),
            params={"size": output_shape},
        )
        output_shape = _infer_output_shape(op_node, [])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = output_strides
        return OpNodeBuildResult(op_node)


class _BackendPool1dHandler(Pool1dHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        if op_spec.name == "adaptive_avg_pool1d":
            input_arg, output_size = _parse_adaptive_avg_pool1d_args(node)
            kernel_size = None
            stride = None
            padding = 0
            dilation = 1
            ceil_mode = False
            count_include_pad = False
            divisor_override = None
        elif op_spec.name == "max_pool1d":
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ) = _parse_max_pool1d_args(node)
            count_include_pad = False
            divisor_override = None
        else:
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            ) = _parse_avg_pool1d_args(node)
            dilation = 1
        if not isinstance(input_arg, torch.fx.Node):
            raise _error_expected_tensor(op_spec.name)
        if input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if isinstance(kernel_size, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(ceil_mode, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
            )
        if stride is not None and isinstance(stride, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant stride values"
            )
        if isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant dilation values"
            )
        if isinstance(count_include_pad, torch.fx.Node) or isinstance(
            divisor_override, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant pooling options"
            )
        input_shape = shapes[input_arg]
        if len(input_shape) != 3:
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires 3D input tensors"
            )
        if not _is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input tensors"
            )
        if op_spec.name == "adaptive_avg_pool1d":
            if isinstance(output_size, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d expects output_size to be an int"
                )
            if isinstance(output_size, (tuple, list)):
                if len(output_size) != 1:
                    raise CodegenBackendError(
                        "codegen adaptive_avg_pool1d expects a single output size"
                    )
                output_size = output_size[0]
            if not isinstance(output_size, int):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d expects output_size to be an int"
                )
            if output_size <= 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d expects output_size to be positive"
                )
            in_l = input_shape[2]
            if in_l % output_size != 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d requires input length divisible by output_size"
                )
            kernel_value = in_l // output_size
            stride_value = kernel_value
            padding_value = 0
            dilation_value = 1
        else:
            kernel_value = _normalize_param(
                normalize_int_or_tuple, "kernel_size", kernel_size, 1
            )[0]
            if stride is None:
                stride_value = kernel_value
            else:
                stride_value = _normalize_param(
                    normalize_int_or_tuple, "stride", stride, 1
                )[0]
            padding_value = _normalize_param(
                normalize_int_or_tuple, "padding", padding, 1
            )[0]
            dilation_value = _normalize_param(
                normalize_int_or_tuple, "dilation", dilation, 1
            )[0]
        if (
            kernel_value <= 0
            or stride_value <= 0
            or dilation_value <= 0
            or padding_value < 0
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
            )
        if ceil_mode and op_spec.name != "max_pool1d":
            raise CodegenBackendError(
                f"codegen {op_spec.name} does not support ceil_mode"
            )
        if isinstance(count_include_pad, bool):
            count_include_pad_value = count_include_pad
        elif isinstance(count_include_pad, numbers.Integral):
            count_include_pad_value = bool(count_include_pad)
        else:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects count_include_pad to be a bool"
            )
        divisor_override_value = divisor_override
        if divisor_override is not None:
            if isinstance(divisor_override, bool) or not isinstance(
                divisor_override, numbers.Integral
            ):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
            divisor_override_value = int(divisor_override)
            if divisor_override_value <= 0:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_value,
                "stride": stride_value,
                "padding": padding_value,
                "dilation": dilation_value,
                "ceil_mode": bool(ceil_mode),
                "count_include_pad": count_include_pad_value,
                "divisor_override": divisor_override_value,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendPool2dHandler(Pool2dHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        if op_spec.name == "adaptive_avg_pool2d":
            input_arg, output_size = _parse_adaptive_avg_pool2d_args(node)
            kernel_size = None
            stride = None
            padding = 0
            dilation = 1
            ceil_mode = False
            count_include_pad = False
            divisor_override = None
        elif op_spec.name == "max_pool2d":
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ) = _parse_max_pool2d_args(node)
            count_include_pad = False
            divisor_override = None
        else:
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            ) = _parse_avg_pool2d_args(node)
            dilation = 1
        if not isinstance(input_arg, torch.fx.Node):
            raise _error_expected_tensor(op_spec.name)
        if input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if isinstance(kernel_size, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(ceil_mode, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
            )
        if stride is not None and isinstance(stride, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant stride values"
            )
        if isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant dilation values"
            )
        if isinstance(count_include_pad, torch.fx.Node) or isinstance(
            divisor_override, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant pooling options"
            )
        input_shape = shapes[input_arg]
        if len(input_shape) != 4:
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires 4D input tensors"
            )
        if not _is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input tensors"
            )
        if op_spec.name == "adaptive_avg_pool2d":
            if isinstance(output_size, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
                )
            if isinstance(output_size, torch.Size):
                output_size = tuple(output_size)
            if isinstance(output_size, int):
                output_pair = (output_size, output_size)
            elif isinstance(output_size, (tuple, list)):
                if len(output_size) != 2:
                    raise CodegenBackendError(
                        "codegen adaptive_avg_pool2d expects output_size to have two values"
                    )
                output_pair = tuple(output_size)
            else:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
                )
            if not all(isinstance(item, int) for item in output_pair):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
                )
            if output_pair[0] <= 0 or output_pair[1] <= 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be positive"
                )
            in_h, in_w = input_shape[2], input_shape[3]
            if in_h % output_pair[0] != 0 or in_w % output_pair[1] != 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d requires input sizes divisible by output_size"
                )
            kernel_pair = (in_h // output_pair[0], in_w // output_pair[1])
            stride_pair = kernel_pair
            padding_pair = (0, 0)
            dilation_pair = (1, 1)
        else:
            kernel_pair = _normalize_param(
                normalize_int_or_pair, "kernel_size", kernel_size
            )
            if stride is None:
                stride_pair = kernel_pair
            else:
                stride_pair = _normalize_param(
                    normalize_int_or_pair, "stride", stride
                )
            padding_pair = _normalize_param(
                normalize_int_or_pair, "padding", padding
            )
            dilation_pair = _normalize_param(
                normalize_int_or_pair, "dilation", dilation
            )
        if (
            kernel_pair[0] <= 0
            or kernel_pair[1] <= 0
            or stride_pair[0] <= 0
            or stride_pair[1] <= 0
            or dilation_pair[0] <= 0
            or dilation_pair[1] <= 0
            or padding_pair[0] < 0
            or padding_pair[1] < 0
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
            )
        if not isinstance(ceil_mode, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects ceil_mode to be a bool"
            )
        if not isinstance(count_include_pad, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects count_include_pad to be a bool"
            )
        if divisor_override is not None:
            if not isinstance(divisor_override, int) or divisor_override <= 0:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_pair,
                "stride": stride_pair,
                "padding": padding_pair,
                "dilation": dilation_pair,
                "ceil_mode": bool(ceil_mode),
                "count_include_pad": count_include_pad,
                "divisor_override": divisor_override,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendPool3dHandler(Pool3dHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        input_arg, output_size = _parse_adaptive_avg_pool3d_args(node)
        kernel_size = None
        stride = None
        padding = (0, 0, 0)
        dilation = (1, 1, 1)
        ceil_mode = False
        count_include_pad = False
        divisor_override = None
        if not isinstance(input_arg, torch.fx.Node):
            raise _error_expected_tensor(op_spec.name)
        if input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if isinstance(kernel_size, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(ceil_mode, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
            )
        if stride is not None and isinstance(stride, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant stride values"
            )
        if isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant dilation values"
            )
        if isinstance(count_include_pad, torch.fx.Node) or isinstance(
            divisor_override, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant pooling options"
            )
        input_shape = shapes[input_arg]
        if len(input_shape) != 5:
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires 5D input tensors"
            )
        if not _is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input tensors"
            )
        if isinstance(output_size, torch.fx.Node):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be a tuple of ints"
            )
        if isinstance(output_size, torch.Size):
            output_size = tuple(output_size)
        if isinstance(output_size, int):
            output_triplet = (output_size, output_size, output_size)
        elif isinstance(output_size, (tuple, list)):
            if len(output_size) != 3:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool3d expects output_size to have three values"
                )
            output_triplet = tuple(output_size)
        else:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be a tuple of ints"
            )
        if not all(isinstance(item, int) for item in output_triplet):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be a tuple of ints"
            )
        if (
            output_triplet[0] <= 0
            or output_triplet[1] <= 0
            or output_triplet[2] <= 0
        ):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be positive"
            )
        in_d, in_h, in_w = input_shape[2], input_shape[3], input_shape[4]
        if (
            in_d % output_triplet[0] != 0
            or in_h % output_triplet[1] != 0
            or in_w % output_triplet[2] != 0
        ):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d requires input sizes divisible by output_size"
            )
        kernel_triplet = (
            in_d // output_triplet[0],
            in_h // output_triplet[1],
            in_w // output_triplet[2],
        )
        stride_triplet = kernel_triplet
        padding_triplet = (0, 0, 0)
        dilation_triplet = (1, 1, 1)
        if (
            kernel_triplet[0] <= 0
            or kernel_triplet[1] <= 0
            or kernel_triplet[2] <= 0
            or stride_triplet[0] <= 0
            or stride_triplet[1] <= 0
            or stride_triplet[2] <= 0
            or dilation_triplet[0] <= 0
            or dilation_triplet[1] <= 0
            or dilation_triplet[2] <= 0
            or padding_triplet[0] < 0
            or padding_triplet[1] < 0
            or padding_triplet[2] < 0
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
            )
        if ceil_mode:
            raise CodegenBackendError(
                f"codegen {op_spec.name} does not support ceil_mode"
            )
        if not isinstance(count_include_pad, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects count_include_pad to be a bool"
            )
        if divisor_override is not None:
            if not isinstance(divisor_override, int) or divisor_override <= 0:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_triplet,
                "stride": stride_triplet,
                "padding": padding_triplet,
                "dilation": dilation_triplet,
                "ceil_mode": bool(ceil_mode),
                "count_include_pad": count_include_pad,
                "divisor_override": divisor_override,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendPool2dBackwardHandler(Pool2dBackwardHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        grad_output, input_arg = _parse_adaptive_avg_pool2d_backward_args(node)
        if not isinstance(grad_output, torch.fx.Node) or not isinstance(
            input_arg, torch.fx.Node
        ):
            raise _error_expected_tensor(op_spec.name)
        if grad_output not in shapes or input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward supports only torch.float32 tensors"
            )
        if (
            dtypes[grad_output] is not torch.float32
            or dtypes[input_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward supports only torch.float32 tensors"
            )
        grad_output_shape = shapes[grad_output]
        input_shape = shapes[input_arg]
        if len(grad_output_shape) != 4 or len(input_shape) != 4:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires 4D input tensors"
            )
        if not _is_contiguous(grad_output_shape, strides[grad_output]):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires contiguous grad_output tensors"
            )
        if not _is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires contiguous input tensors"
            )
        if (
            grad_output_shape[0] != input_shape[0]
            or grad_output_shape[1] != input_shape[1]
        ):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires matching batch and channel sizes"
            )
        out_h, out_w = grad_output_shape[2], grad_output_shape[3]
        in_h, in_w = input_shape[2], input_shape[3]
        if out_h <= 0 or out_w <= 0:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward expects output size to be positive"
            )
        if in_h % out_h != 0 or in_w % out_w != 0:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires input sizes divisible by output_size"
            )
        kernel_pair = (in_h // out_h, in_w // out_w)
        if kernel_pair[0] <= 0 or kernel_pair[1] <= 0:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward expects positive kernel size"
            )
        stride_pair = kernel_pair
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[grad_output, input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_pair,
                "stride": stride_pair,
            },
        )
        output_shape = _infer_output_shape(op_node, [grad_output_shape, input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendSoftmaxHandler(SoftmaxHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        if not node.args:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        dim, dtype = _parse_softmax_args(op_spec.name, node, shapes[input_arg])
        if dtype is not None and dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects dtype to be torch.float32 or None"
            )
        output_shape = shapes[input_arg]
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=output_shape,
            inplace_input=None,
            params={"dim": dim},
        )
        return OpNodeBuildResult(op_node)


class _BackendEmbeddingHandler(EmbeddingHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        weight, indices, padding_idx, scale_grad_by_freq, sparse = (
            _parse_embedding_args(node)
        )
        if scale_grad_by_freq or sparse:
            raise CodegenBackendError(
                "codegen embedding supports only scale_grad_by_freq=False and sparse=False"
            )
        if not isinstance(weight, torch.fx.Node) or weight not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if not isinstance(indices, torch.fx.Node) or indices not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtypes[weight] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen embedding expects weight to match the graph dtype"
            )
        if dtypes[indices] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen embedding expects indices to have dtype torch.int32 or torch.int64"
            )
        weight_shape = shapes[weight]
        if len(weight_shape) != 2:
            raise CodegenBackendError("codegen embedding expects 2D weight tensor")
        if padding_idx != -1:
            if padding_idx < 0 or padding_idx >= weight_shape[0]:
                raise CodegenBackendError(
                    "codegen embedding expects padding_idx to be -1 or within num_embeddings"
                )
        indices_shape = shapes[indices]
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[weight, indices],
            output_shape=(),
            inplace_input=None,
            params={"padding_idx": padding_idx},
        )
        output_shape = _infer_output_shape(
            op_node, [weight_shape, indices_shape]
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendEmbeddingBagHandler(EmbeddingBagHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        (
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        ) = _parse_embedding_bag_args(node)
        if scale_grad_by_freq or sparse:
            raise CodegenBackendError(
                "codegen _embedding_bag supports only scale_grad_by_freq=False and sparse=False"
            )
        if per_sample_weights is not None:
            raise CodegenBackendError(
                "codegen _embedding_bag does not support per_sample_weights"
            )
        if not isinstance(weight, torch.fx.Node) or weight not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if not isinstance(indices, torch.fx.Node) or indices not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if not isinstance(offsets, torch.fx.Node) or offsets not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen _embedding_bag supports only torch.float32 tensors"
            )
        if dtypes[weight] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen _embedding_bag expects weight to match the graph dtype"
            )
        if dtypes[indices] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen _embedding_bag expects indices to have dtype torch.int32 or torch.int64"
            )
        if dtypes[offsets] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen _embedding_bag expects offsets to have dtype torch.int32 or torch.int64"
            )
        weight_shape = shapes[weight]
        if len(weight_shape) != 2:
            raise CodegenBackendError("codegen _embedding_bag expects 2D weight tensor")
        if padding_idx != -1:
            if padding_idx < 0 or padding_idx >= weight_shape[0]:
                raise CodegenBackendError(
                    "codegen _embedding_bag expects padding_idx to be -1 or within num_embeddings"
                )
        indices_shape = shapes[indices]
        if len(indices_shape) != 1:
            raise CodegenBackendError("codegen _embedding_bag expects 1D indices tensor")
        offsets_shape = shapes[offsets]
        if len(offsets_shape) != 1:
            raise CodegenBackendError("codegen _embedding_bag expects 1D offsets tensor")
        if include_last_offset and offsets_shape[0] == 0:
            raise CodegenBackendError(
                "codegen _embedding_bag expects non-empty offsets when include_last_offset is True"
            )
        if mode not in (0, 1):
            raise CodegenBackendError(
                "codegen _embedding_bag supports only mode=0 (sum) or mode=1 (mean)"
            )
        bag_count = (
            offsets_shape[0] - 1 if include_last_offset else offsets_shape[0]
        )
        if bag_count < 0:
            raise CodegenBackendError(
                "codegen _embedding_bag expects offsets to contain at least one entry"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[weight, indices, offsets],
            output_shape=(),
            inplace_input=None,
            params={
                "mode": mode,
                "padding_idx": padding_idx,
                "include_last_offset": include_last_offset,
            },
        )
        output_shape = _infer_output_shape(
            op_node, [weight_shape, indices_shape, offsets_shape]
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendConv1dHandler(Conv1dHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        (
            input_arg,
            weight_arg,
            bias,
            stride,
            padding,
            dilation,
            groups,
        ) = _parse_conv1d_args(node)
        if not isinstance(input_arg, torch.fx.Node) or not isinstance(
            weight_arg, torch.fx.Node
        ):
            raise _error_expected_tensor("conv1d")
        bias_node = None
        if bias is not None:
            if isinstance(bias, torch.fx.Node):
                bias_node = bias
            else:
                raise CodegenBackendError("codegen conv1d expects bias to be a tensor")
        if isinstance(stride, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                "codegen conv1d expects constant stride, padding, and dilation"
            )
        if isinstance(groups, torch.fx.Node):
            raise CodegenBackendError("codegen conv1d expects constant groups")
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen conv1d supports only torch.float32 tensors"
            )
        if input_arg not in shapes or weight_arg not in shapes:
            raise _error_expected_tensor("conv1d")
        if (
            dtypes[input_arg] is not torch.float32
            or dtypes[weight_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen conv1d supports only torch.float32 tensors"
            )
        if bias_node is not None:
            if bias_node not in shapes:
                raise _error_expected_tensor("conv1d")
            if dtypes[bias_node] is not torch.float32:
                raise CodegenBackendError(
                    "codegen conv1d supports only torch.float32 tensors"
                )
        input_shape = shapes[input_arg]
        weight_shape = shapes[weight_arg]
        if bias_node is not None:
            bias_shape = shapes[bias_node]
            if len(bias_shape) != 1 or bias_shape[0] != weight_shape[0]:
                raise CodegenBackendError(
                    "codegen conv1d expects bias shape to match output channels"
                )
        if len(input_shape) != 3 or len(weight_shape) != 3:
            raise CodegenBackendError(
                "codegen conv1d requires 3D input and weight tensors"
            )
        stride_value = _normalize_param(
            normalize_int_or_tuple, "stride", stride, 1
        )[0]
        dilation_value = _normalize_param(
            normalize_int_or_tuple, "dilation", dilation, 1
        )[0]
        padding_value = _normalize_param(
            normalize_padding, "padding", padding, 1, allow_strings=("same", "valid")
        )
        if not isinstance(padding_value, str):
            padding_value = padding_value[0]
        if stride_value <= 0 or dilation_value <= 0 or (
            not isinstance(padding_value, str) and padding_value < 0
        ):
            raise CodegenBackendError(
                "codegen conv1d expects stride and dilation to be positive and padding to be non-negative"
            )
        if not isinstance(groups, int) or groups <= 0:
            raise CodegenBackendError("codegen conv1d requires positive groups")
        inputs = (input_arg, weight_arg)
        if bias_node is not None:
            inputs = (*inputs, bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(inputs),
            output_shape=(),
            inplace_input=None,
            params={
                "stride": stride_value,
                "padding": padding_value,
                "dilation": dilation_value,
                "groups": groups,
                "has_bias": bias_node is not None,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape, weight_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendConv2dHandler(Conv2dHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        (
            input_arg,
            weight_arg,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = _parse_conv2d_args(node)
        if not isinstance(input_arg, torch.fx.Node) or not isinstance(
            weight_arg, torch.fx.Node
        ):
            raise _error_expected_tensor("conv2d")
        bias_node = None
        if bias is not None:
            if isinstance(bias, torch.fx.Node):
                bias_node = bias
            else:
                raise CodegenBackendError("codegen conv2d expects bias to be a tensor")
        if isinstance(stride, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                "codegen conv2d expects constant stride, padding, and dilation"
            )
        if isinstance(transposed, torch.fx.Node):
            raise CodegenBackendError("codegen conv2d expects constant transposed value")
        if isinstance(output_padding, torch.fx.Node):
            raise CodegenBackendError("codegen conv2d expects constant output_padding")
        if isinstance(groups, torch.fx.Node):
            raise CodegenBackendError("codegen conv2d expects constant groups")
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen conv2d supports only torch.float32 tensors"
            )
        if input_arg not in shapes or weight_arg not in shapes:
            raise _error_expected_tensor("conv2d")
        if (
            dtypes[input_arg] is not torch.float32
            or dtypes[weight_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen conv2d supports only torch.float32 tensors"
            )
        if bias_node is not None:
            if bias_node not in shapes:
                raise _error_expected_tensor("conv2d")
            if dtypes[bias_node] is not torch.float32:
                raise CodegenBackendError(
                    "codegen conv2d supports only torch.float32 tensors"
                )
        input_shape = shapes[input_arg]
        weight_shape = shapes[weight_arg]
        if len(weight_shape) != 4:
            raise CodegenBackendError("codegen conv2d requires 4D weight tensors")
        if bias_node is not None:
            bias_shape = shapes[bias_node]
            if len(bias_shape) != 1 or bias_shape[0] != weight_shape[0]:
                raise CodegenBackendError(
                    "codegen conv2d expects bias shape to match output channels"
                )
        if len(input_shape) != 4:
            raise CodegenBackendError("codegen conv2d requires 4D input tensors")
        stride_pair = _normalize_param(
            normalize_int_or_pair, "stride", stride
        )
        padding_pair = _normalize_param(
            normalize_padding, "padding", padding, 2, allow_strings=("same", "valid")
        )
        dilation_pair = _normalize_param(
            normalize_int_or_pair, "dilation", dilation
        )
        if isinstance(transposed, bool):
            transposed_value = transposed
        elif isinstance(transposed, numbers.Integral):
            transposed_value = bool(transposed)
        else:
            raise CodegenBackendError(
                "codegen conv2d expects transposed to be a bool"
            )
        output_padding_pair = _normalize_param(
            normalize_int_or_pair, "output_padding", output_padding
        )
        if isinstance(padding_pair, str):
            if padding_pair not in ("same", "valid"):
                raise CodegenBackendError(
                    "codegen conv2d expects padding to be 'same', 'valid', or an int tuple"
                )
        else:
            if padding_pair[0] < 0 or padding_pair[1] < 0:
                raise CodegenBackendError(
                    "codegen conv2d expects padding to be non-negative"
                )
        if stride_pair[0] <= 0 or stride_pair[1] <= 0:
            raise CodegenBackendError("codegen conv2d expects stride to be positive")
        if dilation_pair[0] <= 0 or dilation_pair[1] <= 0:
            raise CodegenBackendError("codegen conv2d expects dilation to be positive")
        if output_padding_pair[0] < 0 or output_padding_pair[1] < 0:
            raise CodegenBackendError(
                "codegen conv2d expects output_padding to be non-negative"
            )
        if groups <= 0:
            raise CodegenBackendError("codegen conv2d requires positive groups")
        inputs = (input_arg, weight_arg)
        if bias_node is not None:
            inputs = (*inputs, bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(inputs),
            output_shape=(),
            inplace_input=None,
            params={
                "stride": stride_pair,
                "padding": padding_pair,
                "dilation": dilation_pair,
                "groups": groups,
                "transposed": transposed_value,
                "output_padding": output_padding_pair,
                "has_bias": bias_node is not None,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape, weight_shape])
        op_node.output_shape = output_shape
        if bias_node is not None:
            bias_shape = shapes[bias_node]
            if len(output_shape) == 4:
                out_channels = output_shape[1]
            else:
                out_channels = output_shape[0]
            if len(bias_shape) != 1 or bias_shape[0] != out_channels:
                raise CodegenBackendError(
                    "codegen conv2d expects bias shape to match output channels"
                )
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _KindHandlerContext(HandlerContext):
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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
            node, op_spec, dtype_info, shapes, strides, dtypes
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


_HANDLER_CONTEXT = _KindHandlerContext()
_KIND_HANDLERS = build_kind_handlers(_HANDLER_CONTEXT)
_ELEMENTWISE_EMITTER = ElementwiseEmitter()
_KIND_HANDLERS.update(
    {
        OpKind.BINARY: _BackendElementwiseHandler(
            _HANDLER_CONTEXT, _ELEMENTWISE_EMITTER, "binary"
        ),
        OpKind.UNARY: _BackendElementwiseHandler(
            _HANDLER_CONTEXT, _ELEMENTWISE_EMITTER, "unary"
        ),
        OpKind.WHERE: _BackendElementwiseHandler(
            _HANDLER_CONTEXT, _ELEMENTWISE_EMITTER, "where"
        ),
        OpKind.FILL: _BackendElementwiseHandler(
            _HANDLER_CONTEXT, _ELEMENTWISE_EMITTER, "fill"
        ),
        OpKind.ARANGE: _BackendArangeHandler(_HANDLER_CONTEXT, ArangeEmitter()),
        OpKind.CONCAT: _BackendConcatHandler(_HANDLER_CONTEXT, ConcatEmitter()),
        OpKind.EMPTY_STRIDED: _BackendEmptyStridedHandler(
            _HANDLER_CONTEXT, EmptyStridedEmitter()
        ),
        OpKind.REDUCTION: _BackendReductionHandler(
            _HANDLER_CONTEXT, ReductionEmitter()
        ),
        OpKind.ARG_REDUCTION: _BackendArgReductionHandler(
            _HANDLER_CONTEXT, ArgReductionEmitter()
        ),
        OpKind.POOL1D: _BackendPool1dHandler(_HANDLER_CONTEXT, Pool1dEmitter()),
        OpKind.POOL2D: _BackendPool2dHandler(_HANDLER_CONTEXT, Pool2dEmitter()),
        OpKind.POOL3D: _BackendPool3dHandler(_HANDLER_CONTEXT, Pool3dEmitter()),
        OpKind.POOL2D_BACKWARD: _BackendPool2dBackwardHandler(
            _HANDLER_CONTEXT, Pool2dBackwardEmitter()
        ),
        OpKind.SOFTMAX: _BackendSoftmaxHandler(
            _HANDLER_CONTEXT, SoftmaxEmitter()
        ),
        OpKind.EMBEDDING: _BackendEmbeddingHandler(
            _HANDLER_CONTEXT, EmbeddingEmitter()
        ),
        OpKind.EMBEDDING_BAG: _BackendEmbeddingBagHandler(
            _HANDLER_CONTEXT, EmbeddingBagEmitter()
        ),
        OpKind.CONV1D: _BackendConv1dHandler(_HANDLER_CONTEXT, Conv1dEmitter()),
        OpKind.CONV2D: _BackendConv2dHandler(_HANDLER_CONTEXT, Conv2dEmitter()),
        OpKind.MATMUL: _BackendMatmulHandler(
            _HANDLER_CONTEXT, MatmulEmitter()
        ),
        OpKind.ADDR: _BackendAddrHandler(_HANDLER_CONTEXT, AddrEmitter()),
    }
)



def codegen_generic_backend(
    gm: torch.fx.GraphModule, example_inputs: List[object]
) -> Callable[..., torch.Tensor]:
    return _compile_graph(gm, example_inputs)
