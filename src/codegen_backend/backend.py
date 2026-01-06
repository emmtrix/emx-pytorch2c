import hashlib
import math
import numbers
import operator
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from distutils import ccompiler
from distutils import sysconfig as distutils_sysconfig
import torch
import torch.nn.functional as F
from importlib import resources
from jinja2 import Environment, FileSystemLoader
import torch.fx
from torch.fx.immutable_collections import immutable_list

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import (
    _C_TYPE_BY_DTYPE,
    _CODEGEN_DTYPES,
    _CodegenDType,
    _EMBEDDING_INDEX_DTYPES,
    _INTEGER_CODEGEN_DTYPES,
)
from codegen_backend.graph import _GenericGraph, _GenericLibrary, _OpNode
from codegen_backend.kinds import build_kind_handlers
from codegen_backend.ops_registry import SUPPORTED_OPS
from codegen_backend.param_normalize import (
    normalize_int_or_pair,
    normalize_int_or_tuple,
    normalize_padding,
)
from codegen_backend.registry import TARGET_REGISTRY
from codegen_backend.specs import _OpSpec
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

_TEMPLATE_ENV: Environment | None = None


def _get_template_env() -> Environment:
    global _TEMPLATE_ENV
    if _TEMPLATE_ENV is None:
        _TEMPLATE_ENV = Environment(
            loader=FileSystemLoader(
                resources.files("codegen_backend") / "templates"
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _TEMPLATE_ENV


def _is_out_overload(target: object) -> bool:
    schema = getattr(target, "_schema", None)
    return schema is not None and schema.overload_name == "out"


_KIND_HANDLERS = build_kind_handlers()


_LIBRARY_CACHE: Dict[str, object] = {}
_C_SRC_DIR = Path(__file__).resolve().parents[2] / "csrc"


def _format_array_suffix(shape: Sequence[int]) -> str:
    return "".join(f"[{dim}]" for dim in shape) or "[1]"


def _broadcast_output_shape(
    op_spec: _OpSpec, *input_shapes: Sequence[int]
) -> Tuple[int, ...]:
    if not input_shapes:
        return ()
    max_len = max(len(shape) for shape in input_shapes)
    output_shape = []
    for dim in range(1, max_len + 1):
        sizes = [
            shape[-dim] if dim <= len(shape) else 1 for shape in input_shapes
        ]
        max_size = max(sizes)
        if any(size not in (1, max_size) for size in sizes):
            raise RefBackendError(
                f"codegen {op_spec.name} requires inputs to be broadcastable"
            )
        output_shape.append(max_size)
    return tuple(reversed(output_shape))


def _broadcast_index_expr(
    input_shape: Sequence[int], output_shape: Sequence[int]
) -> str:
    output_rank = len(output_shape)
    input_rank = len(input_shape)
    if input_rank == 0:
        return "[0]"
    index_expr = []
    offset = output_rank - input_rank
    for input_dim in range(input_rank):
        output_dim = input_dim + offset
        if input_shape[input_dim] == 1:
            index_expr.append("[0]")
        else:
            index_expr.append(f"[i{output_dim}]")
    return "".join(index_expr)


def _contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    if not shape:
        return ()
    strides = [0] * len(shape)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = stride
        stride *= max(shape[dim], 1)
    return tuple(strides)


def _is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    expected = _contiguous_strides(shape)
    return all(
        size == 1 or stride == expected_stride
        for size, stride, expected_stride in zip(shape, strides, expected)
    )


def _unpack_conv2d_input_shape(
    input_shape: Sequence[int],
) -> Tuple[bool, int, int, int, int]:
    if len(input_shape) == 4:
        batch, in_channels, in_h, in_w = input_shape
        return True, batch, in_channels, in_h, in_w
    if len(input_shape) == 3:
        in_channels, in_h, in_w = input_shape
        return False, 1, in_channels, in_h, in_w
    raise RefBackendError("codegen conv2d requires 3D or 4D input tensors")


def _conv2d_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> Tuple[int, ...]:
    has_batch, batch, in_channels, in_h, in_w = _unpack_conv2d_input_shape(
        input_shape
    )
    out_channels, weight_in_channels, kernel_h, kernel_w = weight_shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "codegen conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "codegen conv2d requires output channels to be divisible by groups"
        )
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    numerator_h = in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1
    numerator_w = in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1
    if numerator_h < 0 or numerator_w < 0:
        raise RefBackendError(
            "codegen conv2d requires output shape (N, C_out, H_out, W_out)"
        )
    out_h = numerator_h // stride_h + 1
    out_w = numerator_w // stride_w + 1
    if has_batch:
        return batch, out_channels, out_h, out_w
    return out_channels, out_h, out_w


def _conv2d_same_padding(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    _, _, in_h, in_w = _unpack_conv2d_input_shape(input_shape)[1:]
    _, _, kernel_h, kernel_w = weight_shape
    stride_h, stride_w = stride
    dil_h, dil_w = dilation
    out_h = math.ceil(in_h / stride_h)
    out_w = math.ceil(in_w / stride_w)
    pad_h = max(
        (out_h - 1) * stride_h + (dil_h * (kernel_h - 1) + 1) - in_h,
        0,
    )
    pad_w = max(
        (out_w - 1) * stride_w + (dil_w * (kernel_w - 1) + 1) - in_w,
        0,
    )
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return (pad_top, pad_left), (out_h, out_w)


def _conv2d_validate_channels(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    groups: int,
) -> Tuple[bool, int]:
    has_batch, _, in_channels, _, _ = _unpack_conv2d_input_shape(input_shape)
    out_channels, weight_in_channels, _, _ = weight_shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "codegen conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "codegen conv2d requires output channels to be divisible by groups"
        )
    return has_batch, out_channels


def _normalize_col2im_output_size(
    op_name: str, value: object
) -> Tuple[int, int]:
    if isinstance(value, torch.Size):
        value = tuple(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RefBackendError(
            f"codegen {op_name} expects output_size to be a tuple of two ints"
        )
    try:
        return normalize_int_or_tuple("output_size", value, 2)
    except ValueError as exc:
        raise RefBackendError(
            f"codegen {op_name} expects output_size to be a tuple of ints"
        ) from exc


def _pool1d_output_shape_from_shapes(
    input_shape: Sequence[int],
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> Tuple[int, int, int]:
    batch, channels, in_l = input_shape
    numerator = in_l + 2 * padding - dilation * (kernel_size - 1) - 1
    if numerator < 0:
        raise RefBackendError(
            "codegen pool1d requires output shape (N, C, L_out)"
        )
    out_l = numerator // stride + 1
    return batch, channels, out_l


def _pool2d_output_shape_from_shapes(
    input_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    batch, channels, in_h, in_w = input_shape
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    numerator_h = in_h + 2 * pad_h - dil_h * (k_h - 1) - 1
    numerator_w = in_w + 2 * pad_w - dil_w * (k_w - 1) - 1
    if numerator_h < 0 or numerator_w < 0:
        raise RefBackendError(
            "codegen pool2d requires output shape (N, C, H_out, W_out)"
        )
    out_h = numerator_h // stride_h + 1
    out_w = numerator_w // stride_w + 1
    return batch, channels, out_h, out_w


def _conv1d_validate_channels(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    groups: int,
) -> Tuple[int, int]:
    batch, in_channels, _ = input_shape
    out_channels, weight_in_channels, _ = weight_shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "codegen conv1d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "codegen conv1d requires output channels to be divisible by groups"
        )
    return batch, out_channels


def _conv1d_same_padding(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: int,
    dilation: int,
) -> Tuple[int, int]:
    _, _, in_l = input_shape
    _, _, kernel_l = weight_shape
    out_l = math.ceil(in_l / stride)
    pad_l = max(
        (out_l - 1) * stride + (dilation * (kernel_l - 1) + 1) - in_l,
        0,
    )
    pad_left = pad_l // 2
    return pad_left, out_l


def _conv1d_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tuple[int, int, int]:
    batch, out_channels = _conv1d_validate_channels(
        input_shape, weight_shape, groups
    )
    in_l = input_shape[2]
    kernel_l = weight_shape[2]
    numerator = in_l + 2 * padding - dilation * (kernel_l - 1) - 1
    if numerator < 0:
        raise RefBackendError(
            "codegen conv1d requires output shape (N, C_out, L_out)"
        )
    out_l = numerator // stride + 1
    return batch, out_channels, out_l


def _emit_strided_access(
    name: str,
    indices: Sequence[str],
    strides: Sequence[int],
    contig: bool,
    sizes: Optional[Sequence[int]] = None,
    *,
    c_type: str = "float",
) -> str:
    if contig:
        return f"{name}{''.join(f'[{idx}]' for idx in indices)}"
    terms = []
    for idx_name, stride, size in zip(
        indices, strides, sizes or [None] * len(indices)
    ):
        if size == 1:
            continue
        terms.append(f"{idx_name} * {stride}")
    index_expr = " + ".join(terms) if terms else "0"
    return f"(({c_type}*){name})[{index_expr}]"


def _format_strided_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    c_type: str = "float",
) -> str:
    output_rank = len(output_shape)
    input_rank = len(input_shape)
    if input_rank == 0:
        return f"(({c_type}*){name})[0]"
    offset = output_rank - input_rank
    indices = [f"i{input_dim + offset}" for input_dim in range(input_rank)]
    return _emit_strided_access(
        name,
        indices,
        input_strides,
        contig=False,
        sizes=input_shape,
        c_type=c_type,
    )


def _format_output_access(
    name: str,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    *,
    c_type: str = "float",
) -> str:
    if not output_shape:
        return f"(({c_type}*){name})[0]"
    terms = []
    for dim, stride in enumerate(output_strides):
        if output_shape[dim] == 1:
            continue
        terms.append(f"i{dim} * {stride}")
    index_expr = " + ".join(terms) if terms else "0"
    return f"(({c_type}*){name})[{index_expr}]"


def _input_c_type(dtype: torch.dtype, graph_dtype: _CodegenDType) -> str:
    if dtype is graph_dtype.torch_dtype:
        return graph_dtype.c_type
    if dtype is torch.bool:
        return _C_TYPE_BY_DTYPE[torch.bool]
    if dtype in _EMBEDDING_INDEX_DTYPES:
        return _C_TYPE_BY_DTYPE[dtype]
    raise RefBackendError(
        "codegen backend supports only torch.float32, torch.int8, torch.int32, torch.int64, or torch.bool tensors"
    )


def _dtype_to_c_type(dtype: torch.dtype, graph_dtype: _CodegenDType) -> str:
    if dtype is graph_dtype.torch_dtype:
        return graph_dtype.c_type
    c_type = _C_TYPE_BY_DTYPE.get(dtype)
    if c_type is not None:
        return c_type
    raise RefBackendError(
        "codegen backend supports only torch.float32, torch.int8, torch.int32, torch.int64, or torch.bool tensors"
    )


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in _INTEGER_CODEGEN_DTYPES


def _format_scalar_literal(value: float, dtype: _CodegenDType) -> str:
    if dtype.torch_dtype is torch.bool:
        return str(int(value))
    if _is_integer_dtype(dtype.torch_dtype):
        return str(int(value))
    if dtype.torch_dtype is torch.float32:
        return f"{float(value)}f"
    raise RefBackendError(
        "codegen addmm-like ops support only floating point or integer tensors"
    )


def _normalize_scalar_value(op_name: str, value: object) -> float | int | bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RefBackendError(
                f"codegen {op_name} expects a scalar value"
            )
        value = value.item()
    if isinstance(value, numbers.Number):
        return value
    raise RefBackendError(f"codegen {op_name} expects a scalar value")


def _resolve_scalar_arg(
    op_name: str,
    value: object,
    scalar_values: Dict[torch.fx.Node, object],
) -> float | int | bool:
    if isinstance(value, torch.fx.Node):
        if value in scalar_values:
            return _normalize_scalar_value(op_name, scalar_values[value])
        raise RefBackendError(f"codegen {op_name} expects a scalar value")
    return _normalize_scalar_value(op_name, value)


def emit_signature(
    node_index: int,
    op_spec: _OpSpec,
    output_shape: Sequence[int],
    input_shapes: Sequence[Sequence[int]],
    input_dtypes: Sequence[torch.dtype],
    dtype: _CodegenDType,
    params: Dict[str, object] | None = None,
) -> str:
    out_suffix = _format_array_suffix(output_shape)
    if op_spec.kind == "binary":
        if len(input_shapes) == 1:
            a_shape = input_shapes[0]
            a_suffix = _format_array_suffix(a_shape)
            a_c_type = _input_c_type(input_dtypes[0], dtype)
            return (
                f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
                f"const {a_c_type} a{a_suffix}, "
                f"{dtype.c_type} out{out_suffix}) {{"
            )
        a_shape, b_shape = input_shapes
        a_suffix = _format_array_suffix(a_shape)
        b_suffix = _format_array_suffix(b_shape)
        a_c_type = _input_c_type(input_dtypes[0], dtype)
        b_c_type = _input_c_type(input_dtypes[1], dtype)
        return (
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {a_c_type} a{a_suffix}, "
            f"const {b_c_type} b{b_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        )
    if op_spec.kind == "where":
        params = params or {}
        input_index = 0
        cond_shape = input_shapes[input_index]
        cond_suffix = _format_array_suffix(cond_shape)
        cond_c_type = _input_c_type(input_dtypes[input_index], dtype)
        input_index += 1
        signature_parts = [
            f"const {cond_c_type} cond{cond_suffix}",
        ]
        if "a_scalar" not in params:
            a_shape = input_shapes[input_index]
            a_suffix = _format_array_suffix(a_shape)
            a_c_type = _input_c_type(input_dtypes[input_index], dtype)
            signature_parts.append(f"const {a_c_type} a{a_suffix}")
            input_index += 1
        if "b_scalar" not in params:
            b_shape = input_shapes[input_index]
            b_suffix = _format_array_suffix(b_shape)
            b_c_type = _input_c_type(input_dtypes[input_index], dtype)
            signature_parts.append(f"const {b_c_type} b{b_suffix}")
            input_index += 1
        signature_parts.append(f"{dtype.c_type} out{out_suffix}")
        return (
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"{', '.join(signature_parts)}) {{"
        )
    a_suffix = _format_array_suffix(input_shapes[0])
    a_c_type = _input_c_type(input_dtypes[0], dtype)
    return (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {a_c_type} a{a_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )


def emit_loops(output_shape: Sequence[int]) -> Tuple[List[str], str]:
    lines: List[str] = []
    indent = "    "
    if output_shape:
        for dim, size in enumerate(output_shape):
            lines.append(
                f"{indent}for (int64_t i{dim} = 0; i{dim} < {size}; ++i{dim}) {{"
            )
            indent += "    "
    return lines, indent


def _close_loops(loop_count: int, indent: str) -> Tuple[List[str], str]:
    lines: List[str] = []
    for _ in range(loop_count):
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    return lines, indent


def emit_output_access(
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    *,
    c_type: str,
) -> str:
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    if output_is_contiguous:
        output_access = (
            "".join(f"[i{dim}]" for dim in range(len(output_shape))) or "[0]"
        )
        return f"out{output_access}"
    return _format_output_access("out", output_shape, output_strides, c_type=c_type)


def emit_input_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    broadcast_contiguous: bool,
    c_type: str,
) -> str:
    if _is_contiguous(input_shape, input_strides):
        if broadcast_contiguous:
            return f"{name}{_broadcast_index_expr(input_shape, output_shape)}"
        return (
            f"{name}{''.join(f'[i{dim}]' for dim in range(len(output_shape))) or '[0]'}"
        )
    return _format_strided_access(
        name, input_shape, input_strides, output_shape, c_type=c_type
    )


def _emit_flip_input_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    flip_dims: Sequence[int],
    *,
    c_type: str,
) -> str:
    if not input_shape:
        return f"(({c_type}*){name})[0]"
    indices = []
    flip_dim_set = set(flip_dims)
    for dim, size in enumerate(input_shape):
        if dim in flip_dim_set:
            indices.append(f"({size - 1} - i{dim})")
        else:
            indices.append(f"i{dim}")
    return _emit_strided_access(
        name,
        indices,
        input_strides,
        contig=_is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=c_type,
    )
def _format_diagonal_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    dim1: int,
    dim2: int,
    offset: int,
    c_type: str,
) -> str:
    output_rank = len(output_shape)
    diag_index = f"i{output_rank - 1}"
    other_indices = [f"i{idx}" for idx in range(output_rank - 1)]
    other_iter = iter(other_indices)
    terms = []
    for dim, stride in enumerate(input_strides):
        if stride == 0:
            continue
        if dim == dim1:
            if offset >= 0:
                index_expr = diag_index
            else:
                index_expr = f"({diag_index} - ({offset}))"
        elif dim == dim2:
            if offset >= 0:
                index_expr = f"({diag_index} + {offset})"
            else:
                index_expr = diag_index
        else:
            index_expr = next(other_iter)
        terms.append(f"{index_expr} * {stride}")
    index_expr = " + ".join(terms) if terms else "0"
    return f"(({c_type}*){name})[{index_expr}]"


def emit_footer(output_shape: Sequence[int], indent: str) -> List[str]:
    lines, _ = _close_loops(len(output_shape), indent)
    lines.append("}")
    return lines


def _write_elementwise_kernel(
    node_index: int,
    op_node: _OpNode,
    output_shape: Sequence[int],
    input_shapes: Sequence[Sequence[int]],
    input_strides: Sequence[Sequence[int]],
    input_dtypes: Sequence[torch.dtype],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    op_spec = op_node.spec
    elementwise_template = _get_template_env().get_template(
        "elementwise_kernel.c.j2"
    )
    params = op_node.params
    signature = emit_signature(
        node_index,
        op_spec,
        output_shape,
        input_shapes,
        input_dtypes,
        dtype,
        params,
    )
    output_dims = [
        {"dim": dim, "size": size}
        for dim, size in enumerate(output_shape)
    ]
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    scalar_fn = f"{dtype.scalar_prefix}{op_spec.name}"
    context: Dict[str, object] = {
        "signature": signature,
        "output_dims": output_dims,
        "op_kind": op_spec.kind,
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
    if op_spec.kind == "binary":
        if "scalar" in params:
            a_shape = input_shapes[0]
            a_strides = input_strides[0]
            context["a_access"] = emit_input_access(
                "a",
                a_shape,
                a_strides,
                output_shape,
                broadcast_contiguous=False,
                c_type=_input_c_type(input_dtypes[0], dtype),
            )
            context["scalar_literal"] = _format_scalar_literal(
                params["scalar"], dtype
            )
        else:
            a_shape, b_shape = input_shapes
            a_strides, b_strides = input_strides
            context["a_access"] = emit_input_access(
                "a",
                a_shape,
                a_strides,
                output_shape,
                broadcast_contiguous=True,
                c_type=_input_c_type(input_dtypes[0], dtype),
            )
            context["b_access"] = emit_input_access(
                "b",
                b_shape,
                b_strides,
                output_shape,
                broadcast_contiguous=True,
                c_type=_input_c_type(input_dtypes[1], dtype),
            )
    elif op_spec.kind == "where":
        input_index = 0
        cond_shape = input_shapes[input_index]
        cond_strides = input_strides[input_index]
        context["cond_access"] = emit_input_access(
            "cond",
            cond_shape,
            cond_strides,
            output_shape,
            broadcast_contiguous=True,
            c_type=_input_c_type(input_dtypes[input_index], dtype),
        )
        input_index += 1
        if "a_scalar" in params:
            context["a_access"] = _format_scalar_literal(
                params["a_scalar"], dtype
            )
        else:
            a_shape = input_shapes[input_index]
            a_strides = input_strides[input_index]
            context["a_access"] = emit_input_access(
                "a",
                a_shape,
                a_strides,
                output_shape,
                broadcast_contiguous=True,
                c_type=_input_c_type(input_dtypes[input_index], dtype),
            )
            input_index += 1
        if "b_scalar" in params:
            context["b_access"] = _format_scalar_literal(
                params["b_scalar"], dtype
            )
        else:
            b_shape = input_shapes[input_index]
            b_strides = input_strides[input_index]
            context["b_access"] = emit_input_access(
                "b",
                b_shape,
                b_strides,
                output_shape,
                broadcast_contiguous=True,
                c_type=_input_c_type(input_dtypes[input_index], dtype),
            )
    elif op_spec.kind == "fill":
        context["fill_value"] = _format_scalar_literal(
            op_node.p("value"), dtype
        )
    else:
        a_shape = input_shapes[0]
        a_strides = input_strides[0]
        context["input_access"] = emit_input_access(
            "a",
            a_shape,
            a_strides,
            output_shape,
            broadcast_contiguous=False,
            c_type=_input_c_type(input_dtypes[0], dtype),
        )
        if op_spec.name in _PARAMETRIC_UNARY_OPS:
            if dtype.torch_dtype is not torch.float32:
                raise RefBackendError(
                    f"codegen {op_spec.name} supports only torch.float32 tensors"
                )
            context.update(
                {
                    "one": _format_scalar_literal(1.0, dtype),
                    "half": _format_scalar_literal(0.5, dtype),
                    "gelu_approximate": params.get(
                        "approximate", "none"
                    ),
                    "sqrt_2_over_pi": _format_scalar_literal(
                        0.7978845608028654, dtype
                    ),
                    "coeff": _format_scalar_literal(0.044715, dtype),
                    "inv_sqrt2": _format_scalar_literal(
                        0.7071067811865475, dtype
                    ),
                    "alpha": _format_scalar_literal(
                        params.get("alpha", 1.0), dtype
                    ),
                    "scale": _format_scalar_literal(
                        params.get("scale", 1.0), dtype
                    ),
                    "input_scale": _format_scalar_literal(
                        params.get("input_scale", 1.0), dtype
                    ),
                    "negative_slope": _format_scalar_literal(
                        params.get("negative_slope", 0.01), dtype
                    ),
                    "beta": _format_scalar_literal(
                        params.get("beta", 1.0), dtype
                    ),
                    "threshold": _format_scalar_literal(
                        params.get("threshold", 20.0), dtype
                    ),
                    "clamp_min": (
                        _format_scalar_literal(params.get("min_val"), dtype)
                        if params.get("min_val") is not None
                        else None
                    ),
                    "clamp_max": (
                        _format_scalar_literal(params.get("max_val"), dtype)
                        if params.get("max_val") is not None
                        else None
                    ),
                    "clamp_has_min": params.get("min_val") is not None,
                    "clamp_has_max": params.get("max_val") is not None,
                    "min_val": _format_scalar_literal(
                        params.get("min_val", -1.0), dtype
                    ),
                    "max_val": _format_scalar_literal(
                        params.get("max_val", 1.0), dtype
                    ),
                }
            )
    rendered = elementwise_template.render(**context)
    return rendered.strip().splitlines()


def _write_arange_kernel(
    node_index: int,
    op_node: _OpNode,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    out_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_node.spec.name}_{dtype.suffix}("
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    lines = [signature]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    start = _format_scalar_literal(op_node.p("start"), dtype)
    step = _format_scalar_literal(op_node.p("step"), dtype)
    index_expr = "i0" if output_shape else "0"
    lines.append(
        f"{indent}{output_access} = {start} + ({step} * {index_expr});"
    )
    lines.extend(emit_footer(output_shape, indent))
    return lines


def _write_flip_kernel(
    node_index: int,
    op_node: _OpNode,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    input_dtype: torch.dtype,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    lines = [
        emit_signature(
            node_index,
            op_node.spec,
            output_shape,
            [input_shape],
            [input_dtype],
            dtype,
        )
    ]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    input_access = _emit_flip_input_access(
        "a",
        input_shape,
        input_strides,
        op_node.p("dims", ()),
        c_type=_input_c_type(input_dtype, dtype),
    )
    lines.append(f"{indent}{output_access} = {input_access};")
    lines.extend(emit_footer(output_shape, indent))
    return lines


def _write_constant_pad_kernel(
    node_index: int,
    op_node: _OpNode,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    input_dtype: torch.dtype,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    pad_template = _get_template_env().get_template(
        "constant_pad_nd_kernel.c.j2"
    )
    signature = emit_signature(
        node_index,
        op_node.spec,
        output_shape,
        [input_shape],
        [input_dtype],
        dtype,
    )
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if not input_shape:
        input_access = _format_strided_access(
            "a",
            input_shape,
            input_strides,
            output_shape,
            c_type=_input_c_type(input_dtype, dtype),
        )
        rendered = pad_template.render(
            signature=signature,
            output_access=output_access,
            input_access=input_access,
            has_input_shape=False,
        )
        return rendered.strip().splitlines()
    pad_before = op_node.p("pad_before", ())
    input_access = _emit_strided_access(
        "a",
        [f"in_{dim}" for dim in range(len(input_shape))],
        input_strides,
        contig=_is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=_input_c_type(input_dtype, dtype),
    )
    rendered = pad_template.render(
        signature=signature,
        output_access=output_access,
        input_access=input_access,
        has_input_shape=True,
        output_shape=output_shape,
        input_shape=input_shape,
        pad_before=pad_before,
        value=_format_scalar_literal(op_node.p("value"), dtype),
    )
    return rendered.strip().splitlines()


def _write_view_kernel(
    node_index: int,
    op_node: _OpNode,
    input_shape: Sequence[int],
    input_dtype: torch.dtype,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_node.spec.name}_{dtype.suffix}("
        f"const {input_c_type} a{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    lines = [signature]
    lines.append(f"    const {input_c_type}* a_ptr = (const {input_c_type}*)a;")
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    view_strides = op_node.p("view_strides", ())
    storage_offset = int(op_node.p("storage_offset", 0))
    if view_strides:
        offset_terms = [
            f"i{dim} * {stride}"
            for dim, stride in enumerate(view_strides)
        ]
        offset_expr = " + ".join(offset_terms)
    else:
        offset_expr = "0"
    if storage_offset:
        offset_expr = f"{storage_offset} + {offset_expr}"
    lines.append(f"{indent}int64_t offset = {offset_expr};")
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    lines.append(f"{indent}{output_access} = a_ptr[offset];")
    lines.extend(emit_footer(output_shape, indent))
    return lines


def _write_empty_strided_kernel(
    node_index: int,
    op_spec: _OpSpec,
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    return [signature, "    (void)out;", "}"]


def _write_matmul_kernel(
    node_index: int,
    op_spec: _OpSpec,
    a_shape: Sequence[int],
    b_shape: Sequence[int],
    a_strides: Sequence[int],
    b_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    matmul_template = _get_template_env().get_template("matmul_kernel.c.j2")
    a_is_contiguous = _is_contiguous(a_shape, a_strides)
    b_is_contiguous = _is_contiguous(b_shape, b_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"

    if op_spec.name == "matmul":
        if len(a_shape) == 1:
            k = a_shape[0]
            a_suffix = _format_array_suffix((k,))
            b_suffix = _format_array_suffix((k,))
            out_suffix = _format_array_suffix(())
            rendered = matmul_template.render(
                signature=(
                    f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
                    f"const {dtype.c_type} a{a_suffix}, "
                    f"const {dtype.c_type} b{b_suffix}, "
                    f"{dtype.c_type} out{out_suffix}) {{"
                ),
                batch=None,
                m=1,
                n=1,
                k=k,
                acc_type=acc_type,
                acc_init=acc_init,
                a_access=_emit_strided_access(
                    "a",
                    ("t",),
                    a_strides,
                    a_is_contiguous,
                    sizes=a_shape,
                    c_type=dtype.c_type,
                ),
                b_access=_emit_strided_access(
                    "b",
                    ("t",),
                    b_strides,
                    b_is_contiguous,
                    sizes=b_shape,
                    c_type=dtype.c_type,
                ),
                out_access="out[0]",
            )
            return rendered.strip().splitlines()
        m, k = a_shape
        _, n = b_shape
        a_suffix = _format_array_suffix((m, k))
        b_suffix = _format_array_suffix((k, n))
        out_suffix = _format_array_suffix((m, n))
        rendered = matmul_template.render(
            signature=(
                f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
                f"const {dtype.c_type} a{a_suffix}, "
                f"const {dtype.c_type} b{b_suffix}, "
                f"{dtype.c_type} out{out_suffix}) {{"
            ),
            batch=None,
            m=m,
            n=n,
            k=k,
            acc_type=acc_type,
            acc_init=acc_init,
            a_access=_emit_strided_access(
                "a",
                ("i", "t"),
                a_strides,
                a_is_contiguous,
                sizes=a_shape,
                c_type=dtype.c_type,
            ),
            b_access=_emit_strided_access(
                "b",
                ("t", "j"),
                b_strides,
                b_is_contiguous,
                sizes=b_shape,
                c_type=dtype.c_type,
            ),
            out_access="out[i][j]",
        )
        return rendered.strip().splitlines()
    batch, m, k = a_shape
    _, _, n = b_shape
    a_suffix = _format_array_suffix((batch, m, k))
    b_suffix = _format_array_suffix((batch, k, n))
    out_suffix = _format_array_suffix((batch, m, n))
    rendered = matmul_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} a{a_suffix}, "
            f"const {dtype.c_type} b{b_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        batch=batch,
        m=m,
        n=n,
        k=k,
        acc_type=acc_type,
        acc_init=acc_init,
        a_access=_emit_strided_access(
            "a",
            ("b_idx", "i", "t"),
            a_strides,
            a_is_contiguous,
            sizes=a_shape,
            c_type=dtype.c_type,
        ),
        b_access=_emit_strided_access(
            "b",
            ("b_idx", "t", "j"),
            b_strides,
            b_is_contiguous,
            sizes=b_shape,
            c_type=dtype.c_type,
        ),
        out_access="out[b_idx][i][j]",
    )
    return rendered.strip().splitlines()


def _write_addmm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    mat1_shape: Sequence[int],
    mat2_shape: Sequence[int],
    input_strides: Sequence[int],
    mat1_strides: Sequence[int],
    mat2_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addmm_template = _get_template_env().get_template("addmm_kernel.c.j2")
    mat1_is_contiguous = _is_contiguous(mat1_shape, mat1_strides)
    mat2_is_contiguous = _is_contiguous(mat2_shape, mat2_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    m, k = mat1_shape
    _, n = mat2_shape
    input_suffix = _format_array_suffix(input_shape)
    mat1_suffix = _format_array_suffix(mat1_shape)
    mat2_suffix = _format_array_suffix(mat2_shape)
    out_suffix = _format_array_suffix(output_shape)
    output_indices = ("i", "j")
    offset = len(output_indices) - len(input_shape)
    input_indices = output_indices[offset:] if input_shape else ()
    rendered = addmm_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} mat1{mat1_suffix}, "
            f"const {dtype.c_type} mat2{mat2_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        m=m,
        n=n,
        k=k,
        acc_type=acc_type,
        acc_init=acc_init,
        input_access=_emit_strided_access(
            "input",
            input_indices,
            input_strides,
            False,
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
        mat1_access=_emit_strided_access(
            "mat1",
            ("i", "t"),
            mat1_strides,
            mat1_is_contiguous,
            sizes=mat1_shape,
            c_type=dtype.c_type,
        ),
        mat2_access=_emit_strided_access(
            "mat2",
            ("t", "j"),
            mat2_strides,
            mat2_is_contiguous,
            sizes=mat2_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i", "j"),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
    )
    return rendered.strip().splitlines()


def _write_addbmm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    batch1_shape: Sequence[int],
    batch2_shape: Sequence[int],
    input_strides: Sequence[int],
    batch1_strides: Sequence[int],
    batch2_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addbmm_template = _get_template_env().get_template("addbmm_kernel.c.j2")
    batch1_is_contiguous = _is_contiguous(batch1_shape, batch1_strides)
    batch2_is_contiguous = _is_contiguous(batch2_shape, batch2_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    batch, m, k = batch1_shape
    _, _, n = batch2_shape
    input_suffix = _format_array_suffix(input_shape)
    batch1_suffix = _format_array_suffix(batch1_shape)
    batch2_suffix = _format_array_suffix(batch2_shape)
    out_suffix = _format_array_suffix(output_shape)
    output_indices = ("i", "j")
    offset = len(output_indices) - len(input_shape)
    input_indices = output_indices[offset:] if input_shape else ()
    rendered = addbmm_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} batch1{batch1_suffix}, "
            f"const {dtype.c_type} batch2{batch2_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        batch=batch,
        m=m,
        n=n,
        k=k,
        acc_type=acc_type,
        acc_init=acc_init,
        input_access=_emit_strided_access(
            "input",
            input_indices,
            input_strides,
            False,
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
        batch1_access=_emit_strided_access(
            "batch1",
            ("b_idx", "i", "t"),
            batch1_strides,
            batch1_is_contiguous,
            sizes=batch1_shape,
            c_type=dtype.c_type,
        ),
        batch2_access=_emit_strided_access(
            "batch2",
            ("b_idx", "t", "j"),
            batch2_strides,
            batch2_is_contiguous,
            sizes=batch2_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i", "j"),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
    )
    return rendered.strip().splitlines()


def _write_addmv_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    mat_shape: Sequence[int],
    vec_shape: Sequence[int],
    input_strides: Sequence[int],
    mat_strides: Sequence[int],
    vec_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addmv_template = _get_template_env().get_template("addmv_kernel.c.j2")
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    mat_is_contiguous = _is_contiguous(mat_shape, mat_strides)
    vec_is_contiguous = _is_contiguous(vec_shape, vec_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    m, n = mat_shape
    input_suffix = _format_array_suffix(input_shape)
    mat_suffix = _format_array_suffix(mat_shape)
    vec_suffix = _format_array_suffix(vec_shape)
    out_suffix = _format_array_suffix(output_shape)
    broadcast_input = input_shape != output_shape
    input_access = _emit_strided_access(
        "input",
        ("i",),
        input_strides,
        contig=input_is_contiguous and not broadcast_input,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    rendered = addmv_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} mat{mat_suffix}, "
            f"const {dtype.c_type} vec{vec_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        m=m,
        n=n,
        acc_type=acc_type,
        acc_init=acc_init,
        input_access=input_access,
        mat_access=_emit_strided_access(
            "mat",
            ("i", "t"),
            mat_strides,
            mat_is_contiguous,
            sizes=mat_shape,
            c_type=dtype.c_type,
        ),
        vec_access=_emit_strided_access(
            "vec",
            ("t",),
            vec_strides,
            vec_is_contiguous,
            sizes=vec_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i",),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
    )
    return rendered.strip().splitlines()


def _write_addr_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    vec1_shape: Sequence[int],
    vec2_shape: Sequence[int],
    input_strides: Sequence[int],
    vec1_strides: Sequence[int],
    vec2_strides: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    alpha: float,
    beta: float,
) -> List[str]:
    addr_template = _get_template_env().get_template("addr_kernel.c.j2")
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    vec1_is_contiguous = _is_contiguous(vec1_shape, vec1_strides)
    vec2_is_contiguous = _is_contiguous(vec2_shape, vec2_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    m, n = output_shape
    input_suffix = _format_array_suffix(input_shape)
    vec1_suffix = _format_array_suffix(vec1_shape)
    vec2_suffix = _format_array_suffix(vec2_shape)
    out_suffix = _format_array_suffix(output_shape)
    skip_input = math.isclose(beta, 0.0)
    if not input_shape:
        input_access = f"(({dtype.c_type}*)input)[0]"
    else:
        input_rank = len(input_shape)
        input_indices = ("i", "j") if input_rank == 2 else ("j",)
        use_contig = input_is_contiguous and input_shape == output_shape
        input_access = _emit_strided_access(
            "input",
            input_indices,
            input_strides,
            use_contig,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    rendered = addr_template.render(
        signature=(
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {dtype.c_type} input{input_suffix}, "
            f"const {dtype.c_type} vec1{vec1_suffix}, "
            f"const {dtype.c_type} vec2{vec2_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
        ),
        m=m,
        n=n,
        input_access=input_access,
        vec1_access=_emit_strided_access(
            "vec1",
            ("i",),
            vec1_strides,
            vec1_is_contiguous,
            sizes=vec1_shape,
            c_type=dtype.c_type,
        ),
        vec2_access=_emit_strided_access(
            "vec2",
            ("j",),
            vec2_strides,
            vec2_is_contiguous,
            sizes=vec2_shape,
            c_type=dtype.c_type,
        ),
        out_access=_emit_strided_access(
            "out",
            ("i", "j"),
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
        skip_input=skip_input,
    )
    return rendered.strip().splitlines()


_REDUCTION_CONFIG = {
    "sum": {
        "init_value": 0,
        "reduce_op": "+=",
        "post_op": None,
    },
    "prod": {
        "init_value": 1,
        "reduce_op": "*=",
        "post_op": None,
    },
    "mean": {
        "init_value": 0,
        "reduce_op": "+=",
        "post_op": "mean",
    },
    "any": {
        "init_value": 0,
        "reduce_op": "|=",
        "post_op": None,
        "bool_reduction": True,
    },
    "all": {
        "init_value": 1,
        "reduce_op": "&=",
        "post_op": None,
        "bool_reduction": True,
    },
    "amax": {
        "init_value": 0,
        "reduce_op": None,
        "post_op": None,
    },
    "amin": {
        "init_value": 0,
        "reduce_op": None,
        "post_op": None,
    },
}

_MINMAX_INIT_VALUES = {
    torch.float32: {
        "amax": "-INFINITY",
        "amin": "INFINITY",
    },
    torch.int8: {
        "amax": "INT8_MIN",
        "amin": "INT8_MAX",
    },
    torch.int32: {
        "amax": "INT32_MIN",
        "amin": "INT32_MAX",
    },
}


def _write_std_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
    *,
    unbiased: bool,
) -> List[str]:
    std_template = _get_template_env().get_template("std_kernel.c.j2")
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    reduction_set = set(reduction_dims)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        if keepdim:
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim}"
                for dim in range(input_rank)
            ]
        else:
            dim_to_output: Dict[int, int] = {}
            output_idx = 0
            for dim in range(input_rank):
                if dim in reduction_set:
                    continue
                dim_to_output[dim] = output_idx
                output_idx += 1
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim_to_output[dim]}"
                for dim in range(input_rank)
            ]
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    reduction_count = 1
    for dim in reduction_dims:
        reduction_count *= input_shape[dim]
    acc_type = dtype.c_type
    sqrt_fn = f"{dtype.scalar_prefix}sqrt"
    if dtype.torch_dtype is torch.bool or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        acc_type = "float"
        sqrt_fn = "ref_scalar_f32_sqrt"
    rendered = std_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        output_access=output_access,
        acc_type=acc_type,
        reduction_count=reduction_count,
        unbiased=int(unbiased),
        sqrt_fn=sqrt_fn,
    )
    return rendered.strip().splitlines()


def _write_var_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
    *,
    unbiased: bool,
) -> List[str]:
    var_template = _get_template_env().get_template("var_kernel.c.j2")
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    reduction_set = set(reduction_dims)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        if keepdim:
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim}"
                for dim in range(input_rank)
            ]
        else:
            dim_to_output: Dict[int, int] = {}
            output_idx = 0
            for dim in range(input_rank):
                if dim in reduction_set:
                    continue
                dim_to_output[dim] = output_idx
                output_idx += 1
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim_to_output[dim]}"
                for dim in range(input_rank)
            ]
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    reduction_count = 1
    for dim in reduction_dims:
        reduction_count *= input_shape[dim]
    acc_type = dtype.c_type
    if dtype.torch_dtype is torch.bool or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        acc_type = "float"
    rendered = var_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        output_access=output_access,
        acc_type=acc_type,
        reduction_count=reduction_count,
        unbiased=int(unbiased),
    )
    return rendered.strip().splitlines()


def _write_norm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
    *,
    p_value: float,
) -> List[str]:
    norm_template = _get_template_env().get_template("norm_kernel.c.j2")
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    reduction_set = set(reduction_dims)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        if keepdim:
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim}"
                for dim in range(input_rank)
            ]
        else:
            dim_to_output: Dict[int, int] = {}
            output_idx = 0
            for dim in range(input_rank):
                if dim in reduction_set:
                    continue
                dim_to_output[dim] = output_idx
                output_idx += 1
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim_to_output[dim]}"
                for dim in range(input_rank)
            ]
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    acc_type = dtype.c_type
    abs_fn = f"{dtype.scalar_prefix}abs"
    pow_fn = f"{dtype.scalar_prefix}pow"
    is_zero_p = math.isclose(p_value, 0.0)
    if dtype.torch_dtype is torch.bool or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        acc_type = "float"
        abs_fn = "ref_scalar_f32_abs"
        pow_fn = "ref_scalar_f32_pow"
        input_access = f"(float){input_access}"
    p_literal = f"{float(p_value)}f"
    inv_p_literal = f"{(1.0 / p_value)}f" if not is_zero_p else "0.0f"
    rendered = norm_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        output_access=output_access,
        acc_type=acc_type,
        abs_fn=abs_fn,
        pow_fn=pow_fn,
        p_value=p_literal,
        inv_p_value=inv_p_literal,
        is_zero_p=is_zero_p,
    )
    return rendered.strip().splitlines()


def _write_reduction_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
) -> List[str]:
    reduction_template = _get_template_env().get_template("sum_kernel.c.j2")
    config = _REDUCTION_CONFIG[op_spec.name]
    if dtype.torch_dtype is torch.bool:
        if op_spec.name in {"sum", "mean"}:
            config = {
                "init_value": 0,
                "reduce_op": "|=",
                "post_op": None,
                "bool_reduction": True,
            }
        elif op_spec.name == "amax":
            config = {
                "init_value": 0,
                "reduce_op": "|=",
                "post_op": None,
                "bool_reduction": True,
            }
        elif op_spec.name == "amin":
            config = {
                "init_value": 1,
                "reduce_op": "&=",
                "post_op": None,
                "bool_reduction": True,
            }
        elif op_spec.name == "prod":
            config = {
                "init_value": 1,
                "reduce_op": "&=",
                "post_op": None,
                "bool_reduction": True,
            }
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    reduction_set = set(reduction_dims)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        if keepdim:
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim}"
                for dim in range(input_rank)
            ]
        else:
            dim_to_output: Dict[int, int] = {}
            output_idx = 0
            for dim in range(input_rank):
                if dim in reduction_set:
                    continue
                dim_to_output[dim] = output_idx
                output_idx += 1
            input_indices = [
                f"r{dim}" if dim in reduction_set else f"i{dim_to_output[dim]}"
                for dim in range(input_rank)
            ]
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    compare_op = None
    isnan_fn = None
    if op_spec.name in {"amax", "amin"} and dtype.torch_dtype is not torch.bool:
        compare_op = ">" if op_spec.name == "amax" else "<"
        init_value_config = _MINMAX_INIT_VALUES[dtype.torch_dtype][op_spec.name]
        config = {
            "init_value": init_value_config,
            "post_op": None,
        }
        isnan_fn = (
            f"{dtype.scalar_prefix}isnan"
            if dtype.torch_dtype is torch.float32
            else None
        )
    reduction_count = 1
    for dim in reduction_dims:
        reduction_count *= input_shape[dim]
    bool_reduction = config.get("bool_reduction", False)
    acc_type = "int32_t" if bool_reduction else dtype.c_type
    init_value_config = config["init_value"]
    if isinstance(init_value_config, str):
        init_value = init_value_config
    elif bool_reduction or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        init_value = str(init_value_config)
    else:
        init_value = f"{init_value_config}.0f"
    post_op = None
    if config["post_op"] == "mean":
        if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
            post_op = f"acc /= {reduction_count};"
        else:
            post_op = f"acc /= (float){reduction_count};"
    rendered = reduction_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        bool_reduction=bool_reduction,
        reduce_op=config.get("reduce_op"),
        is_minmax=op_spec.name in {"amax", "amin"}
        and dtype.torch_dtype is not torch.bool,
        compare_op=compare_op if op_spec.name in {"amax", "amin"} else None,
        is_float=dtype.torch_dtype is torch.float32,
        isnan_fn=isnan_fn,
        output_access=output_access,
        acc_type=acc_type,
        init_value=init_value,
        post_op=post_op,
    )
    return rendered.strip().splitlines()


def _write_argminmax_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    reduce_all: bool,
    dtype: _CodegenDType,
) -> List[str]:
    input_c_type = _input_c_type(dtype.torch_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} a{_format_array_suffix(input_shape)}, "
        f"int64_t out{_format_array_suffix(output_shape)}) {{"
    )
    lines = [signature]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type="int64_t"
    )
    if not input_shape:
        lines.append(f"{indent}{output_access} = 0;")
        lines.extend(emit_footer(output_shape, indent))
        return lines
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    compare_op = ">" if op_spec.name == "argmax" else "<"

    def linear_index_expr() -> str:
        expr = f"r0" if input_shape else "0"
        for dim in range(1, len(input_shape)):
            expr = f"({expr} * {input_shape[dim]} + r{dim})"
        return expr

    if reduce_all:
        lines.append(f"{indent}bool has_value = false;")
        lines.append(f"{indent}{input_c_type} best_value = 0;")
        lines.append(f"{indent}int64_t best_index = 0;")
        reduction_indent = indent
        for dim, size in enumerate(input_shape):
            lines.append(
                f"{reduction_indent}for (int64_t r{dim} = 0; r{dim} < {size}; ++r{dim}) {{"
            )
            reduction_indent += "    "
        input_access = _emit_strided_access(
            "a",
            [f"r{dim}" for dim in range(len(input_shape))],
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=input_c_type,
        )
        lines.append(f"{reduction_indent}int64_t linear_index = {linear_index_expr()};")
        lines.append(f"{reduction_indent}{input_c_type} value = {input_access};")
        lines.append(f"{reduction_indent}if (!has_value) {{")
        lines.append(f"{reduction_indent}    best_value = value;")
        lines.append(f"{reduction_indent}    best_index = linear_index;")
        lines.append(f"{reduction_indent}    has_value = true;")
        lines.append(
            f"{reduction_indent}}} else if (value {compare_op} best_value) {{"
        )
        lines.append(f"{reduction_indent}    best_value = value;")
        lines.append(f"{reduction_indent}    best_index = linear_index;")
        lines.append(f"{reduction_indent}}}")
        close_lines, _ = _close_loops(len(input_shape), reduction_indent)
        lines.extend(close_lines)
        lines.append(f"{indent}{output_access} = best_index;")
        lines.extend(emit_footer(output_shape, indent))
        return lines

    reduction_dim = reduction_dims[0]
    dim_to_output: Dict[int, int] = {}
    output_idx = 0
    for dim in range(len(input_shape)):
        if dim == reduction_dim:
            continue
        dim_to_output[dim] = output_idx
        output_idx += 1
    init_indices = []
    loop_indices = []
    for dim in range(len(input_shape)):
        if dim == reduction_dim:
            init_indices.append("0")
            loop_indices.append(f"r{dim}")
        else:
            idx = f"i{dim}" if keepdim else f"i{dim_to_output[dim]}"
            init_indices.append(idx)
            loop_indices.append(idx)
    init_access = _emit_strided_access(
        "a",
        init_indices,
        input_strides,
        contig=a_is_contiguous,
        sizes=input_shape,
        c_type=input_c_type,
    )
    loop_access = _emit_strided_access(
        "a",
        loop_indices,
        input_strides,
        contig=a_is_contiguous,
        sizes=input_shape,
        c_type=input_c_type,
    )
    lines.append(f"{indent}{input_c_type} best_value = {init_access};")
    lines.append(f"{indent}int64_t best_index = 0;")
    lines.append(
        f"{indent}for (int64_t r{reduction_dim} = 1; r{reduction_dim} < {input_shape[reduction_dim]}; ++r{reduction_dim}) {{"
    )
    lines.append(f"{indent}    {input_c_type} value = {loop_access};")
    lines.append(
        f"{indent}    if (value {compare_op} best_value) {{"
    )
    lines.append(f"{indent}        best_value = value;")
    lines.append(f"{indent}        best_index = r{reduction_dim};")
    lines.append(f"{indent}    }}")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}{output_access} = best_index;")
    lines.extend(emit_footer(output_shape, indent))
    return lines


def _write_concat_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shapes: Sequence[Sequence[int]],
    input_strides: Sequence[Sequence[int]],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    concat_dim: int,
    dtype: _CodegenDType,
) -> List[str]:
    input_args = []
    for idx, shape in enumerate(input_shapes):
        suffix = _format_array_suffix(shape)
        input_args.append(f"const {dtype.c_type} a{idx}{suffix}")
    out_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{', '.join(input_args)}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    lines = [signature]
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    offset = 0
    for idx, (shape, strides) in enumerate(zip(input_shapes, input_strides)):
        loop_lines, indent = emit_loops(shape)
        lines.extend(loop_lines)
        indices = [f"i{dim}" for dim in range(len(shape))]
        input_access = _emit_strided_access(
            f"a{idx}",
            indices,
            strides,
            _is_contiguous(shape, strides),
            sizes=shape,
            c_type=dtype.c_type,
        )
        output_indices = [
            f"i{dim} + {offset}" if dim == concat_dim else f"i{dim}"
            for dim in range(len(shape))
        ]
        output_access = _emit_strided_access(
            "out",
            output_indices,
            output_strides,
            output_is_contiguous,
            sizes=output_shape,
            c_type=dtype.c_type,
        )
        lines.append(f"{indent}{output_access} = {input_access};")
        close_lines, indent = _close_loops(len(shape), indent)
        lines.extend(close_lines)
        offset += shape[concat_dim]
    lines.append("}")
    return lines


def _write_embedding_kernel(
    node_index: int,
    op_spec: _OpSpec,
    weight_shape: Sequence[int],
    indices_shape: Sequence[int],
    weight_strides: Sequence[int],
    indices_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    indices_dtype: torch.dtype,
    dtype: _CodegenDType,
    padding_idx: int,
) -> List[str]:
    embedding_template = _get_template_env().get_template(
        "embedding_kernel.c.j2"
    )
    weight_suffix = _format_array_suffix(weight_shape)
    indices_suffix = _format_array_suffix(indices_shape)
    out_suffix = _format_array_suffix(output_shape)
    index_c_type = _dtype_to_c_type(indices_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} weight{weight_suffix}, "
        f"const {index_c_type} indices{indices_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    loop_lines, indent = emit_loops(output_shape)
    indices_rank = len(indices_shape)
    output_rank = len(output_shape)
    output_indices = [f"i{dim}" for dim in range(output_rank)]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    index_indices = [f"i{dim}" for dim in range(indices_rank)]
    index_access = _emit_strided_access(
        "indices",
        index_indices,
        indices_strides,
        _is_contiguous(indices_shape, indices_strides),
        sizes=indices_shape,
        c_type=index_c_type,
    )
    weight_access = _emit_strided_access(
        "weight",
        ["idx", f"i{output_rank - 1}"],
        weight_strides,
        _is_contiguous(weight_shape, weight_strides),
        sizes=weight_shape,
        c_type=dtype.c_type,
    )
    body_lines = [f"{indent}int64_t idx = (int64_t)({index_access});"]
    if padding_idx != -1:
        zero_literal = _format_scalar_literal(0.0, dtype)
        body_lines.extend(
            [
                f"{indent}if (idx == {padding_idx}) {{",
                f"{indent}    {output_access} = {zero_literal};",
                f"{indent}}} else {{",
                f"{indent}    {output_access} = {weight_access};",
                f"{indent}}}",
            ]
        )
    else:
        body_lines.append(f"{indent}{output_access} = {weight_access};")
    footer_lines = emit_footer(output_shape, indent)
    rendered = embedding_template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


def _write_gather_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    index_shape: Sequence[int],
    input_strides: Sequence[int],
    index_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    index_dtype: torch.dtype,
    gather_dim: int,
    dtype: _CodegenDType,
) -> List[str]:
    gather_template = _get_template_env().get_template("gather_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    index_suffix = _format_array_suffix(index_shape)
    out_suffix = _format_array_suffix(output_shape)
    index_c_type = _dtype_to_c_type(index_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {index_c_type} index{index_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    loop_lines, indent = emit_loops(output_shape)
    output_indices = [f"i{dim}" for dim in range(len(output_shape))]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    index_access = _emit_strided_access(
        "index",
        output_indices,
        index_strides,
        _is_contiguous(index_shape, index_strides),
        sizes=index_shape,
        c_type=index_c_type,
    )
    input_indices = [
        "idx" if dim == gather_dim else f"i{dim}"
        for dim in range(len(input_shape))
    ]
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    body_lines = [
        f"{indent}int64_t idx = (int64_t)({index_access});",
        f"{indent}{output_access} = {input_access};",
    ]
    footer_lines = emit_footer(output_shape, indent)
    rendered = gather_template.render(
        signature=signature,
        loop_lines=loop_lines,
        body_lines=body_lines,
        footer_lines=footer_lines,
    )
    return rendered.splitlines()


def _write_conv2d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    output_shape: Sequence[int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    dtype: _CodegenDType,
    has_bias: bool,
) -> List[str]:
    conv2d_template = _get_template_env().get_template("conv2d_kernel.c.j2")
    has_batch, batch, in_channels, in_h, in_w = _unpack_conv2d_input_shape(
        input_shape
    )
    out_channels, _, k_h, k_w = weight_shape
    if has_batch:
        out_h, out_w = output_shape[2], output_shape[3]
    else:
        out_h, out_w = output_shape[1], output_shape[2]
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    weight_suffix = _format_array_suffix(weight_shape)
    output_suffix = _format_array_suffix(output_shape)
    bias_arg = (
        f"const {dtype.c_type} bias[{out_channels}], " if has_bias else ""
    )
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} weight{weight_suffix}, "
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = conv2d_template.render(
        signature=signature,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        groups=groups,
        c_type=dtype.c_type,
        has_bias=has_bias,
        has_batch=has_batch,
    )
    return rendered.strip().splitlines()


def _write_pool2d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    dtype: _CodegenDType,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> List[str]:
    pool2d_template = _get_template_env().get_template("pool2d_kernel.c.j2")
    batch, channels, in_h, in_w = input_shape
    out_h, out_w = output_shape[2], output_shape[3]
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pool2d_template.render(
        signature=signature,
        pool_kind=op_spec.name,
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        c_type=dtype.c_type,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        has_divisor_override=divisor_override is not None,
    )
    return rendered.strip().splitlines()


def _write_pool1d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: _CodegenDType,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> List[str]:
    pool1d_template = _get_template_env().get_template("pool1d_kernel.c.j2")
    batch, channels, in_l = input_shape
    out_l = output_shape[2]
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pool1d_template.render(
        signature=signature,
        pool_kind=op_spec.name,
        batch=batch,
        channels=channels,
        in_l=in_l,
        out_l=out_l,
        k_l=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        c_type=dtype.c_type,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        has_divisor_override=divisor_override is not None,
    )
    return rendered.strip().splitlines()


def _write_col2im_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    output_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    padding: Tuple[int, int],
    stride: Tuple[int, int],
    dtype: _CodegenDType,
    out_blocks_h: int,
    out_blocks_w: int,
) -> List[str]:
    col2im_template = _get_template_env().get_template("col2im_kernel.c.j2")
    if len(output_shape) == 4:
        batch, channels, out_h, out_w = output_shape
        has_batch = True
    else:
        channels, out_h, out_w = output_shape
        batch = 1
        has_batch = False
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = col2im_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        c_type=dtype.c_type,
        out_blocks_h=out_blocks_h,
        out_blocks_w=out_blocks_w,
        has_batch=has_batch,
    )
    return rendered.strip().splitlines()


def _write_batch_norm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
    eps: float,
    has_weight: bool,
    has_bias: bool,
) -> List[str]:
    batch_norm_template = _get_template_env().get_template(
        "batch_norm_kernel.c.j2"
    )
    batch = input_shape[0]
    channels = input_shape[1]
    inner_size = 1
    for dim in input_shape[2:]:
        inner_size *= dim
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    weight_arg = (
        f"const {dtype.c_type} weight[{channels}], " if has_weight else ""
    )
    bias_arg = (
        f"const {dtype.c_type} bias[{channels}], " if has_bias else ""
    )
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} running_mean[{channels}], "
        f"const {dtype.c_type} running_var[{channels}], "
        f"{weight_arg}"
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = batch_norm_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        inner_size=inner_size,
        c_type=dtype.c_type,
        eps=_format_scalar_literal(eps, dtype),
        has_weight=has_weight,
        has_bias=has_bias,
        one_literal=_format_scalar_literal(1.0, dtype),
        zero_literal=_format_scalar_literal(0.0, dtype),
    )
    return rendered.strip().splitlines()


def _write_pdist_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    pdist_template = _get_template_env().get_template("pdist_kernel.c.j2")
    n, m = input_shape
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pdist_template.render(
        signature=signature,
        n=n,
        m=m,
        c_type=dtype.c_type,
    )
    return rendered.strip().splitlines()


def _write_conv1d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    output_shape: Sequence[int],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    dtype: _CodegenDType,
    has_bias: bool,
) -> List[str]:
    conv1d_template = _get_template_env().get_template("conv1d_kernel.c.j2")
    batch, in_channels, in_l = input_shape
    out_channels, _, k_l = weight_shape
    out_l = output_shape[2]
    input_suffix = _format_array_suffix(input_shape)
    weight_suffix = _format_array_suffix(weight_shape)
    output_suffix = _format_array_suffix(output_shape)
    bias_arg = (
        f"const {dtype.c_type} bias[{out_channels}], " if has_bias else ""
    )
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} weight{weight_suffix}, "
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = conv1d_template.render(
        signature=signature,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_l=in_l,
        out_l=out_l,
        k_l=k_l,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        c_type=dtype.c_type,
        has_bias=has_bias,
    )
    return rendered.strip().splitlines()


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
            raise RefBackendError(
                f"codegen backend does not support kind '{op_node.spec.kind}'"
            )
        kernel_lines = handler.emit_kernel(index, op_node, graph)
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
    template = _get_template_env().get_template("generic_source.c.j2")
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
            raise RefBackendError(
                "codegen backend supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
            )
        return None
    for example in _iter_example_tensors(example_inputs):
        if example.device.type != "cpu":
            raise RefBackendError("codegen backend supports only CPU tensors")
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
            raise RefBackendError(
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
        raise RefBackendError(
            "codegen backend supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
        )
    for example in tensor_examples:
        if example.dtype is torch.bool:
            continue
        if (
            example.dtype is not first_dtype
            and example.dtype not in _EMBEDDING_INDEX_DTYPES
        ):
            raise RefBackendError(
                "codegen backend expects all tensors to share a dtype"
            )
    return dtype_info


def _infer_empty_strided_dtype(
    gm: torch.fx.GraphModule,
) -> _CodegenDType | None:
    dtype_value = None
    found_empty_strided = False
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target_info = TARGET_REGISTRY.get(node.target)
        if target_info is None or target_info.op_spec.kind != "empty_strided":
            continue
        found_empty_strided = True
        node_dtype = None
        if len(node.args) > 2:
            node_dtype = node.args[2]
        if "dtype" in node.kwargs:
            if node_dtype is not None:
                raise _error_kwarg_specified_once("empty_strided", "dtype")
            node_dtype = node.kwargs["dtype"]
        if isinstance(node_dtype, torch.fx.Node):
            raise RefBackendError(
                "codegen empty_strided expects dtype to be a constant"
            )
        if node_dtype is None:
            raise RefBackendError(
                "codegen empty_strided requires dtype when no tensor inputs are provided"
            )
        if dtype_value is None:
            dtype_value = node_dtype
        elif dtype_value is not node_dtype:
            raise RefBackendError(
                "codegen empty_strided requires a consistent dtype"
            )
    if not found_empty_strided:
        return None
    dtype_info = _CODEGEN_DTYPES.get(dtype_value)
    if dtype_info is None:
        raise RefBackendError(
            "codegen empty_strided supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
        )
    return dtype_info


def _unwrap_output_node(output_node: torch.fx.Node) -> Tuple[torch.fx.Node, object]:
    output_value = output_node.args[0]
    output_structure = output_value
    if isinstance(output_value, (tuple, list, immutable_list)):
        if len(output_value) != 1 or not isinstance(output_value[0], torch.fx.Node):
            raise RefBackendError("codegen backend expects a single output node")
        output_value = output_value[0]
    if not isinstance(output_value, torch.fx.Node):
        raise RefBackendError("codegen backend expects a single output node")
    return output_value, output_structure


def _infer_output_shape(
    op_spec: _OpSpec, input_shapes: Sequence[Tuple[int, ...]]
) -> Tuple[int, ...]:
    handler = _KIND_HANDLERS.get(op_spec.kind)
    if handler is None:
        raise RefBackendError(
            f"codegen backend does not support kind '{op_spec.kind}'"
        )
    return handler.infer_output_shape(op_spec, input_shapes)


def _normalize_flip_dims(
    op_name: str, dims: object, rank: int
) -> Tuple[int, ...]:
    if dims is None:
        raise RefBackendError(f"codegen {op_name} expects dims to be provided")
    if isinstance(dims, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_name} expects dims to be an int or tuple of ints"
        )
    if isinstance(dims, (tuple, list)):
        dims_list = list(dims)
    else:
        dims_list = [dims]
    if not dims_list:
        return ()
    if rank == 0:
        raise RefBackendError(
            f"codegen {op_name} expects dims to be within the input rank"
        )
    normalized = []
    seen = set()
    for item in dims_list:
        if isinstance(item, torch.fx.Node):
            raise RefBackendError(
                f"codegen {op_name} expects dims to be an int or tuple of ints"
            )
        try:
            dim = operator.index(item)
        except TypeError as exc:
            raise RefBackendError(
                f"codegen {op_name} expects dims to be an int or tuple of ints"
            ) from exc
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise RefBackendError(
                f"codegen {op_name} expects dims to be within the input rank"
            )
        if dim in seen:
            raise RefBackendError(
                f"codegen {op_name} expects dims to be unique"
            )
        seen.add(dim)
        normalized.append(dim)
    return tuple(normalized)
def _infer_diagonal_output_shape(
    input_shape: Sequence[int], offset: int, dim1: int, dim2: int
) -> Tuple[int, ...]:
    size1 = input_shape[dim1]
    size2 = input_shape[dim2]
    if offset >= 0:
        diag_len = min(size1, size2 - offset)
    else:
        diag_len = min(size1 + offset, size2)
    diag_len = max(0, diag_len)
    output_dims = [
        size
        for index, size in enumerate(input_shape)
        if index not in (dim1, dim2)
    ]
    output_dims.append(diag_len)
    return tuple(output_dims)


def _normalize_reduction_dims(
    op_name: str, dim: object | None, rank: int
) -> Tuple[int, ...]:
    if dim is None:
        return tuple(range(rank))
    if isinstance(dim, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_name} expects dim to be an int or tuple of ints"
        )
    if rank == 0:
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        for item in dims:
            if isinstance(item, torch.fx.Node):
                raise RefBackendError(
                    f"codegen {op_name} expects dim to be an int or tuple of ints"
                )
            try:
                dim_value = operator.index(item)
            except TypeError as exc:
                raise RefBackendError(
                    f"codegen {op_name} expects dim to be an int or tuple of ints"
                ) from exc
            if dim_value not in (-1, 0):
                raise RefBackendError(f"codegen {op_name} dim is out of range")
        return ()
    if isinstance(dim, (tuple, list)):
        dims = dim
    else:
        dims = (dim,)
    normalized: List[int] = []
    seen: set[int] = set()
    for item in dims:
        if isinstance(item, torch.fx.Node):
            raise RefBackendError(
                f"codegen {op_name} expects dim to be an int or tuple of ints"
            )
        try:
            dim_value = operator.index(item)
        except TypeError as exc:
            raise RefBackendError(
                f"codegen {op_name} expects dim to be an int or tuple of ints"
            ) from exc
        if dim_value < 0:
            dim_value += rank
        if dim_value < 0 or dim_value >= rank:
            raise RefBackendError(f"codegen {op_name} dim is out of range")
        if dim_value in seen:
            continue
        seen.add(dim_value)
        normalized.append(dim_value)
    return tuple(sorted(normalized))


def _error_expected_tensor(op_name: str) -> RefBackendError:
    return RefBackendError(f"codegen {op_name} expects tensor inputs only")


def _error_kwarg_specified_once(op_name: str, kwarg: str) -> RefBackendError:
    return RefBackendError(f"codegen {op_name} expects {kwarg} to be specified once")


def _normalize_param(normalizer: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return normalizer(*args, **kwargs)
    except ValueError as exc:
        raise RefBackendError(str(exc)) from exc


def _infer_reduction_output_shape(
    input_shape: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    *,
    reduce_all: bool,
) -> Tuple[int, ...]:
    if reduce_all:
        return ()
    if not reduction_dims:
        return tuple(input_shape)
    if keepdim:
        output_shape = list(input_shape)
        for dim in reduction_dims:
            output_shape[dim] = 1
        return tuple(output_shape)
    return tuple(
        size for dim, size in enumerate(input_shape) if dim not in reduction_dims
    )


def _parse_reduction_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[Tuple[int, ...], bool, bool, bool | None]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 4:
        raise RefBackendError(f"codegen {op_name} expects at most four inputs")
    if op_name == "std":
        unbiased = True
        if len(node.args) > 2:
            raise RefBackendError(
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
                raise RefBackendError(
                    f"codegen std got unexpected kwargs: {sorted(extra)}"
                )
        if isinstance(unbiased, torch.fx.Node):
            raise RefBackendError("codegen std expects unbiased to be a bool")
        if not isinstance(unbiased, bool):
            raise RefBackendError("codegen std expects unbiased to be a bool")
        reduction_dims = tuple(range(len(input_shape)))
        keepdim = False
        reduce_all = True
        return reduction_dims, keepdim, reduce_all, unbiased
    if op_name == "var":
        if len(node.args) > 4:
            raise RefBackendError(
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
                raise RefBackendError(
                    f"codegen var got unexpected kwargs: {sorted(extra)}"
                )
        if isinstance(unbiased, torch.fx.Node):
            raise RefBackendError("codegen var expects unbiased to be a bool")
        if not isinstance(unbiased, bool):
            raise RefBackendError("codegen var expects unbiased to be a bool")
        if isinstance(keepdim, torch.fx.Node):
            raise RefBackendError("codegen var expects keepdim to be a bool")
        if not isinstance(keepdim, bool):
            raise RefBackendError("codegen var expects keepdim to be a bool")
        if correction is not None:
            if isinstance(correction, torch.fx.Node):
                raise RefBackendError("codegen var expects correction to be 0 or 1")
            if not isinstance(correction, numbers.Number):
                raise RefBackendError("codegen var expects correction to be 0 or 1")
            correction_value = float(correction)
            if correction_value not in (0.0, 1.0):
                raise RefBackendError("codegen var expects correction to be 0 or 1")
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
            raise RefBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(keepdim, torch.fx.Node):
        raise RefBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if not isinstance(keepdim, bool):
        raise RefBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if dtype is not None:
        if isinstance(dtype, torch.fx.Node):
            raise RefBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.int8, torch.int32, or torch.bool"
            )
        if dtype not in (torch.float32, torch.int8, torch.int32, torch.bool):
            raise RefBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.int8, torch.int32, or torch.bool"
            )
    reduction_dims = _normalize_reduction_dims(op_name, dim, len(input_shape))
    reduce_all = dim is None
    return reduction_dims, keepdim, reduce_all, None


def _parse_norm_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[Tuple[int, ...], bool, bool, float]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 4:
        raise RefBackendError(
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
            raise RefBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(keepdim, torch.fx.Node):
        raise RefBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if not isinstance(keepdim, bool):
        raise RefBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if isinstance(p, torch.fx.Node):
        raise RefBackendError(f"codegen {op_name} expects p to be a number")
    if p is None:
        p_value = 2.0
    elif isinstance(p, bool):
        p_value = float(p)
    elif isinstance(p, (int, float)):
        p_value = float(p)
    else:
        raise RefBackendError(f"codegen {op_name} expects p to be a number")
    if math.isinf(p_value) or math.isnan(p_value):
        raise RefBackendError(f"codegen {op_name} expects p to be finite")
    reduction_dims = _normalize_reduction_dims(op_name, dim, len(input_shape))
    reduce_all = dim is None
    return reduction_dims, keepdim, reduce_all, p_value


def _parse_argminmax_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[Tuple[int, ...], bool, bool]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 3:
        raise RefBackendError(f"codegen {op_name} expects at most three inputs")
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
            raise RefBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(keepdim, torch.fx.Node):
        raise RefBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if not isinstance(keepdim, bool):
        raise RefBackendError(f"codegen {op_name} expects keepdim to be a bool")
    if dim is None:
        reduction_dims = tuple(range(len(input_shape)))
        reduce_all = True
        return reduction_dims, keepdim, reduce_all
    if isinstance(dim, (tuple, list)):
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    dim_value = _parse_constant_int(op_name, "dim", dim)
    rank = len(input_shape)
    if rank == 0:
        if dim_value not in (-1, 0):
            raise RefBackendError(f"codegen {op_name} dim is out of range")
        return (), keepdim, True
    if dim_value < 0:
        dim_value += rank
    if dim_value < 0 or dim_value >= rank:
        raise RefBackendError(f"codegen {op_name} dim is out of range")
    return (dim_value,), keepdim, False


def _parse_constant_float(op_name: str, name: str, value: object) -> float:
    if isinstance(value, torch.fx.Node):
        raise RefBackendError(f"codegen {op_name} expects {name} to be constant")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise RefBackendError(f"codegen {op_name} expects {name} to be numeric")


def _parse_constant_int(op_name: str, name: str, value: object) -> int:
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise RefBackendError(f"codegen {op_name} expects {name} to be an int")
        return _parse_constant_int(op_name, name, meta_value)
    if isinstance(value, bool):
        raise RefBackendError(f"codegen {op_name} expects {name} to be an int")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RefBackendError(f"codegen {op_name} expects {name} to be an int")
        return _parse_constant_int(op_name, name, value.item())
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise RefBackendError(
                f"codegen {op_name} expects {name} to be an int"
            ) from exc
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise RefBackendError(f"codegen {op_name} expects {name} to be an int")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise RefBackendError(
            f"codegen {op_name} expects {name} to be an int"
        ) from exc


def _parse_bitwise_scalar(
    op_name: str, value: object, dtype: torch.dtype
) -> object:
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise RefBackendError(
                f"codegen {op_name} expects scalar to be constant"
            )
        return _parse_bitwise_scalar(op_name, meta_value, dtype)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RefBackendError(
                f"codegen {op_name} expects scalar to be a single value"
            )
        return _parse_bitwise_scalar(op_name, value.item(), dtype)
    if dtype is torch.bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise RefBackendError(
                    f"codegen {op_name} expects scalar to be a boolean value"
                )
            return bool(int(value))
        try:
            return bool(operator.index(value))
        except TypeError as exc:
            raise RefBackendError(
                f"codegen {op_name} expects scalar to be a boolean value"
            ) from exc
    if isinstance(value, bool):
        return int(value)
    return _parse_constant_int(op_name, "scalar", value)


def _parse_parametric_unary_args(
    op_name: str, node: torch.fx.Node
) -> Tuple[torch.fx.Node, Dict[str, object]]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    input_node = node.args[0]
    params: Dict[str, object] = {}
    if op_name == "gelu":
        if len(node.args) > 2:
            raise RefBackendError("codegen gelu expects one input")
        if len(node.args) > 1:
            params["approximate"] = node.args[1]
        if "approximate" in node.kwargs:
            if len(node.args) > 1:
                raise RefBackendError("codegen gelu expects approximate as a keyword")
            params["approximate"] = node.kwargs["approximate"]
        extra = set(node.kwargs) - {"approximate"}
        if extra:
            raise RefBackendError(
                f"codegen gelu got unexpected kwargs: {sorted(extra)}"
            )
        approximate = params.get("approximate", "none")
        if isinstance(approximate, torch.fx.Node):
            raise RefBackendError("codegen gelu expects approximate to be constant")
        if approximate is None:
            approximate = "none"
        if approximate not in {"none", "tanh"}:
            raise RefBackendError(
                "codegen gelu expects approximate to be 'none' or 'tanh'"
            )
        params["approximate"] = approximate
        return input_node, params
    if op_name == "elu":
        if len(node.args) > 4:
            raise RefBackendError("codegen elu expects one input")
        args = list(node.args[1:])
        kwargs = dict(node.kwargs)
        for name in ("alpha", "scale", "input_scale"):
            if name in kwargs and args:
                raise RefBackendError(f"codegen elu got multiple values for {name}")
            if args:
                params[name] = args.pop(0)
            elif name in kwargs:
                params[name] = kwargs[name]
        extra = set(kwargs) - {"alpha", "scale", "input_scale"}
        if extra:
            raise RefBackendError(
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
            raise RefBackendError("codegen leaky_relu expects one input")
        if len(node.args) > 1:
            params["negative_slope"] = node.args[1]
        if "negative_slope" in node.kwargs:
            if len(node.args) > 1:
                raise RefBackendError(
                    "codegen leaky_relu expects negative_slope as a keyword"
                )
            params["negative_slope"] = node.kwargs["negative_slope"]
        extra = set(node.kwargs) - {"negative_slope"}
        if extra:
            raise RefBackendError(
                f"codegen leaky_relu got unexpected kwargs: {sorted(extra)}"
            )
        params["negative_slope"] = _parse_constant_float(
            op_name, "negative_slope", params.get("negative_slope", 0.01)
        )
        return input_node, params
    if op_name == "softplus":
        if len(node.args) > 3:
            raise RefBackendError("codegen softplus expects one input")
        if len(node.args) > 1:
            params["beta"] = node.args[1]
        if len(node.args) > 2:
            params["threshold"] = node.args[2]
        if "beta" in node.kwargs:
            if len(node.args) > 1:
                raise RefBackendError("codegen softplus expects beta as a keyword")
            params["beta"] = node.kwargs["beta"]
        if "threshold" in node.kwargs:
            if len(node.args) > 2:
                raise RefBackendError(
                    "codegen softplus expects threshold as a keyword"
                )
            params["threshold"] = node.kwargs["threshold"]
        extra = set(node.kwargs) - {"beta", "threshold"}
        if extra:
            raise RefBackendError(
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
            raise RefBackendError("codegen clamp expects one input")
        if len(node.args) > 1:
            params["min_val"] = node.args[1]
        if len(node.args) > 2:
            params["max_val"] = node.args[2]
        if "min" in node.kwargs:
            if len(node.args) > 1:
                raise RefBackendError("codegen clamp expects min as a keyword")
            params["min_val"] = node.kwargs["min"]
        if "max" in node.kwargs:
            if len(node.args) > 2:
                raise RefBackendError("codegen clamp expects max as a keyword")
            params["max_val"] = node.kwargs["max"]
        extra = set(node.kwargs) - {"min", "max"}
        if extra:
            raise RefBackendError(
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
            raise RefBackendError("codegen hardtanh expects one input")
        if len(node.args) > 1:
            params["min_val"] = node.args[1]
        if len(node.args) > 2:
            params["max_val"] = node.args[2]
        if "min_val" in node.kwargs:
            if len(node.args) > 1:
                raise RefBackendError("codegen hardtanh expects min_val as a keyword")
            params["min_val"] = node.kwargs["min_val"]
        if "max_val" in node.kwargs:
            if len(node.args) > 2:
                raise RefBackendError("codegen hardtanh expects max_val as a keyword")
            params["max_val"] = node.kwargs["max_val"]
        extra = set(node.kwargs) - {"min_val", "max_val"}
        if extra:
            raise RefBackendError(
                f"codegen hardtanh got unexpected kwargs: {sorted(extra)}"
            )
        params["min_val"] = _parse_constant_float(
            op_name, "min_val", params.get("min_val", -1.0)
        )
        params["max_val"] = _parse_constant_float(
            op_name, "max_val", params.get("max_val", 1.0)
        )
        return input_node, params
    raise RefBackendError(f"Unsupported parametric op: {op_name}")


def _parse_softmax_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, object | None]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    is_internal = op_name in {"_softmax", "_log_softmax"}
    if len(node.args) > 3:
        raise RefBackendError(f"codegen {op_name} expects at most three inputs")
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
            raise RefBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None:
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, torch.fx.Node):
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, (tuple, list)):
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    try:
        dim_value = operator.index(dim)
    except TypeError as exc:
        raise RefBackendError(f"codegen {op_name} expects dim to be an int") from exc
    rank = len(input_shape)
    if dim_value < 0:
        dim_value += rank
    if dim_value < 0 or dim_value >= rank:
        raise RefBackendError(f"codegen {op_name} dim is out of range")
    if is_internal:
        if half_to_float is None:
            half_to_float = False
        if isinstance(half_to_float, torch.fx.Node):
            raise RefBackendError(
                f"codegen {op_name} expects half_to_float to be a bool"
            )
        if half_to_float not in (False, 0):
            raise RefBackendError(
                f"codegen {op_name} expects half_to_float to be False"
            )
    else:
        if dtype is not None:
            if isinstance(dtype, torch.fx.Node):
                raise RefBackendError(
                    f"codegen {op_name} expects dtype to be torch.float32 or None"
                )
            if dtype is not torch.float32:
                raise RefBackendError(
                    f"codegen {op_name} expects dtype to be torch.float32 or None"
                )
    return dim_value, dtype


def _parse_diagonal_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, int, int]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 4:
        raise RefBackendError(f"codegen {op_name} expects at most four inputs")
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
            raise RefBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    offset_value = _parse_constant_int(op_name, "offset", offset)
    dim1_value = _parse_constant_int(op_name, "dim1", dim1)
    dim2_value = _parse_constant_int(op_name, "dim2", dim2)
    rank = len(input_shape)
    if rank < 2:
        raise RefBackendError(f"codegen {op_name} expects input rank >= 2")
    if dim1_value < 0:
        dim1_value += rank
    if dim2_value < 0:
        dim2_value += rank
    if dim1_value < 0 or dim1_value >= rank:
        raise RefBackendError(f"codegen {op_name} dim1 is out of range")
    if dim2_value < 0 or dim2_value >= rank:
        raise RefBackendError(f"codegen {op_name} dim2 is out of range")
    if dim1_value == dim2_value:
        raise RefBackendError(f"codegen {op_name} expects dim1 != dim2")
    return offset_value, dim1_value, dim2_value


def _write_softmax_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_strides: Sequence[int],
    softmax_dim: int | None,
    dtype: _CodegenDType,
) -> List[str]:
    if softmax_dim is None:
        raise RefBackendError("codegen softmax expects a reduction dimension")
    softmax_template = _get_template_env().get_template("softmax_kernel.c.j2")
    rank = len(input_shape)
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(input_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    output_dims = [{"dim": dim, "size": size} for dim, size in enumerate(input_shape)]
    input_contig = _is_contiguous(input_shape, input_strides)
    current_indices = [f"i{dim}" for dim in range(rank)]
    r_indices = current_indices.copy()
    r_indices[softmax_dim] = f"r{softmax_dim}"
    zero_indices = current_indices.copy()
    zero_indices[softmax_dim] = "0"
    input_access_r = _emit_strided_access(
        "input",
        r_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    input_access_zero = _emit_strided_access(
        "input",
        zero_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    input_access_current = _emit_strided_access(
        "input",
        current_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    output_access = _format_output_access(
        "out", input_shape, output_strides, c_type=dtype.c_type
    )
    rendered = softmax_template.render(
        signature=signature,
        output_dims=output_dims,
        softmax_dim=softmax_dim,
        softmax_size=input_shape[softmax_dim],
        c_type=dtype.c_type,
        input_access_zero=input_access_zero,
        input_access_r=input_access_r,
        input_access_current=input_access_current,
        output_access=output_access,
        is_log=op_spec.name in {"log_softmax", "_log_softmax"},
    )
    return rendered.strip().splitlines()


def _write_diagonal_kernel(
    node_index: int,
    op_node: _OpNode,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    input_dtype: torch.dtype,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_node.spec.name}_{dtype.suffix}("
        f"const {input_c_type} a{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    lines = [signature]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    input_access = _format_diagonal_access(
        "a",
        input_shape,
        input_strides,
        output_shape,
        dim1=int(op_node.p("dim1")),
        dim2=int(op_node.p("dim2")),
        offset=int(op_node.p("offset")),
        c_type=input_c_type,
    )
    lines.append(f"{indent}{output_access} = {input_access};")
    lines.extend(emit_footer(output_shape, indent))
    return lines


def _write_cumsum_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_strides: Sequence[int],
    cumsum_dim: int,
    dtype: _CodegenDType,
) -> List[str]:
    input_c_type = _input_c_type(dtype.torch_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} input{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(input_shape)}) {{"
    )
    lines = [signature]
    if not input_shape:
        lines.append("    out[0] = input[0];")
        lines.append("}")
        return lines
    loop_lines, indent = emit_loops(input_shape)
    lines.extend(loop_lines)
    output_access = _emit_strided_access(
        "out",
        [f"i{dim}" for dim in range(len(input_shape))],
        output_strides,
        _is_contiguous(input_shape, output_strides),
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    lines.append(f"{indent}{dtype.c_type} acc = {acc_init};")
    lines.append(
        f"{indent}for (int64_t r{cumsum_dim} = 0; r{cumsum_dim} <= i{cumsum_dim}; ++r{cumsum_dim}) {{"
    )
    inner_indent = f"{indent}    "
    input_indices = [
        f"r{cumsum_dim}" if dim == cumsum_dim else f"i{dim}"
        for dim in range(len(input_shape))
    ]
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=input_c_type,
    )
    lines.append(f"{inner_indent}acc += {input_access};")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}{output_access} = acc;")
    lines.extend(emit_footer(input_shape, indent))
    return lines


def _parse_cumsum_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, torch.dtype | None]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 3:
        raise RefBackendError(
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
            raise RefBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None:
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, (tuple, list)):
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    dim_value = _parse_constant_int(op_name, "dim", dim)
    rank = len(input_shape)
    if rank == 0:
        if dim_value not in (-1, 0):
            raise RefBackendError(f"codegen {op_name} dim is out of range")
        dim_value = 0
    else:
        if dim_value < 0:
            dim_value += rank
        if dim_value < 0 or dim_value >= rank:
            raise RefBackendError(f"codegen {op_name} dim is out of range")
    if dtype is not None:
        if isinstance(dtype, torch.fx.Node):
            raise RefBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.int8, or torch.int32"
            )
        if dtype not in (torch.float32, torch.int8, torch.int32):
            raise RefBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.int8, or torch.int32"
            )
    return dim_value, dtype


def _parse_gather_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object, object]:
    if len(node.args) > 4:
        raise RefBackendError("codegen gather expects at most four inputs")
    if not node.args:
        raise RefBackendError("codegen gather expects input, dim, and index")
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
            raise RefBackendError(
                f"codegen gather got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None or index is None:
        raise RefBackendError("codegen gather expects dim and index arguments")
    if sparse_grad is None:
        sparse_grad = False
    return input_arg, dim, index, sparse_grad


def _parse_addmm_like_scalar(
    op_name: str, name: str, value: object | None
) -> float:
    if value is None:
        return 1.0
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise RefBackendError(f"codegen {op_name} expects {name} to be a number")
        return _parse_addmm_like_scalar(op_name, name, meta_value)
    if isinstance(value, bool):
        raise RefBackendError(f"codegen {op_name} expects {name} to be a number")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RefBackendError(f"codegen {op_name} expects {name} to be a number")
        return float(value.item())
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise RefBackendError(
                f"codegen {op_name} expects {name} to be a number"
            ) from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RefBackendError(
            f"codegen {op_name} expects {name} to be a number"
        ) from exc


def _validate_addmm_like_scalars(
    op_name: str, dtype: torch.dtype, alpha: float, beta: float
) -> None:
    if dtype in _INTEGER_CODEGEN_DTYPES or dtype is torch.bool:
        for name, value in (("alpha", alpha), ("beta", beta)):
            if not float(value).is_integer():
                raise RefBackendError(
                    f"codegen {op_name} expects {name} to be an integer for integral tensors"
                )


def _parse_addmm_like_args(
    op_name: str, node: torch.fx.Node
) -> Tuple[Sequence[torch.fx.Node], float, float]:
    if len(node.args) < 3:
        raise RefBackendError(f"codegen {op_name} expects at least three inputs")
    if len(node.args) > 5:
        raise RefBackendError(f"codegen {op_name} expects at most five inputs")
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
            raise RefBackendError(
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
        raise RefBackendError("codegen cat expects a tensor list input")
    if len(node.args) > 2:
        raise RefBackendError("codegen cat expects at most two inputs")
    tensors_arg = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise _error_kwarg_specified_once("cat", "dim")
            dim = node.kwargs["dim"]
        extra = set(node.kwargs) - {"dim"}
        if extra:
            raise RefBackendError(
                f"codegen cat got unexpected kwargs: {sorted(extra)}"
            )
    if isinstance(dim, torch.fx.Node):
        raise RefBackendError("codegen cat expects dim to be an int")
    if dim is None:
        dim_value = 0
    else:
        try:
            dim_value = operator.index(dim)
        except TypeError as exc:
            raise RefBackendError("codegen cat expects dim to be an int") from exc
    if not isinstance(tensors_arg, (list, tuple)) or not tensors_arg:
        raise RefBackendError("codegen cat expects a non-empty tensor list input")
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
        raise RefBackendError("codegen conv2d expects convolution arguments")
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
    elif len(args) == 9:
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
            raise RefBackendError(
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
        raise RefBackendError("codegen conv1d expects convolution arguments")
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
            raise RefBackendError(
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
        raise RefBackendError(
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
            raise RefBackendError(
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
        raise RefBackendError(
            "codegen col2im expects input, output_size, kernel_size, dilation, padding, and stride"
        )
    return input_arg, output_size, kernel_size, dilation, padding, stride


def _parse_max_pool1d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 6:
        raise RefBackendError("codegen max_pool1d expects pooling arguments")
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
            raise RefBackendError(
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
        raise RefBackendError("codegen avg_pool1d expects pooling arguments")
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
            raise RefBackendError(
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
        raise RefBackendError(
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
            raise RefBackendError(
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
        raise RefBackendError(
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
            raise RefBackendError(
                "codegen adaptive_avg_pool2d got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    return input_arg, output_size


def _parse_max_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 6:
        raise RefBackendError("codegen max_pool2d expects pooling arguments")
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
            raise RefBackendError(
                f"codegen max_pool2d got unexpected kwargs: {sorted(extra)}"
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


def _parse_avg_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 7:
        raise RefBackendError("codegen avg_pool2d expects pooling arguments")
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
            raise RefBackendError(
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


def _handle_concat_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    input_nodes, concat_dim = _parse_concat_args(node)
    input_shapes: List[Tuple[int, ...]] = []
    for arg in input_nodes:
        if arg not in shapes:
            raise _error_expected_tensor("cat")
        input_shapes.append(shapes[arg])
    if not input_shapes:
        raise RefBackendError("codegen cat expects a non-empty tensor list input")
    rank = len(input_shapes[0])
    if rank == 0:
        raise RefBackendError("codegen cat expects inputs with rank >= 1")
    if concat_dim < 0:
        concat_dim += rank
    if concat_dim < 0 or concat_dim >= rank:
        raise RefBackendError("codegen cat dim is out of range")
    for shape in input_shapes:
        if len(shape) != rank:
            raise RefBackendError("codegen cat expects inputs with the same rank")
        for dim, size in enumerate(shape):
            if dim == concat_dim:
                continue
            if size != input_shapes[0][dim]:
                raise RefBackendError(
                    "codegen cat expects input shapes to match except in the concat dimension"
                )
    input_dtypes = [dtypes[arg] for arg in input_nodes]
    if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
        raise RefBackendError("codegen cat expects inputs to share the graph dtype")
    output_shape = list(input_shapes[0])
    output_shape[concat_dim] = sum(shape[concat_dim] for shape in input_shapes)
    output_shape = tuple(output_shape)
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=list(input_nodes),
        output_shape=output_shape,
        inplace_input=None,
        params={"dim": concat_dim},
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
        raise RefBackendError(f"codegen {op_spec.name} expects one input")
    input_node = node.args[0]
    if not isinstance(input_node, torch.fx.Node) or input_node not in shapes:
        raise _error_expected_tensor(op_spec.name)
    dims = None
    if len(node.args) >= 2:
        dims = node.args[1]
        if len(node.args) > 2:
            raise RefBackendError(
                f"codegen {op_spec.name} expects one input and dims"
            )
    if "dims" in node.kwargs:
        if dims is not None:
            raise RefBackendError(
                f"codegen {op_spec.name} expects dims only once"
            )
        dims = node.kwargs["dims"]
    if node.kwargs and "dims" not in node.kwargs:
        raise RefBackendError(
            f"codegen {op_spec.name} expects dims as the only keyword argument"
        )
    input_shape = shapes[input_node]
    if dtypes[input_node] is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    normalized_dims = _normalize_flip_dims(op_spec.name, dims, len(input_shape))
    output_shape = input_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_node],
        output_shape=output_shape,
        inplace_input=None,
        params={"dims": normalized_dims},
    )


def _handle_conv1d_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
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
            raise RefBackendError("codegen conv1d expects bias to be a tensor")
    if isinstance(stride, torch.fx.Node) or isinstance(
        padding, torch.fx.Node
    ) or isinstance(dilation, torch.fx.Node):
        raise RefBackendError(
            "codegen conv1d expects constant stride, padding, and dilation"
        )
    if isinstance(groups, torch.fx.Node):
        raise RefBackendError("codegen conv1d expects constant groups")
    if dtype_info.torch_dtype is not torch.float32:
        raise RefBackendError("codegen conv1d supports only torch.float32 tensors")
    if input_arg not in shapes or weight_arg not in shapes:
        raise _error_expected_tensor("conv1d")
    if dtypes[input_arg] is not torch.float32 or dtypes[weight_arg] is not torch.float32:
        raise RefBackendError("codegen conv1d supports only torch.float32 tensors")
    if bias_node is not None:
        if bias_node not in shapes:
            raise _error_expected_tensor("conv1d")
        if dtypes[bias_node] is not torch.float32:
            raise RefBackendError("codegen conv1d supports only torch.float32 tensors")
    input_shape = shapes[input_arg]
    weight_shape = shapes[weight_arg]
    if bias_node is not None:
        bias_shape = shapes[bias_node]
        if len(bias_shape) != 1 or bias_shape[0] != weight_shape[0]:
            raise RefBackendError(
                "codegen conv1d expects bias shape to match output channels"
            )
    if len(input_shape) != 3 or len(weight_shape) != 3:
        raise RefBackendError("codegen conv1d requires 3D input and weight tensors")
    stride_value = _normalize_param(normalize_int_or_tuple, "stride", stride, 1)[0]
    dilation_value = _normalize_param(
        normalize_int_or_tuple, "dilation", dilation, 1
    )[0]
    padding_value = _normalize_param(
        normalize_padding, "padding", padding, 1, allow_strings=("same", "valid")
    )
    if isinstance(padding_value, str):
        if padding_value == "valid":
            padding_value = 0
            output_shape = _conv1d_output_shape_from_shapes(
                input_shape,
                weight_shape,
                stride_value,
                padding_value,
                dilation_value,
                groups,
            )
        else:
            batch, out_channels = _conv1d_validate_channels(
                input_shape, weight_shape, groups
            )
            padding_value, out_l = _conv1d_same_padding(
                input_shape, weight_shape, stride_value, dilation_value
            )
            output_shape = (batch, out_channels, out_l)
    else:
        padding_value = padding_value[0]
        output_shape = _conv1d_output_shape_from_shapes(
            input_shape,
            weight_shape,
            stride_value,
            padding_value,
            dilation_value,
            groups,
        )
    if stride_value <= 0 or dilation_value <= 0 or padding_value < 0:
        raise RefBackendError(
            "codegen conv1d expects stride and dilation to be positive and padding to be non-negative"
        )
    if not isinstance(groups, int) or groups <= 0:
        raise RefBackendError("codegen conv1d requires positive groups")
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    inputs = (input_arg, weight_arg)
    if bias_node is not None:
        inputs = (*inputs, bias_node)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=list(inputs),
        output_shape=output_shape,
        inplace_input=None,
        params={
            "stride": stride_value,
            "padding": padding_value,
            "dilation": dilation_value,
            "groups": groups,
            "has_bias": bias_node is not None,
        },
    )


def _handle_conv2d_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
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
            raise RefBackendError("codegen conv2d expects bias to be a tensor")
    if isinstance(stride, torch.fx.Node) or isinstance(
        padding, torch.fx.Node
    ) or isinstance(dilation, torch.fx.Node):
        raise RefBackendError(
            "codegen conv2d expects constant stride, padding, and dilation"
        )
    if isinstance(transposed, torch.fx.Node) or transposed:
        raise RefBackendError("codegen conv2d does not support transposed")
    if isinstance(output_padding, torch.fx.Node) or output_padding not in (
        (0, 0),
        [0, 0],
        0,
    ):
        raise RefBackendError("codegen conv2d expects zero output padding")
    if isinstance(groups, torch.fx.Node):
        raise RefBackendError("codegen conv2d expects constant groups")
    if dtype_info.torch_dtype is not torch.float32:
        raise RefBackendError("codegen conv2d supports only torch.float32 tensors")
    if input_arg not in shapes or weight_arg not in shapes:
        raise _error_expected_tensor("conv2d")
    if dtypes[input_arg] is not torch.float32 or dtypes[weight_arg] is not torch.float32:
        raise RefBackendError("codegen conv2d supports only torch.float32 tensors")
    if bias_node is not None:
        if bias_node not in shapes:
            raise _error_expected_tensor("conv2d")
        if dtypes[bias_node] is not torch.float32:
            raise RefBackendError("codegen conv2d supports only torch.float32 tensors")
    input_shape = shapes[input_arg]
    weight_shape = shapes[weight_arg]
    if bias_node is not None:
        bias_shape = shapes[bias_node]
        if len(bias_shape) != 1 or bias_shape[0] != weight_shape[0]:
            raise RefBackendError(
                "codegen conv2d expects bias shape to match output channels"
            )
    if len(weight_shape) != 4:
        raise RefBackendError("codegen conv2d requires 4D weight tensors")
    if len(input_shape) not in (3, 4):
        raise RefBackendError("codegen conv2d requires 3D or 4D input tensors")
    stride_pair = _normalize_param(normalize_int_or_pair, "stride", stride)
    dilation_pair = _normalize_param(normalize_int_or_pair, "dilation", dilation)
    padding_value = _normalize_param(
        normalize_padding, "padding", padding, 2, allow_strings=("same", "valid")
    )
    if isinstance(padding_value, str):
        if padding_value == "valid":
            padding_pair = (0, 0)
            output_shape = _conv2d_output_shape_from_shapes(
                input_shape,
                weight_shape,
                stride_pair,
                padding_pair,
                dilation_pair,
                groups,
            )
        else:
            has_batch, out_channels = _conv2d_validate_channels(
                input_shape, weight_shape, groups
            )
            padding_pair, (out_h, out_w) = _conv2d_same_padding(
                input_shape, weight_shape, stride_pair, dilation_pair
            )
            if has_batch:
                output_shape = (input_shape[0], out_channels, out_h, out_w)
            else:
                output_shape = (out_channels, out_h, out_w)
    else:
        padding_pair = padding_value
        output_shape = _conv2d_output_shape_from_shapes(
            input_shape,
            weight_shape,
            stride_pair,
            padding_pair,
            dilation_pair,
            groups,
        )
    if (
        stride_pair[0] <= 0
        or stride_pair[1] <= 0
        or dilation_pair[0] <= 0
        or dilation_pair[1] <= 0
        or padding_pair[0] < 0
        or padding_pair[1] < 0
    ):
        raise RefBackendError(
            "codegen conv2d expects stride and dilation to be positive and padding to be non-negative"
        )
    if not isinstance(groups, int) or groups <= 0:
        raise RefBackendError("codegen conv2d requires positive groups")
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    inputs = (input_arg, weight_arg)
    if bias_node is not None:
        inputs = (*inputs, bias_node)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=list(inputs),
        output_shape=output_shape,
        inplace_input=None,
        params={
            "stride": stride_pair,
            "padding": padding_pair,
            "dilation": dilation_pair,
            "groups": groups,
            "has_bias": bias_node is not None,
        },
    )


def _handle_pool1d_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
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
        raise RefBackendError(
            f"codegen {op_spec.name} supports only torch.float32 tensors"
        )
    if dtypes[input_arg] is not torch.float32:
        raise RefBackendError(
            f"codegen {op_spec.name} supports only torch.float32 tensors"
        )
    if isinstance(kernel_size, torch.fx.Node) or isinstance(
        padding, torch.fx.Node
    ) or isinstance(ceil_mode, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
        )
    if stride is not None and isinstance(stride, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant stride values"
        )
    if isinstance(dilation, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant dilation values"
        )
    if isinstance(count_include_pad, torch.fx.Node) or isinstance(
        divisor_override, torch.fx.Node
    ):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant pooling options"
        )
    input_shape = shapes[input_arg]
    if len(input_shape) != 3:
        raise RefBackendError(
            f"codegen {op_spec.name} requires 3D input tensors"
        )
    if not _is_contiguous(input_shape, strides[input_arg]):
        raise RefBackendError(
            f"codegen {op_spec.name} requires contiguous input tensors"
        )
    if op_spec.name == "adaptive_avg_pool1d":
        if isinstance(output_size, torch.fx.Node):
            raise RefBackendError(
                "codegen adaptive_avg_pool1d expects output_size to be an int"
            )
        if isinstance(output_size, (tuple, list)):
            if len(output_size) != 1:
                raise RefBackendError(
                    "codegen adaptive_avg_pool1d expects a single output size"
                )
            output_size = output_size[0]
        if not isinstance(output_size, int):
            raise RefBackendError(
                "codegen adaptive_avg_pool1d expects output_size to be an int"
            )
        if output_size <= 0:
            raise RefBackendError(
                "codegen adaptive_avg_pool1d expects output_size to be positive"
            )
        in_l = input_shape[2]
        if in_l % output_size != 0:
            raise RefBackendError(
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
        raise RefBackendError(
            f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
        )
    if ceil_mode:
        raise RefBackendError(
            f"codegen {op_spec.name} does not support ceil_mode"
        )
    if not isinstance(count_include_pad, bool):
        raise RefBackendError(
            f"codegen {op_spec.name} expects count_include_pad to be a bool"
        )
    if divisor_override is not None:
        if not isinstance(divisor_override, int) or divisor_override <= 0:
            raise RefBackendError(
                f"codegen {op_spec.name} expects divisor_override to be a positive int"
            )
    output_shape = _pool1d_output_shape_from_shapes(
        input_shape,
        kernel_value,
        stride_value,
        padding_value,
        dilation_value,
    )
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        inplace_input=None,
        params={
            "kernel_size": kernel_value,
            "stride": stride_value,
            "padding": padding_value,
            "dilation": dilation_value,
            "ceil_mode": bool(ceil_mode),
            "count_include_pad": count_include_pad,
            "divisor_override": divisor_override,
        },
    )


def _handle_pool2d_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
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
        raise RefBackendError(
            f"codegen {op_spec.name} supports only torch.float32 tensors"
        )
    if dtypes[input_arg] is not torch.float32:
        raise RefBackendError(
            f"codegen {op_spec.name} supports only torch.float32 tensors"
        )
    if isinstance(kernel_size, torch.fx.Node) or isinstance(
        padding, torch.fx.Node
    ) or isinstance(ceil_mode, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
        )
    if stride is not None and isinstance(stride, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant stride values"
        )
    if isinstance(dilation, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant dilation values"
        )
    if isinstance(count_include_pad, torch.fx.Node) or isinstance(
        divisor_override, torch.fx.Node
    ):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant pooling options"
        )
    input_shape = shapes[input_arg]
    if len(input_shape) != 4:
        raise RefBackendError(
            f"codegen {op_spec.name} requires 4D input tensors"
        )
    if not _is_contiguous(input_shape, strides[input_arg]):
        raise RefBackendError(
            f"codegen {op_spec.name} requires contiguous input tensors"
        )
    if op_spec.name == "adaptive_avg_pool2d":
        if isinstance(output_size, torch.fx.Node):
            raise RefBackendError(
                "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
            )
        if isinstance(output_size, torch.Size):
            output_size = tuple(output_size)
        if isinstance(output_size, int):
            output_pair = (output_size, output_size)
        elif isinstance(output_size, (tuple, list)):
            if len(output_size) != 2:
                raise RefBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to have two values"
                )
            output_pair = tuple(output_size)
        else:
            raise RefBackendError(
                "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
            )
        if not all(isinstance(item, int) for item in output_pair):
            raise RefBackendError(
                "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
            )
        if output_pair[0] <= 0 or output_pair[1] <= 0:
            raise RefBackendError(
                "codegen adaptive_avg_pool2d expects output_size to be positive"
            )
        in_h, in_w = input_shape[2], input_shape[3]
        if in_h % output_pair[0] != 0 or in_w % output_pair[1] != 0:
            raise RefBackendError(
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
            stride_pair = _normalize_param(normalize_int_or_pair, "stride", stride)
        padding_pair = _normalize_param(normalize_int_or_pair, "padding", padding)
        dilation_pair = _normalize_param(normalize_int_or_pair, "dilation", dilation)
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
        raise RefBackendError(
            f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
        )
    if ceil_mode:
        raise RefBackendError(
            f"codegen {op_spec.name} does not support ceil_mode"
        )
    if not isinstance(count_include_pad, bool):
        raise RefBackendError(
            f"codegen {op_spec.name} expects count_include_pad to be a bool"
        )
    if divisor_override is not None:
        if not isinstance(divisor_override, int) or divisor_override <= 0:
            raise RefBackendError(
                f"codegen {op_spec.name} expects divisor_override to be a positive int"
            )
    output_shape = _pool2d_output_shape_from_shapes(
        input_shape,
        kernel_pair,
        stride_pair,
        padding_pair,
        dilation_pair,
    )
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
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


def _handle_col2im_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    (
        input_arg,
        output_size,
        kernel_size,
        dilation,
        padding,
        stride,
    ) = _parse_col2im_args(node)
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtype_info.torch_dtype is not torch.float32:
        raise RefBackendError(
            "codegen col2im supports only torch.float32 tensors"
        )
    if dtypes[input_arg] is not torch.float32:
        raise RefBackendError(
            "codegen col2im supports only torch.float32 tensors"
        )
    if (
        isinstance(output_size, torch.fx.Node)
        or isinstance(kernel_size, torch.fx.Node)
        or isinstance(dilation, torch.fx.Node)
        or isinstance(padding, torch.fx.Node)
        or isinstance(stride, torch.fx.Node)
    ):
        raise RefBackendError(
            "codegen col2im expects constant output_size, kernel_size, dilation, padding, and stride"
        )
    output_pair = _normalize_col2im_output_size(op_spec.name, output_size)
    kernel_pair = _normalize_param(
        normalize_int_or_pair, "kernel_size", kernel_size
    )
    dilation_pair = _normalize_param(
        normalize_int_or_pair, "dilation", dilation
    )
    padding_pair = _normalize_param(normalize_int_or_pair, "padding", padding)
    stride_pair = _normalize_param(normalize_int_or_pair, "stride", stride)
    if (
        output_pair[0] <= 0
        or output_pair[1] <= 0
        or kernel_pair[0] <= 0
        or kernel_pair[1] <= 0
        or dilation_pair[0] <= 0
        or dilation_pair[1] <= 0
        or stride_pair[0] <= 0
        or stride_pair[1] <= 0
        or padding_pair[0] < 0
        or padding_pair[1] < 0
    ):
        raise RefBackendError(
            "codegen col2im expects positive output_size, kernel_size, stride, and dilation with non-negative padding"
        )
    input_shape = shapes[input_arg]
    if len(input_shape) == 3:
        batch, col_channels, col_length = input_shape
        has_batch = True
    elif len(input_shape) == 2:
        col_channels, col_length = input_shape
        batch = 1
        has_batch = False
    else:
        raise RefBackendError("codegen col2im expects 2D or 3D input tensors")
    if not _is_contiguous(input_shape, strides[input_arg]):
        raise RefBackendError("codegen col2im requires contiguous input tensors")
    k_h, k_w = kernel_pair
    channels_divisor = k_h * k_w
    if channels_divisor <= 0 or col_channels % channels_divisor != 0:
        raise RefBackendError(
            "codegen col2im expects input channels divisible by kernel_size"
        )
    out_h, out_w = output_pair
    dil_h, dil_w = dilation_pair
    pad_h, pad_w = padding_pair
    stride_h, stride_w = stride_pair
    effective_kh = dil_h * (k_h - 1) + 1
    effective_kw = dil_w * (k_w - 1) + 1
    numerator_h = out_h + 2 * pad_h - effective_kh
    numerator_w = out_w + 2 * pad_w - effective_kw
    if (
        numerator_h < 0
        or numerator_w < 0
        or numerator_h % stride_h != 0
        or numerator_w % stride_w != 0
    ):
        raise RefBackendError(
            "codegen col2im expects output_size to be compatible with kernel_size, dilation, padding, and stride"
        )
    out_blocks_h = numerator_h // stride_h + 1
    out_blocks_w = numerator_w // stride_w + 1
    expected_length = out_blocks_h * out_blocks_w
    if col_length != expected_length:
        raise RefBackendError(
            "codegen col2im expects input length to match output_size and stride"
        )
    channels = col_channels // channels_divisor
    if has_batch:
        output_shape = (batch, channels, out_h, out_w)
    else:
        output_shape = (channels, out_h, out_w)
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        inplace_input=None,
        params={
            "output_size": output_pair,
            "kernel_size": kernel_pair,
            "dilation": dilation_pair,
            "padding": padding_pair,
            "stride": stride_pair,
            "out_blocks_h": out_blocks_h,
            "out_blocks_w": out_blocks_w,
        },
    )


def _handle_to_copy_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if not node.args:
        raise RefBackendError("codegen _to_copy expects one input")
    if len(node.args) != 1:
        raise RefBackendError("codegen _to_copy expects only self as positional arg")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor("_to_copy")
    dtype = node.kwargs.get("dtype")
    layout = node.kwargs.get("layout")
    device = node.kwargs.get("device")
    pin_memory = node.kwargs.get("pin_memory")
    non_blocking = node.kwargs.get("non_blocking", False)
    memory_format = node.kwargs.get("memory_format")
    extra = set(node.kwargs) - {
        "dtype",
        "layout",
        "device",
        "pin_memory",
        "non_blocking",
        "memory_format",
    }
    if extra:
        raise RefBackendError(
            f"codegen _to_copy got unexpected kwargs: {sorted(extra)}"
        )
    if isinstance(dtype, torch.fx.Node):
        raise RefBackendError("codegen _to_copy expects dtype to be a constant")
    if dtype is not None and dtype is not dtypes[input_arg]:
        raise RefBackendError("codegen _to_copy does not support dtype conversion")
    if layout is not None or device is not None or pin_memory is not None:
        raise RefBackendError("codegen _to_copy does not support layout/device moves")
    if isinstance(non_blocking, torch.fx.Node) or non_blocking not in (False, 0):
        raise RefBackendError(
            "codegen _to_copy expects non_blocking to be False"
        )
    if memory_format is not None:
        raise RefBackendError("codegen _to_copy does not support memory_format")
    output_shape = shapes[input_arg]
    shapes[node] = output_shape
    dtypes[node] = dtypes[input_arg]
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        inplace_input=None,
        params={},
    )


def _handle_batch_norm_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    has_training_flag = op_spec.name == "_native_batch_norm_legit"
    expected_inputs = 8 if has_training_flag else 7
    if len(node.args) < expected_inputs:
        raise RefBackendError(
            f"codegen {op_spec.name} expects {expected_inputs} inputs"
        )
    if node.kwargs:
        raise RefBackendError(
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
            raise RefBackendError(
                f"codegen {op_spec.name} expects constant training flag"
            )
        if training not in (False, 0):
            raise RefBackendError(
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
        raise RefBackendError(f"codegen {op_spec.name} supports only torch.float32")
    if dtypes[input_arg] is not torch.float32:
        raise RefBackendError(f"codegen {op_spec.name} supports only torch.float32")
    input_shape = shapes[input_arg]
    if len(input_shape) < 2:
        raise RefBackendError(
            f"codegen {op_spec.name} expects at least 2D inputs"
        )
    if not _is_contiguous(input_shape, strides[input_arg]):
        raise RefBackendError(
            f"codegen {op_spec.name} requires contiguous input"
        )
    channels = input_shape[1]
    for stat_arg, name in (
        (running_mean, "running_mean"),
        (running_var, "running_var"),
    ):
        stat_shape = shapes[stat_arg]
        if stat_shape != (channels,):
            raise RefBackendError(
                f"codegen {op_spec.name} expects {name} shape to match channels"
            )
        if not _is_contiguous(stat_shape, strides[stat_arg]):
            raise RefBackendError(
                f"codegen {op_spec.name} requires contiguous stats"
            )
        if dtypes[stat_arg] is not torch.float32:
            raise RefBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
            )
    if weight_node is not None:
        if shapes[weight_node] != (channels,):
            raise RefBackendError(
                f"codegen {op_spec.name} expects weight shape to match channels"
            )
        if dtypes[weight_node] is not torch.float32:
            raise RefBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
            )
    if bias_node is not None:
        if shapes[bias_node] != (channels,):
            raise RefBackendError(
                f"codegen {op_spec.name} expects bias shape to match channels"
            )
        if dtypes[bias_node] is not torch.float32:
            raise RefBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
            )
    if isinstance(momentum, torch.fx.Node) or isinstance(eps, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} expects constant momentum and eps"
        )
    try:
        eps_value = float(eps)
    except (TypeError, ValueError) as exc:
        raise RefBackendError(
            f"codegen {op_spec.name} expects eps to be a float"
        ) from exc
    shapes[node] = input_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(input_shape)
    inputs = [input_arg, running_mean, running_var]
    if weight_node is not None:
        inputs.append(weight_node)
    if bias_node is not None:
        inputs.append(bias_node)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=inputs,
        output_shape=input_shape,
        inplace_input=None,
        params={
            "eps": eps_value,
            "has_weight": weight_node is not None,
            "has_bias": bias_node is not None,
        },
    )


def _handle_pdist_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if not node.args:
        raise RefBackendError("codegen _pdist_forward expects one input")
    if len(node.args) > 2:
        raise RefBackendError("codegen _pdist_forward expects at most two inputs")
    if node.kwargs:
        raise RefBackendError("codegen _pdist_forward expects positional args only")
    input_arg = node.args[0]
    p_value = node.args[1] if len(node.args) > 1 else 2.0
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtype_info.torch_dtype is not torch.float32:
        raise RefBackendError("codegen _pdist_forward supports only torch.float32")
    if dtypes[input_arg] is not torch.float32:
        raise RefBackendError("codegen _pdist_forward supports only torch.float32")
    if isinstance(p_value, torch.fx.Node):
        raise RefBackendError("codegen _pdist_forward expects constant p value")
    try:
        p_value = float(p_value)
    except (TypeError, ValueError) as exc:
        raise RefBackendError("codegen _pdist_forward expects p to be a float") from exc
    if p_value != 2.0:
        raise RefBackendError("codegen _pdist_forward supports only p=2")
    input_shape = shapes[input_arg]
    if len(input_shape) != 2:
        raise RefBackendError("codegen _pdist_forward expects a 2D input tensor")
    if not _is_contiguous(input_shape, strides[input_arg]):
        raise RefBackendError("codegen _pdist_forward requires contiguous input")
    n = input_shape[0]
    output_len = n * (n - 1) // 2
    output_shape = (output_len,)
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        inplace_input=None,
        params={},
    )


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
        raise RefBackendError(
            f"codegen {op_name} expects inputs to share the graph dtype"
        )
    _validate_addmm_like_scalars(op_name, dtype_info.torch_dtype, alpha, beta)
    output_shape = _infer_output_shape(op_spec, input_shapes)
    if op_spec.kind == "addr" and inplace_input is not None:
        input_shape = input_shapes[inplace_input]
        if len(input_shape) != 2:
            raise RefBackendError(
                "codegen addr expects 2D input and 1D vectors"
            )
        if input_shape != output_shape:
            raise RefBackendError(
                "codegen addr expects input shape to match outer product output"
            )
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    if inplace_input is not None:
        strides[node] = strides[input_nodes[inplace_input]]
    else:
        strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=list(input_nodes),
        output_shape=output_shape,
        inplace_input=inplace_input,
        params={"alpha": alpha, "beta": beta},
    )


def _handle_diagonal_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if not node.args:
        raise RefBackendError(f"codegen {op_spec.name} expects one input")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    offset, dim1, dim2 = _parse_diagonal_args(
        op_spec.name, node, shapes[input_arg]
    )
    output_shape = _infer_diagonal_output_shape(
        shapes[input_arg], offset, dim1, dim2
    )
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        params={"offset": offset, "dim1": dim1, "dim2": dim2},
    )


def _handle_softmax_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if not node.args:
        raise RefBackendError(f"codegen {op_spec.name} expects one input")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtype_info.torch_dtype is not torch.float32:
        raise RefBackendError(
            f"codegen {op_spec.name} supports only torch.float32 tensors"
        )
    if dtypes[input_arg] is not torch.float32:
        raise RefBackendError(
            f"codegen {op_spec.name} supports only torch.float32 tensors"
        )
    dim, dtype = _parse_softmax_args(op_spec.name, node, shapes[input_arg])
    if dtype is not None and dtype is not torch.float32:
        raise RefBackendError(
            f"codegen {op_spec.name} expects dtype to be torch.float32 or None"
        )
    output_shape = shapes[input_arg]
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        inplace_input=None,
        params={"dim": dim},
    )

def _handle_embedding_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    weight, indices, padding_idx, scale_grad_by_freq, sparse = (
        _parse_embedding_args(node)
    )
    if scale_grad_by_freq or sparse:
        raise RefBackendError(
            "codegen embedding supports only scale_grad_by_freq=False and sparse=False"
        )
    if not isinstance(weight, torch.fx.Node) or weight not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if not isinstance(indices, torch.fx.Node) or indices not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[weight] is not dtype_info.torch_dtype:
        raise RefBackendError(
            "codegen embedding expects weight to match the graph dtype"
        )
    if dtypes[indices] not in _EMBEDDING_INDEX_DTYPES:
        raise RefBackendError(
            "codegen embedding expects indices to have dtype torch.int32 or torch.int64"
        )
    weight_shape = shapes[weight]
    if len(weight_shape) != 2:
        raise RefBackendError("codegen embedding expects 2D weight tensor")
    if padding_idx != -1:
        if padding_idx < 0 or padding_idx >= weight_shape[0]:
            raise RefBackendError(
                "codegen embedding expects padding_idx to be -1 or within num_embeddings"
            )
    indices_shape = shapes[indices]
    output_shape = tuple(indices_shape) + (weight_shape[1],)
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[weight, indices],
        output_shape=output_shape,
        inplace_input=None,
        params={"padding_idx": padding_idx},
    )

def _handle_cumsum_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if not node.args:
        raise RefBackendError(f"codegen {op_spec.name} expects one input")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtype_info.torch_dtype is torch.bool:
        raise RefBackendError("codegen cumsum does not support torch.bool tensors")
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    dim, dtype_override = _parse_cumsum_args(
        op_spec.name, node, shapes[input_arg]
    )
    if dtype_override is not None and dtype_override is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_spec.name} expects dtype to match the graph dtype"
        )
    output_shape = shapes[input_arg]
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        inplace_input=None,
        params={"dim": dim},
    )


def _handle_constant_pad_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if not node.args:
        raise RefBackendError(
            "codegen constant_pad_nd expects at least one argument"
        )
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise RefBackendError(
            "codegen constant_pad_nd expects inputs to share the graph dtype"
        )
    pad = None
    value = 0
    if len(node.args) > 1:
        pad = node.args[1]
    if len(node.args) > 2:
        value = node.args[2]
    if len(node.args) > 3:
        raise RefBackendError(
            "codegen constant_pad_nd expects at most three positional arguments"
        )
    extra_kwargs = set(node.kwargs) - {"pad", "value"}
    if extra_kwargs:
        raise RefBackendError(
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
        raise RefBackendError("codegen constant_pad_nd expects a pad argument")
    if isinstance(pad, torch.fx.Node):
        raise RefBackendError(
            "codegen constant_pad_nd expects pad to be a constant list"
        )
    if not isinstance(pad, (tuple, list)):
        raise RefBackendError(
            "codegen constant_pad_nd expects pad to be a list or tuple"
        )
    if len(pad) % 2 != 0:
        raise RefBackendError(
            "codegen constant_pad_nd expects pad to have an even number of values"
        )
    pad_values = []
    for item in pad:
        if isinstance(item, numbers.Real) and not float(item).is_integer():
            raise RefBackendError(
                "codegen constant_pad_nd expects pad values to be integers"
            )
        try:
            pad_values.append(int(operator.index(item)))
        except TypeError:
            try:
                pad_values.append(int(item))
            except (TypeError, ValueError) as exc:
                raise RefBackendError(
                    "codegen constant_pad_nd expects pad values to be integers"
                ) from exc
    input_shape = shapes[input_arg]
    rank = len(input_shape)
    if len(pad_values) > 2 * rank:
        raise RefBackendError(
            "codegen constant_pad_nd expects pad to have at most 2 * input rank values"
        )
    pad_before = [0] * rank
    pad_after = [0] * rank
    for idx in range(len(pad_values) // 2):
        dim = rank - 1 - idx
        pad_before[dim] = pad_values[2 * idx]
        pad_after[dim] = pad_values[2 * idx + 1]
    output_shape: List[int] = []
    for size, before, after in zip(input_shape, pad_before, pad_after):
        new_size = size + before + after
        if new_size < 0:
            raise RefBackendError(
                "codegen constant_pad_nd expects non-negative output sizes"
            )
        output_shape.append(new_size)
    if isinstance(value, torch.fx.Node):
        raise RefBackendError(
            "codegen constant_pad_nd expects a constant padding value"
        )
    value = _normalize_scalar_value(op_spec.name, value)
    shapes[node] = tuple(output_shape)
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=tuple(output_shape),
        params={
            "pad_before": tuple(pad_before),
            "pad_after": tuple(pad_after),
            "value": value,
        },
    )


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
        raise RefBackendError(
            "codegen gather expects input to match the graph dtype"
        )
    if dtypes[index] not in _EMBEDDING_INDEX_DTYPES:
        raise RefBackendError(
            "codegen gather expects index dtype to be torch.int32 or torch.int64"
        )
    if isinstance(sparse_grad, torch.fx.Node):
        raise RefBackendError("codegen gather expects sparse_grad to be False")
    if sparse_grad not in (False, 0, None):
        raise RefBackendError("codegen gather supports only sparse_grad=False")
    input_shape = shapes[input_arg]
    index_shape = shapes[index]
    if not input_shape:
        raise RefBackendError("codegen gather expects input to have at least 1 dimension")
    if len(index_shape) != len(input_shape):
        raise RefBackendError(
            "codegen gather expects index to have the same rank as input"
        )
    dim_value = _parse_constant_int(op_spec.name, "dim", dim)
    if dim_value < 0:
        dim_value += len(input_shape)
    if dim_value < 0 or dim_value >= len(input_shape):
        raise RefBackendError("codegen gather dim is out of range")
    for idx, (input_dim, index_dim) in enumerate(
        zip(input_shape, index_shape)
    ):
        if idx == dim_value:
            continue
        if input_dim != index_dim:
            raise RefBackendError(
                "codegen gather expects index shape to match input shape"
            )
    output_shape = index_shape
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg, index],
        output_shape=output_shape,
        inplace_input=None,
        params={"dim": dim_value},
    )


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
        raise RefBackendError(f"codegen {op_spec.name} expects inputs")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    value = None
    if len(node.args) > 2:
        raise RefBackendError(
            f"codegen {op_spec.name} expects a tensor and scalar value"
        )
    if len(node.args) > 1:
        value = node.args[1]
    if "value" in node.kwargs:
        if value is not None:
            raise RefBackendError(
                f"codegen {op_spec.name} expects a single scalar value"
            )
        value = node.kwargs["value"]
    if op_spec.name == "full_like" and "fill_value" in node.kwargs:
        if value is not None:
            raise RefBackendError(
                f"codegen {op_spec.name} expects a single scalar value"
            )
        value = node.kwargs["fill_value"]
    if op_spec.name == "full_like":
        allowed = {
            "value",
            "fill_value",
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "memory_format",
        }
        extra = set(node.kwargs) - allowed
        if extra:
            raise RefBackendError(
                f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
            )
        dtype = node.kwargs.get("dtype")
        if dtype is not None and dtype is not dtype_info.torch_dtype:
            raise RefBackendError(
                f"codegen {op_spec.name} expects dtype to match the graph dtype"
            )
        for name in ("layout", "device", "memory_format"):
            value_kw = node.kwargs.get(name)
            if value_kw is not None:
                raise RefBackendError(
                    f"codegen {op_spec.name} expects {name} to be None"
                )
        pin_memory = node.kwargs.get("pin_memory")
        if pin_memory not in (None, False):
            raise RefBackendError(
                f"codegen {op_spec.name} expects pin_memory to be False"
            )
    elif node.kwargs and set(node.kwargs) != {"value"}:
        raise RefBackendError(
            f"codegen {op_spec.name} expects only 'value' as a keyword argument"
        )
    if value is None:
        raise RefBackendError(
            f"codegen {op_spec.name} expects a scalar value"
        )
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    scalar_value = _normalize_scalar_value(op_spec.name, value)
    output_shape = shapes[input_arg]
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    if inplace_input is not None:
        strides[node] = strides[input_arg]
    else:
        strides[node] = _contiguous_strides(output_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=output_shape,
        inplace_input=inplace_input,
        params={"value": scalar_value},
    )


def _parse_arange_dtype(
    op_name: str,
    dtype: torch.dtype | None,
    dtype_info: _CodegenDType | None,
) -> _CodegenDType:
    if dtype is None:
        if dtype_info is not None:
            dtype = dtype_info.torch_dtype
        else:
            dtype = torch.get_default_dtype()
    if dtype is torch.bool:
        raise RefBackendError(
            f"codegen {op_name} supports only numeric dtypes"
        )
    dtype_spec = _CODEGEN_DTYPES.get(dtype)
    if dtype_spec is None:
        raise RefBackendError(
            f"codegen {op_name} supports only torch.float32, torch.int8, or torch.int32"
        )
    if dtype_info is not None and dtype_spec.torch_dtype is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_name} expects dtype to match the graph dtype"
        )
    return dtype_spec


def _compute_arange_size(start: float, end: float, step: float) -> int:
    if step == 0:
        raise RefBackendError("codegen arange expects step to be non-zero")
    delta = (end - start) / step
    size = int(math.ceil(delta))
    return max(size, 0)


def _handle_arange_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType | None,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    scalar_values: Dict[torch.fx.Node, object],
) -> Tuple[_OpNode, _CodegenDType]:
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
        raise RefBackendError(
            f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
        )
    if node.kwargs.get("layout") is not None:
        raise RefBackendError(
            f"codegen {op_spec.name} expects layout to be None"
        )
    device = node.kwargs.get("device")
    if device is not None and device != "cpu" and device != torch.device("cpu"):
        raise RefBackendError(
            f"codegen {op_spec.name} expects device to be None or cpu"
        )
    pin_memory = node.kwargs.get("pin_memory")
    if pin_memory not in (None, False):
        raise RefBackendError(
            f"codegen {op_spec.name} expects pin_memory to be False"
        )
    start_arg = None
    end_arg = None
    step_arg = None
    if node.args:
        if len(node.args) > 3:
            raise RefBackendError(
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
        raise RefBackendError(
            f"codegen {op_spec.name} expects start and end arguments"
        )
    if step_arg is None:
        step_arg = 1
    start = _resolve_scalar_arg(op_spec.name, start_arg, scalar_values)
    end = _resolve_scalar_arg(op_spec.name, end_arg, scalar_values)
    step = _resolve_scalar_arg(op_spec.name, step_arg, scalar_values)
    dtype_spec = _parse_arange_dtype(
        op_spec.name, node.kwargs.get("dtype"), dtype_info
    )
    output_size = _compute_arange_size(
        float(start), float(end), float(step)
    )
    output_shape = (output_size,)
    shapes[node] = output_shape
    dtypes[node] = dtype_spec.torch_dtype
    strides[node] = _contiguous_strides(output_shape)
    return (
        _OpNode(
            node=node,
            spec=op_spec,
            inputs=[],
            output_shape=output_shape,
            params={"start": start, "step": step},
        ),
        dtype_spec,
    )


def _handle_view_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if not node.args:
        raise RefBackendError(f"codegen {op_spec.name} expects one input")
    input_arg = node.args[0]
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_spec.name} expects inputs to share the graph dtype"
        )
    if op_spec.name == "as_strided":
        if len(node.args) > 4:
            raise RefBackendError(
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
                raise RefBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        if size is None or stride is None:
            raise RefBackendError("codegen as_strided expects size and stride")
        if storage_offset is None:
            storage_offset = 0
        if isinstance(size, torch.fx.Node) or isinstance(stride, torch.fx.Node):
            raise RefBackendError("codegen as_strided expects size/stride to be constants")
        if isinstance(storage_offset, torch.fx.Node):
            raise RefBackendError(
                "codegen as_strided expects storage_offset to be an int"
            )
        if not isinstance(size, (tuple, list)):
            raise RefBackendError("codegen as_strided expects size to be a list")
        if not isinstance(stride, (tuple, list)):
            raise RefBackendError("codegen as_strided expects stride to be a list")
        size_tuple = tuple(int(operator.index(dim)) for dim in size)
        stride_tuple = tuple(int(operator.index(dim)) for dim in stride)
        if len(size_tuple) != len(stride_tuple):
            raise RefBackendError(
                "codegen as_strided expects size and stride to match length"
            )
        storage_offset_value = int(operator.index(storage_offset))
        if storage_offset_value < 0:
            raise RefBackendError(
                "codegen as_strided expects storage_offset to be non-negative"
            )
        output_shape = size_tuple
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=output_shape,
            params={
                "view_strides": stride_tuple,
                "storage_offset": storage_offset_value,
            },
        )
    if op_spec.name == "squeeze":
        input_shape = shapes[input_arg]
        input_strides = strides[input_arg]
        if node.target is torch.ops.aten.squeeze.dim:
            if len(node.args) > 2:
                raise RefBackendError(
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
                    raise RefBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if dim is None:
                raise RefBackendError("codegen squeeze expects dim to be an int")
            dim_value = _parse_constant_int(op_spec.name, "dim", dim)
            if dim_value < 0:
                dim_value += len(input_shape)
            if dim_value < 0 or dim_value >= len(input_shape):
                raise RefBackendError("codegen squeeze dim is out of range")
            remove_dims = {
                dim_value
            } if input_shape[dim_value] == 1 else set()
        else:
            if len(node.args) > 2:
                raise RefBackendError(
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
                    raise RefBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if dims is None:
                raise RefBackendError("codegen squeeze expects dim to be a list")
            if isinstance(dims, torch.fx.Node) or not isinstance(dims, (tuple, list)):
                raise RefBackendError("codegen squeeze expects dim to be a list")
            dim_values = []
            for dim in dims:
                dim_value = _parse_constant_int(op_spec.name, "dim", dim)
                if dim_value < 0:
                    dim_value += len(input_shape)
                if dim_value < 0 or dim_value >= len(input_shape):
                    raise RefBackendError("codegen squeeze dim is out of range")
                dim_values.append(dim_value)
            remove_dims = {
                dim for dim in set(dim_values) if input_shape[dim] == 1
            }
        output_shape = tuple(
            size
            for dim, size in enumerate(input_shape)
            if dim not in remove_dims
        )
        view_strides = tuple(
            stride
            for dim, stride in enumerate(input_strides)
            if dim not in remove_dims
        )
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=output_shape,
            params={"view_strides": view_strides, "storage_offset": 0},
        )
    raise RefBackendError(f"Unsupported view op: {op_spec.name}")


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
                raise RefBackendError(
                    f"codegen {op_name} expects size values to be integers"
                ) from exc
    raise RefBackendError(f"codegen {op_name} expects size to be a sequence")


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
                raise RefBackendError(
                    f"codegen {op_name} expects stride values to be integers"
                ) from exc
    raise RefBackendError(f"codegen {op_name} expects stride to be a sequence")


def _handle_empty_strided_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if len(node.args) < 2:
        raise RefBackendError(
            f"codegen {op_spec.name} expects size and stride arguments"
        )
    if len(node.args) > 7:
        raise RefBackendError(
            f"codegen {op_spec.name} expects at most seven arguments"
        )
    size_arg, stride_arg = node.args[:2]
    if isinstance(size_arg, torch.fx.Node) or isinstance(
        stride_arg, torch.fx.Node
    ):
        raise RefBackendError(
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
        raise RefBackendError(
            f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
        )
    dtype_value = kwargs.get("dtype")
    if dtype_value is not None and dtype_value is not dtype_info.torch_dtype:
        raise RefBackendError(
            f"codegen {op_spec.name} expects dtype to match the graph dtype"
        )
    for name in ("layout", "device"):
        if kwargs.get(name) is not None:
            raise RefBackendError(
                f"codegen {op_spec.name} expects {name} to be None"
            )
    for name in ("pin_memory", "requires_grad"):
        if kwargs.get(name) not in (None, False):
            raise RefBackendError(
                f"codegen {op_spec.name} expects {name} to be False"
            )
    output_shape = _parse_resize_size(op_spec.name, size_arg)
    output_strides = _parse_empty_strided_stride(op_spec.name, stride_arg)
    if len(output_shape) != len(output_strides):
        raise RefBackendError(
            f"codegen {op_spec.name} expects size and stride to match length"
        )
    shapes[node] = output_shape
    dtypes[node] = dtype_info.torch_dtype
    strides[node] = output_strides
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[],
        output_shape=output_shape,
    )


def _handle_resize_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
    inplace_input: int | None,
) -> _OpNode:
    if node.kwargs:
        if (
            set(node.kwargs) != {"memory_format"}
            or node.kwargs["memory_format"] is not None
        ):
            raise RefBackendError(
                "codegen resize_ supports only memory_format=None"
            )
    if len(node.args) != 2:
        raise RefBackendError("codegen resize_ expects input and size arguments")
    input_arg, size_arg = node.args
    if not isinstance(input_arg, torch.fx.Node):
        raise _error_expected_tensor(op_spec.name)
    if input_arg not in shapes:
        raise _error_expected_tensor(op_spec.name)
    if dtypes[input_arg] is not dtype_info.torch_dtype:
        raise RefBackendError(
            "codegen resize_ expects inputs to share the graph dtype"
        )
    size = _parse_resize_size(op_spec.name, size_arg)
    input_shape = shapes[input_arg]
    if size != input_shape:
        raise RefBackendError(
            "codegen resize_ supports only size values that match the input shape"
        )
    shapes[node] = input_shape
    dtypes[node] = dtype_info.torch_dtype
    if inplace_input is not None:
        strides[node] = strides[input_arg]
    else:
        strides[node] = _contiguous_strides(input_shape)
    return _OpNode(
        node=node,
        spec=op_spec,
        inputs=[input_arg],
        output_shape=input_shape,
        inplace_input=inplace_input,
    )


def _parse_where_inputs(
    op_spec: _OpSpec,
    node: torch.fx.Node,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    scalar_values: Dict[torch.fx.Node, object],
) -> Tuple[List[torch.fx.Node], List[Tuple[int, ...]], Dict[str, object]]:
    if len(node.args) < 3:
        raise RefBackendError(f"codegen {op_spec.name} expects three inputs")
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
    input_iter = iter(example_inputs)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            try:
                example = next(input_iter)
            except StopIteration as exc:
                raise RefBackendError(
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
                    raise RefBackendError(
                        "codegen backend expects getitem to use positional args"
                    )
                if len(node.args) != 2:
                    raise RefBackendError(
                        "codegen backend expects getitem to have two inputs"
                    )
                source, index = node.args
                if not isinstance(source, torch.fx.Node):
                    raise RefBackendError(
                        "codegen backend expects getitem source to be a tensor op"
                    )
                if isinstance(index, torch.fx.Node) or index not in (0, 0.0):
                    raise RefBackendError(
                        "codegen backend supports only getitem[0]"
                    )
                if source not in shapes:
                    raise RefBackendError(
                        "codegen backend expects getitem source to be analyzed"
                    )
                if source.target not in {
                    torch.ops.aten._native_batch_norm_legit,
                    torch.ops.aten._native_batch_norm_legit.default,
                    torch.ops.aten._native_batch_norm_legit_no_training,
                    torch.ops.aten._native_batch_norm_legit_no_training.default,
                }:
                    raise RefBackendError(
                        "codegen backend supports getitem only for _native_batch_norm_legit* ops"
                    )
                alias_map[node] = source
                shapes[node] = shapes[source]
                strides[node] = strides[source]
                dtypes[node] = dtypes[source]
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
                }:
                    raise RefBackendError(f"Unsupported call_method: {node.target}")
                op_spec = SUPPORTED_OPS[node.target]
                inplace_input = None
            else:
                target_info = TARGET_REGISTRY.get(node.target)
                if target_info is None:
                    raise RefBackendError(f"Unsupported call_function: {node.target}")
                op_spec = target_info.op_spec
                inplace_input = target_info.inplace_arg_index
            if op_spec.kind == "arange":
                op_node, dtype_info = _handle_arange_node(
                    node,
                    op_spec,
                    dtype_info,
                    shapes,
                    strides,
                    dtypes,
                    scalar_values,
                )
                op_nodes.append(op_node)
                continue
            if dtype_info is None:
                raise RefBackendError(
                    "codegen backend requires at least one tensor input or a factory op dtype"
                )
            if op_spec.kind == "concat":
                op_nodes.append(
                    _handle_concat_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "pool1d":
                op_nodes.append(
                    _handle_pool1d_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "pool2d":
                op_nodes.append(
                    _handle_pool2d_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "col2im":
                op_nodes.append(
                    _handle_col2im_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "batch_norm":
                op_nodes.append(
                    _handle_batch_norm_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "pdist":
                op_nodes.append(
                    _handle_pdist_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "conv1d":
                op_nodes.append(
                    _handle_conv1d_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "conv2d":
                op_nodes.append(
                    _handle_conv2d_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            if op_spec.kind == "diagonal":
                op_nodes.append(
                    _handle_diagonal_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind in {"addmm", "addbmm", "addmv", "addr"}:
                op_nodes.append(
                    _handle_addmm_like_node(
                        node,
                        op_spec,
                        dtype_info,
                        shapes,
                        strides,
                        dtypes,
                        inplace_input,
                    )
                )
                continue
            elif op_spec.kind == "softmax":
                op_nodes.append(
                    _handle_softmax_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "flip":
                op_nodes.append(
                    _handle_flip_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "cumsum":
                op_nodes.append(
                    _handle_cumsum_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "pad":
                op_nodes.append(
                    _handle_constant_pad_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "gather":
                op_nodes.append(
                    _handle_gather_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "embedding":
                op_nodes.append(
                    _handle_embedding_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "view":
                op_nodes.append(
                    _handle_view_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "empty_strided":
                op_nodes.append(
                    _handle_empty_strided_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.name == "_to_copy":
                op_nodes.append(
                    _handle_to_copy_node(
                        node, op_spec, dtype_info, shapes, strides, dtypes
                    )
                )
                continue
            elif op_spec.kind == "fill":
                op_nodes.append(
                    _handle_fill_node(
                        node,
                        op_spec,
                        dtype_info,
                        shapes,
                        strides,
                        dtypes,
                        inplace_input,
                    )
                )
                continue
            elif op_spec.name == "resize_":
                op_nodes.append(
                    _handle_resize_node(
                        node,
                        op_spec,
                        dtype_info,
                        shapes,
                        strides,
                        dtypes,
                        inplace_input,
                    )
                )
                continue
            reduction_dims: Tuple[int, ...] | None = None
            keepdim = False
            reduce_all = False
            param_values: Dict[str, object] = {}
            input_nodes: List[torch.fx.Node] = []
            input_shapes: List[Tuple[int, ...]] = []
            out_arg: torch.fx.Node | None = None
            if op_spec.kind == "binary" and len(node.args) == 2:
                lhs, rhs = node.args
                if isinstance(lhs, torch.fx.Node) ^ isinstance(rhs, torch.fx.Node):
                    if node.kwargs:
                        raise RefBackendError(
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
                else:
                    if op_spec.kind in {"reduction", "arg_reduction"}:
                        if len(node.args) < 1:
                            raise RefBackendError(
                                f"codegen {op_spec.name} expects one input"
                            )
                        args_to_check = node.args[:1]
                    elif (
                        op_spec.kind == "unary"
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
                        if is_out_overload:
                            allowed_kwargs.add("out")
                        if node.kwargs and set(node.kwargs) - allowed_kwargs:
                            raise RefBackendError(
                                "codegen backend expects positional args only"
                            )
                        if op_spec.kind == "unary":
                            expected_arity = 1
                        elif op_spec.kind == "binary":
                            expected_arity = 2
                        elif op_spec.kind == "where":
                            expected_arity = 3
                        else:
                            expected_arity = 2
                        if is_out_overload:
                            if inplace_input is None:
                                raise RefBackendError(
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
                                    raise RefBackendError(
                                        f"codegen {op_spec.name} expects one input"
                                    )
                                if expected_arity == 2:
                                    raise RefBackendError(
                                        f"codegen {op_spec.name} expects exactly two inputs"
                                    )
                                raise RefBackendError(
                                    f"codegen {op_spec.name} expects exactly three inputs"
                                )
                            if out_arg is None:
                                raise RefBackendError(
                                    f"codegen {op_spec.name} expects out to be provided"
                                )
                        elif op_spec.name == "copy":
                            if len(node.args) not in {2, 3}:
                                raise RefBackendError(
                                    "codegen copy expects two inputs and optional non_blocking"
                                )
                        elif len(node.args) != expected_arity:
                            if expected_arity == 1:
                                raise RefBackendError(
                                    f"codegen {op_spec.name} expects one input"
                                )
                            if expected_arity == 2:
                                raise RefBackendError(
                                    f"codegen {op_spec.name} expects exactly two inputs"
                                )
                            raise RefBackendError(
                                f"codegen {op_spec.name} expects exactly three inputs"
                            )
                        if op_spec.name == "div":
                            rounding_mode = node.kwargs.get("rounding_mode")
                            if rounding_mode is not None:
                                raise RefBackendError(
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
                                raise RefBackendError(
                                    "codegen copy expects non_blocking to be False"
                                )
                        if op_spec.name == "copy":
                            args_to_check = node.args[:2]
                        else:
                            args_to_check = node.args
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
            else:
                if op_spec.kind in {"reduction", "arg_reduction"}:
                    if len(node.args) < 1:
                        raise RefBackendError(
                            f"codegen {op_spec.name} expects one input"
                        )
                    args_to_check = node.args[:1]
                elif (
                    op_spec.kind == "unary"
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
                    if is_out_overload:
                        allowed_kwargs.add("out")
                    if node.kwargs and set(node.kwargs) - allowed_kwargs:
                        raise RefBackendError(
                            "codegen backend expects positional args only"
                        )
                    if op_spec.kind == "unary":
                        expected_arity = 1
                    elif op_spec.kind == "binary":
                        expected_arity = 2
                    elif op_spec.kind == "where":
                        expected_arity = 3
                    else:
                        expected_arity = 2
                    if is_out_overload:
                        if inplace_input is None:
                            raise RefBackendError(
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
                                raise RefBackendError(
                                    f"codegen {op_spec.name} expects one input"
                                )
                            if expected_arity == 2:
                                raise RefBackendError(
                                    f"codegen {op_spec.name} expects exactly two inputs"
                                )
                            raise RefBackendError(
                                f"codegen {op_spec.name} expects exactly three inputs"
                            )
                        if out_arg is None:
                            raise RefBackendError(
                                f"codegen {op_spec.name} expects out to be provided"
                            )
                    elif op_spec.name == "copy":
                        if len(node.args) not in {2, 3}:
                            raise RefBackendError(
                                "codegen copy expects two inputs and optional non_blocking"
                            )
                    elif len(node.args) != expected_arity:
                        if expected_arity == 1:
                            raise RefBackendError(
                                f"codegen {op_spec.name} expects one input"
                            )
                        if expected_arity == 2:
                            raise RefBackendError(
                                f"codegen {op_spec.name} expects exactly two inputs"
                            )
                        raise RefBackendError(
                            f"codegen {op_spec.name} expects exactly three inputs"
                        )
                    if op_spec.name == "div":
                        rounding_mode = node.kwargs.get("rounding_mode")
                        if rounding_mode is not None:
                            raise RefBackendError(
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
                            raise RefBackendError(
                                "codegen copy expects non_blocking to be False"
                            )
                    if op_spec.name == "copy":
                        args_to_check = node.args[:2]
                    else:
                        args_to_check = node.args
                if op_spec.kind == "where":
                    (
                        input_nodes,
                        input_shapes,
                        where_params,
                    ) = _parse_where_inputs(
                        op_spec, node, shapes, scalar_values
                    )
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
            if op_spec.kind == "where":
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
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects integer tensors"
                    )
            if op_spec.name in _FLOAT_ONLY_UNARY_OPS:
                if dtype_info.torch_dtype is not torch.float32:
                    raise RefBackendError(
                        f"codegen {op_spec.name} supports only torch.float32 tensors"
                    )
                if any(dtype is not torch.float32 for dtype in input_dtypes):
                    raise RefBackendError(
                        f"codegen {op_spec.name} supports only torch.float32 tensors"
                    )
            if op_spec.name == "clamp" and dtype_info.torch_dtype is torch.bool:
                raise RefBackendError(
                    "codegen clamp supports only numeric tensors"
                )
            if (
                op_spec.name == "clamp"
                and dtype_info.torch_dtype in _INTEGER_CODEGEN_DTYPES
            ):
                for name in ("min_val", "max_val"):
                    value = param_values.get(name)
                    if value is None:
                        continue
                    if not float(value).is_integer():
                        raise RefBackendError(
                            "codegen clamp expects integer min/max for integer tensors"
                        )
            if op_spec.kind == "where":
                if input_dtypes[0] is not torch.bool:
                    raise RefBackendError(
                        "codegen where expects condition to be a boolean tensor"
                    )
                if any(
                    dtype is not dtype_info.torch_dtype for dtype in input_dtypes[1:]
                ):
                    raise RefBackendError(
                        "codegen where expects self and other to match the graph dtype"
                    )
            elif any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
                raise RefBackendError(
                    f"codegen {op_spec.name} expects inputs to share the graph dtype"
                )
            if op_spec.kind == "reduction":
                if op_spec.name == "norm":
                    if dtype_info.torch_dtype is not torch.float32:
                        raise RefBackendError(
                            "codegen norm supports only torch.float32 tensors"
                        )
                    (
                        reduction_dims,
                        keepdim,
                        reduce_all,
                        norm_p,
                    ) = _parse_norm_args(
                        op_spec.name, node, shape_input_shapes[0]
                    )
                    param_values["norm_p"] = norm_p
                else:
                    (
                        reduction_dims,
                        keepdim,
                        reduce_all,
                        unbiased,
                    ) = _parse_reduction_args(
                        op_spec.name, node, shape_input_shapes[0]
                    )
                    if unbiased is not None:
                        param_values["unbiased"] = unbiased
                    if (
                        op_spec.name == "var"
                        and dtype_info.torch_dtype is not torch.float32
                    ):
                        raise RefBackendError(
                            "codegen var supports only torch.float32 tensors"
                        )
                output_shape = _infer_reduction_output_shape(
                    shape_input_shapes[0],
                    reduction_dims,
                    keepdim,
                    reduce_all=reduce_all,
                )
            elif op_spec.kind == "arg_reduction":
                (
                    reduction_dims,
                    keepdim,
                    reduce_all,
                ) = _parse_argminmax_args(
                    op_spec.name, node, shape_input_shapes[0]
                )
                reduction_count = 1
                if reduce_all:
                    for size in shape_input_shapes[0]:
                        reduction_count *= size
                else:
                    for dim in reduction_dims:
                        reduction_count *= shape_input_shapes[0][dim]
                if reduction_count == 0:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects a non-empty reduction dimension"
                    )
                output_shape = _infer_reduction_output_shape(
                    shape_input_shapes[0],
                    reduction_dims,
                    keepdim,
                    reduce_all=reduce_all,
                )
            else:
                output_shape = _infer_output_shape(op_spec, shape_input_shapes)
            if out_arg is not None and shapes[out_arg] != output_shape:
                raise RefBackendError(
                    f"codegen {op_spec.name} expects out to match output shape"
                )
            if op_spec.kind in {"reduction", "arg_reduction"}:
                param_values["reduce_all"] = reduce_all
            shapes[node] = output_shape
            if op_spec.kind == "arg_reduction":
                dtypes[node] = torch.int64
            else:
                dtypes[node] = dtype_info.torch_dtype
            if inplace_input is not None:
                strides[node] = strides[input_nodes[inplace_input]]
            else:
                strides[node] = _contiguous_strides(output_shape)
            op_nodes.append(
                _OpNode(
                    node=node,
                    spec=op_spec,
                    inputs=list(input_nodes),
                    output_shape=output_shape,
                    inplace_input=inplace_input,
                    reduction_dims=reduction_dims,
                    keepdim=keepdim,
                    params=param_values,
                )
            )
            continue
        if node.op == "output":
            output_node = node
            continue
        raise RefBackendError(f"Unsupported node op: {node.op}")

    try:
        next(input_iter)
    except StopIteration:
        pass
    else:
        raise RefBackendError(
            "codegen backend expects example inputs to match placeholder count"
        )

    if not op_nodes:
        raise RefBackendError("codegen backend requires at least one operation")
    if output_node is None:
        raise RefBackendError("codegen backend requires an output node")
    if not tensor_placeholders and dtype_info is None:
        raise RefBackendError(
            "codegen backend requires at least one tensor input or a factory op dtype"
        )
    if dtype_info is None:
        raise RefBackendError("codegen backend could not infer a graph dtype")
    output_value, output_structure = _unwrap_output_node(output_node)
    while output_value in alias_map:
        output_value = alias_map[output_value]
    if output_value not in shapes:
        raise RefBackendError("codegen backend expects a single output node")
    if output_value not in {op.node for op in op_nodes}:
        raise RefBackendError("codegen backend output must be an operator result")

    output_op = next(op for op in op_nodes if op.node is output_value)
    for op_node in op_nodes:
        if (
            op_node.spec.kind == "empty_strided"
            and op_node.node is not output_value
            and not _is_contiguous(op_node.output_shape, strides[op_node.node])
        ):
            raise RefBackendError(
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
    )


def _compile_generic_library(graph: _GenericGraph) -> _GenericLibrary:
    source = _write_generic_source(graph)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _LIBRARY_CACHE.get(digest)
    if cached is not None:
        return cached

    build_dir = Path(tempfile.mkdtemp(prefix="codegen_generic_"))
    c_path = build_dir / "ref_codegen_generic.c"
    c_path.write_text(source, encoding="utf-8")

    compiler = ccompiler.new_compiler()
    distutils_sysconfig.customize_compiler(compiler)
    compile_args: List[str]
    if compiler.compiler_type == "msvc":
        compile_args = ["/O2"]
    else:
        compile_args = ["-O3", "-fPIC"]
    objects = compiler.compile(
        [str(c_path)],
        output_dir=str(build_dir),
        include_dirs=[str(_C_SRC_DIR)],
        extra_postargs=compile_args,
    )
    lib_name = "ref_codegen_generic"
    compiler.link_shared_lib(objects, lib_name, output_dir=str(build_dir))
    so_path = build_dir / compiler.library_filename(lib_name, lib_type="shared")

    import ctypes

    lib = ctypes.CDLL(str(so_path))
    argtypes = [ctypes.c_void_p for _ in graph.tensor_placeholders]
    argtypes.append(ctypes.c_void_p)
    entry_name = f"ref_codegen_main_{graph.dtype.suffix}"
    getattr(lib, entry_name).argtypes = argtypes
    getattr(lib, entry_name).restype = None

    input_shapes = tuple(graph.shapes[node] for node in graph.tensor_placeholders)
    input_strides = tuple(graph.strides[node] for node in graph.tensor_placeholders)
    compiled = _GenericLibrary(
        so_path=so_path,
        lib=lib,
        input_shapes=input_shapes,
        input_strides=input_strides,
        output_shape=graph.shapes[graph.output_value],
        dtype=graph.dtype,
    )
    _LIBRARY_CACHE[digest] = compiled
    return compiled


def _validate_runtime_inputs(
    inputs: Iterable[torch.Tensor],
    expected_dtypes: Sequence[torch.dtype],
    graph_dtype: _CodegenDType,
) -> None:
    for tensor, expected_dtype in zip(inputs, expected_dtypes):
        if expected_dtype is torch.bool:
            if tensor.dtype is not torch.bool:
                raise RefBackendError(
                    "codegen backend expects boolean condition tensors"
                )
        elif expected_dtype in _EMBEDDING_INDEX_DTYPES:
            if tensor.dtype is not expected_dtype:
                raise RefBackendError(
                    "codegen backend expects int32 or int64 index tensors"
                )
        elif tensor.dtype is not graph_dtype.torch_dtype:
            raise RefBackendError(
                f"codegen backend supports only {graph_dtype.torch_dtype} tensors"
            )
        if tensor.device.type != "cpu":
            raise RefBackendError("codegen backend supports only CPU tensors")


def _compile_graph(
    gm: torch.fx.GraphModule, example_inputs: List[object]
) -> Callable[..., torch.Tensor]:
    graph = _analyze_generic_graph(gm, example_inputs)
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
    conv_contiguous_indices = tuple(
        sorted(
            {
                graph.tensor_placeholders.index(input_node)
                for op_node in graph.op_nodes
                if op_node.spec.kind in {"conv1d", "conv2d"}
                for input_node in op_node.inputs
                if input_node in graph.tensor_placeholders
            }
        )
    )

    def _recompile(new_inputs: Sequence[object]) -> None:
        nonlocal graph, lib, output_inplace_input
        graph = _analyze_generic_graph(gm, new_inputs)
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
            raise RefBackendError(
                f"codegen backend expects {len(graph.placeholders)} inputs, got {len(normalized_args)}"
            )
        env: Dict[torch.fx.Node, object] = {}
        input_tensors = []
        for node, value in zip(graph.placeholders, normalized_args):
            env[node] = value
            if node in graph.tensor_placeholders:
                if not isinstance(value, torch.Tensor):
                    raise RefBackendError("codegen backend expects tensor inputs only")
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
            updated_graph = _analyze_generic_graph(gm, list(normalized_args))
            cached_lib = _compile_generic_library(updated_graph)
            library_cache[cache_key] = cached_lib
        lib = cached_lib
        if output_inplace_input is not None:
            original_input = env[output_inplace_input]
            if not isinstance(original_input, torch.Tensor):
                raise RefBackendError("codegen backend expects tensor inputs only")
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
            if graph.output_op.spec.kind == "empty_strided":
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
        return resolve_output(output_structure, env)

    return compiled


def get_generic_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> str:
    graph = _analyze_generic_graph(gm, example_inputs)
    return _write_generic_source(graph)



def codegen_generic_backend(
    gm: torch.fx.GraphModule, example_inputs: List[object]
) -> Callable[..., torch.Tensor]:
    return _compile_graph(gm, example_inputs)
