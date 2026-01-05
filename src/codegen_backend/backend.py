import hashlib
import math
import operator
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from importlib import resources
from jinja2 import Environment, FileSystemLoader
import torch.fx
from torch.fx.immutable_collections import immutable_list

from c_ref_backend.cffi_bindings import (
    RefBackendError,
    _normalize_conv2d_param as _normalize_conv2d_pair,
)
from codegen_backend.ops_registry import SUPPORTED_OPS
from codegen_backend.specs import _OpSpec

@dataclass(frozen=True)
class _CodegenDType:
    torch_dtype: torch.dtype
    c_type: str
    scalar_header: str
    scalar_prefix: str
    suffix: str


_CODEGEN_DTYPES = {
    torch.float32: _CodegenDType(
        torch_dtype=torch.float32,
        c_type="float",
        scalar_header="ops_scalar_f32.h",
        scalar_prefix="ref_scalar_f32_",
        suffix="f32",
    ),
    torch.int8: _CodegenDType(
        torch_dtype=torch.int8,
        c_type="int8_t",
        scalar_header="ops_scalar_i8.h",
        scalar_prefix="ref_scalar_i8_",
        suffix="i8",
    ),
    torch.int32: _CodegenDType(
        torch_dtype=torch.int32,
        c_type="int32_t",
        scalar_header="ops_scalar_i32.h",
        scalar_prefix="ref_scalar_i32_",
        suffix="i32",
    ),
    torch.bool: _CodegenDType(
        torch_dtype=torch.bool,
        c_type="bool",
        scalar_header="ops_scalar_bool.h",
        scalar_prefix="ref_scalar_bool_",
        suffix="bool",
    ),
}

_INTEGER_CODEGEN_DTYPES = {torch.int8, torch.int32}
_C_TYPE_BY_DTYPE = {
    torch.bool: "uint8_t",
    torch.int8: "int8_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.float32: "float",
}
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
_PARAMETRIC_UNARY_OPS = {"gelu", "elu", "leaky_relu", "softplus"}
_FLOAT_ONLY_UNARY_OPS = _PARAMETRIC_UNARY_OPS

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


@dataclass(frozen=True)
class _TargetInfo:
    op_spec: _OpSpec
    inplace_arg_index: int | None


def _build_target_registry() -> Dict[object, _TargetInfo]:
    registry: Dict[object, _TargetInfo] = {}
    for spec in SUPPORTED_OPS.values():
        for target in spec.supported_targets:
            inplace_arg_index = (
                spec.inplace_arg_index if target in spec.inplace_targets else None
            )
            registry[target] = _TargetInfo(
                op_spec=spec, inplace_arg_index=inplace_arg_index
            )
    return registry


TARGET_REGISTRY = _build_target_registry()


@dataclass
class _OpNode:
    node: torch.fx.Node
    spec: _OpSpec
    inputs: List[torch.fx.Node]
    output_shape: Tuple[int, ...] | List[int]
    inplace_input: int | None = None
    reduction_dims: Tuple[int, ...] | None = None
    keepdim: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def p(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass
class _GenericGraph:
    placeholders: List[torch.fx.Node]
    tensor_placeholders: List[torch.fx.Node]
    op_nodes: List[_OpNode]
    output_node: torch.fx.Node
    output_value: torch.fx.Node
    output_inplace_input: torch.fx.Node | None
    output_structure: object
    shapes: Dict[torch.fx.Node, Tuple[int, ...]]
    strides: Dict[torch.fx.Node, Tuple[int, ...]]
    dtypes: Dict[torch.fx.Node, torch.dtype]
    dtype: _CodegenDType


@dataclass
class _GenericLibrary:
    so_path: Path
    lib: object
    input_shapes: Tuple[Tuple[int, ...], ...]
    input_strides: Tuple[Tuple[int, ...], ...]
    output_shape: Tuple[int, ...]
    dtype: _CodegenDType

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = getattr(self.lib, f"ref_codegen_main_{self.dtype.suffix}")
        args = [tensor.data_ptr() for tensor in inputs]
        args.append(out.data_ptr())
        fn(*args)


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


def _normalize_pool2d_param(name: str, value: object) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return (value[0], value[1])
    raise RefBackendError(f"pool2d expects {name} to be an int or a pair of ints")


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


def _conv1d_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tuple[int, int, int]:
    batch, in_channels, in_l = input_shape
    out_channels, weight_in_channels, kernel_l = weight_shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "codegen conv1d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "codegen conv1d requires output channels to be divisible by groups"
        )
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
    raise RefBackendError(
        "codegen backend supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
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
    if _is_integer_dtype(dtype.torch_dtype):
        return str(int(value))
    if dtype.torch_dtype is torch.float32:
        return f"{float(value)}f"
    raise RefBackendError(
        "codegen addmm-like ops support only floating point tensors"
    )


def emit_signature(
    node_index: int,
    op_spec: _OpSpec,
    output_shape: Sequence[int],
    input_shapes: Sequence[Sequence[int]],
    input_dtypes: Sequence[torch.dtype],
    dtype: _CodegenDType,
) -> str:
    out_suffix = _format_array_suffix(output_shape)
    if op_spec.kind == "binary":
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
        cond_shape, a_shape, b_shape = input_shapes
        cond_suffix = _format_array_suffix(cond_shape)
        a_suffix = _format_array_suffix(a_shape)
        b_suffix = _format_array_suffix(b_shape)
        cond_c_type = _input_c_type(input_dtypes[0], dtype)
        a_c_type = _input_c_type(input_dtypes[1], dtype)
        b_c_type = _input_c_type(input_dtypes[2], dtype)
        return (
            f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
            f"const {cond_c_type} cond{cond_suffix}, "
            f"const {a_c_type} a{a_suffix}, "
            f"const {b_c_type} b{b_suffix}, "
            f"{dtype.c_type} out{out_suffix}) {{"
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


def _emit_parametric_unary(
    op_name: str,
    input_access: str,
    output_access: str,
    indent: str,
    dtype: _CodegenDType,
    params: Dict[str, object],
) -> List[str]:
    if dtype.torch_dtype is not torch.float32:
        raise RefBackendError(
            f"codegen {op_name} supports only torch.float32 tensors"
        )
    one = _format_scalar_literal(1.0, dtype)
    half = _format_scalar_literal(0.5, dtype)
    if op_name == "gelu":
        approximate = params.get("approximate", "none")
        if approximate == "tanh":
            sqrt_2_over_pi = _format_scalar_literal(0.7978845608028654, dtype)
            coeff = _format_scalar_literal(0.044715, dtype)
            return [
                f"{indent}{output_access} = {half} * {input_access} * "
                f"({one} + tanhf({sqrt_2_over_pi} * "
                f"({input_access} + {coeff} * {input_access} * {input_access} * {input_access})));"
            ]
        inv_sqrt2 = _format_scalar_literal(0.7071067811865475, dtype)
        return [
            f"{indent}{output_access} = {half} * {input_access} * "
            f"({one} + erff({input_access} * {inv_sqrt2}));"
        ]
    if op_name == "elu":
        alpha = _format_scalar_literal(params.get("alpha", 1.0), dtype)
        scale = _format_scalar_literal(params.get("scale", 1.0), dtype)
        input_scale = _format_scalar_literal(params.get("input_scale", 1.0), dtype)
        return [
            f"{indent}{output_access} = ({input_access} > 0.0f) ? "
            f"({scale} * {input_access}) : "
            f"({scale} * {alpha} * (expf({input_scale} * {input_access}) - {one}));"
        ]
    if op_name == "leaky_relu":
        negative_slope = _format_scalar_literal(
            params.get("negative_slope", 0.01), dtype
        )
        return [
            f"{indent}{output_access} = ({input_access} > 0.0f) ? "
            f"{input_access} : ({negative_slope} * {input_access});"
        ]
    if op_name == "softplus":
        beta = _format_scalar_literal(params.get("beta", 1.0), dtype)
        threshold = _format_scalar_literal(params.get("threshold", 20.0), dtype)
        return [
            f"{indent}{output_access} = ({beta} * {input_access} > {threshold}) ? "
            f"{input_access} : (log1pf(expf({beta} * {input_access})) / {beta});"
        ]
    raise RefBackendError(f"Unsupported parametric unary op: {op_name}")


def emit_body(
    op_node: _OpNode,
    output_access: str,
    input_shapes: Sequence[Sequence[int]],
    input_strides: Sequence[Sequence[int]],
    input_dtypes: Sequence[torch.dtype],
    output_shape: Sequence[int],
    indent: str,
    dtype: _CodegenDType,
) -> List[str]:
    op_spec = op_node.spec
    scalar_fn = f"{dtype.scalar_prefix}{op_spec.name}"
    if op_spec.kind == "binary":
        a_shape, b_shape = input_shapes
        a_strides, b_strides = input_strides
        a_index_expr = emit_input_access(
            "a",
            a_shape,
            a_strides,
            output_shape,
            broadcast_contiguous=True,
            c_type=_input_c_type(input_dtypes[0], dtype),
        )
        b_index_expr = emit_input_access(
            "b",
            b_shape,
            b_strides,
            output_shape,
            broadcast_contiguous=True,
            c_type=_input_c_type(input_dtypes[1], dtype),
        )
        return [
            f"{indent}{output_access} = {scalar_fn}({a_index_expr}, {b_index_expr});"
        ]
    if op_spec.kind == "where":
        cond_shape, a_shape, b_shape = input_shapes
        cond_strides, a_strides, b_strides = input_strides
        cond_index_expr = emit_input_access(
            "cond",
            cond_shape,
            cond_strides,
            output_shape,
            broadcast_contiguous=True,
            c_type=_input_c_type(input_dtypes[0], dtype),
        )
        a_index_expr = emit_input_access(
            "a",
            a_shape,
            a_strides,
            output_shape,
            broadcast_contiguous=True,
            c_type=_input_c_type(input_dtypes[1], dtype),
        )
        b_index_expr = emit_input_access(
            "b",
            b_shape,
            b_strides,
            output_shape,
            broadcast_contiguous=True,
            c_type=_input_c_type(input_dtypes[2], dtype),
        )
        return [
            f"{indent}{output_access} = ({cond_index_expr} != 0) ? {a_index_expr} : {b_index_expr};"
        ]
    a_shape = input_shapes[0]
    a_strides = input_strides[0]
    input_access = emit_input_access(
        "a",
        a_shape,
        a_strides,
        output_shape,
        broadcast_contiguous=False,
        c_type=_input_c_type(input_dtypes[0], dtype),
    )
    if op_spec.name in _PARAMETRIC_UNARY_OPS:
        return _emit_parametric_unary(
            op_spec.name,
            input_access,
            output_access,
            indent,
            dtype,
            op_node.params,
        )
    return [f"{indent}{output_access} = {scalar_fn}({input_access});"]


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
    lines = [
        emit_signature(
            node_index,
            op_node.spec,
            output_shape,
            input_shapes,
            input_dtypes,
            dtype,
        )
    ]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    lines.extend(
        emit_body(
            op_node,
            output_access,
            input_shapes,
            input_strides,
            input_dtypes,
            output_shape,
            indent,
            dtype,
        )
    )
    lines.extend(emit_footer(output_shape, indent))
    return lines


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
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    mat1_is_contiguous = _is_contiguous(mat1_shape, mat1_strides)
    mat2_is_contiguous = _is_contiguous(mat2_shape, mat2_strides)
    output_is_contiguous = _is_contiguous(input_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    m, k = mat1_shape
    _, n = mat2_shape
    input_suffix = _format_array_suffix(input_shape)
    mat1_suffix = _format_array_suffix(mat1_shape)
    mat2_suffix = _format_array_suffix(mat2_shape)
    out_suffix = _format_array_suffix(input_shape)
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
            ("i", "j"),
            input_strides,
            input_is_contiguous,
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
            sizes=input_shape,
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
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    batch1_is_contiguous = _is_contiguous(batch1_shape, batch1_strides)
    batch2_is_contiguous = _is_contiguous(batch2_shape, batch2_strides)
    output_is_contiguous = _is_contiguous(input_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    batch, m, k = batch1_shape
    _, _, n = batch2_shape
    input_suffix = _format_array_suffix(input_shape)
    batch1_suffix = _format_array_suffix(batch1_shape)
    batch2_suffix = _format_array_suffix(batch2_shape)
    out_suffix = _format_array_suffix(input_shape)
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
            ("i", "j"),
            input_strides,
            input_is_contiguous,
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
            sizes=input_shape,
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
    output_is_contiguous = _is_contiguous(input_shape, output_strides)
    acc_type = dtype.c_type
    acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"
    m, n = mat_shape
    input_suffix = _format_array_suffix(input_shape)
    mat_suffix = _format_array_suffix(mat_shape)
    vec_suffix = _format_array_suffix(vec_shape)
    out_suffix = _format_array_suffix(input_shape)
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
        input_access=_emit_strided_access(
            "input",
            ("i",),
            input_strides,
            input_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
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
            sizes=input_shape,
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
    output_is_contiguous = _is_contiguous(input_shape, output_strides)
    m, n = input_shape
    input_suffix = _format_array_suffix(input_shape)
    vec1_suffix = _format_array_suffix(vec1_shape)
    vec2_suffix = _format_array_suffix(vec2_shape)
    out_suffix = _format_array_suffix(input_shape)
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
        input_access=_emit_strided_access(
            "input",
            ("i", "j"),
            input_strides,
            input_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
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
            sizes=input_shape,
            c_type=dtype.c_type,
        ),
        alpha=_format_scalar_literal(alpha, dtype),
        beta=_format_scalar_literal(beta, dtype),
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
    reduce_expr = None
    if op_spec.name in {"amax", "amin"} and dtype.torch_dtype is not torch.bool:
        compare_op = ">" if op_spec.name == "amax" else "<"
        init_value_config = _MINMAX_INIT_VALUES[dtype.torch_dtype][op_spec.name]
        config = {
            "init_value": init_value_config,
            "post_op": None,
        }
        if dtype.torch_dtype is torch.float32:
            isnan_fn = f"{dtype.scalar_prefix}isnan"
            reduce_expr = (
                f"acc = {isnan_fn}({input_access}) ? {input_access} : "
                f"({isnan_fn}(acc) ? acc : ({input_access} {compare_op} acc ? {input_access} : acc))"
            )
        else:
            reduce_expr = (
                f"acc = ({input_access} {compare_op} acc ? {input_access} : acc)"
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
    if reduce_expr is None:
        if bool_reduction:
            reduce_expr = f"acc {config['reduce_op']} ({input_access} != 0)"
        else:
            reduce_expr = f"acc {config['reduce_op']} {input_access}"
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
        reduce_expr=reduce_expr,
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
        if op_node.spec.kind in {"binary", "unary", "where"}:
            input_shapes = [graph.shapes[arg] for arg in op_node.inputs]
            input_strides = [graph.strides[arg] for arg in op_node.inputs]
            input_dtypes = [graph.dtypes[arg] for arg in op_node.inputs]
            output_strides = graph.strides[op_node.node]
            kernel_lines = _write_elementwise_kernel(
                index,
                op_node,
                op_node.output_shape,
                input_shapes,
                input_strides,
                input_dtypes,
                output_strides,
                graph.dtype,
            )
        elif op_node.spec.kind == "reduction":
            input_node = op_node.inputs[0]
            if op_node.spec.name == "std":
                kernel_lines = _write_std_kernel(
                    index,
                    op_node.spec,
                    graph.shapes[input_node],
                    graph.strides[input_node],
                    op_node.output_shape,
                    graph.strides[op_node.node],
                    op_node.reduction_dims or (),
                    op_node.keepdim,
                    graph.dtype,
                    unbiased=bool(op_node.p("unbiased", True)),
                )
            elif op_node.spec.name == "var":
                kernel_lines = _write_var_kernel(
                    index,
                    op_node.spec,
                    graph.shapes[input_node],
                    graph.strides[input_node],
                    op_node.output_shape,
                    graph.strides[op_node.node],
                    op_node.reduction_dims or (),
                    op_node.keepdim,
                    graph.dtype,
                    unbiased=bool(op_node.p("unbiased", True)),
                )
            elif op_node.spec.name == "norm":
                kernel_lines = _write_norm_kernel(
                    index,
                    op_node.spec,
                    graph.shapes[input_node],
                    graph.strides[input_node],
                    op_node.output_shape,
                    graph.strides[op_node.node],
                    op_node.reduction_dims or (),
                    op_node.keepdim,
                    graph.dtype,
                    p_value=float(op_node.p("norm_p", 2.0)),
                )
            else:
                kernel_lines = _write_reduction_kernel(
                    index,
                    op_node.spec,
                    graph.shapes[input_node],
                    graph.strides[input_node],
                    op_node.output_shape,
                    graph.strides[op_node.node],
                    op_node.reduction_dims or (),
                    op_node.keepdim,
                    graph.dtype,
                )
        elif op_node.spec.kind == "arg_reduction":
            input_node = op_node.inputs[0]
            kernel_lines = _write_argminmax_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.strides[input_node],
                op_node.output_shape,
                graph.strides[op_node.node],
                op_node.reduction_dims or (),
                op_node.keepdim,
                bool(op_node.p("reduce_all", False)),
                graph.dtype,
            )
        elif op_node.spec.kind == "softmax":
            input_node = op_node.inputs[0]
            kernel_lines = _write_softmax_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.strides[input_node],
                graph.strides[op_node.node],
                op_node.p("dim"),
                graph.dtype,
            )
        elif op_node.spec.kind == "concat":
            input_shapes = [graph.shapes[arg] for arg in op_node.inputs]
            input_strides = [graph.strides[arg] for arg in op_node.inputs]
            kernel_lines = _write_concat_kernel(
                index,
                op_node.spec,
                input_shapes,
                input_strides,
                op_node.output_shape,
                graph.strides[op_node.node],
                op_node.p("dim", 0),
                graph.dtype,
            )
        elif op_node.spec.kind == "pool2d":
            input_node = op_node.inputs[0]
            kernel_lines = _write_pool2d_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                op_node.output_shape,
                op_node.p("kernel_size", (1, 1)),
                op_node.p("stride", (1, 1)),
                op_node.p("padding", (0, 0)),
                op_node.p("dilation", (1, 1)),
                graph.dtype,
                bool(op_node.p("ceil_mode", False)),
                bool(op_node.p("count_include_pad", False)),
                op_node.p("divisor_override"),
            )
        elif op_node.spec.kind == "conv1d":
            input_node, weight_node, *bias_nodes = op_node.inputs
            kernel_lines = _write_conv1d_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.shapes[weight_node],
                op_node.output_shape,
                op_node.p("stride", 1),
                op_node.p("padding", 0),
                op_node.p("dilation", 1),
                op_node.p("groups", 1),
                graph.dtype,
                bool(op_node.p("has_bias", False)),
            )
        elif op_node.spec.kind == "conv2d":
            input_node, weight_node, *bias_nodes = op_node.inputs
            kernel_lines = _write_conv2d_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.shapes[weight_node],
                op_node.output_shape,
                op_node.p("stride", (1, 1)),
                op_node.p("padding", (0, 0)),
                op_node.p("dilation", (1, 1)),
                op_node.p("groups", 1),
                graph.dtype,
                bool(op_node.p("has_bias", False)),
            )
        elif op_node.spec.kind == "addmm":
            input_node, mat1_node, mat2_node = op_node.inputs
            kernel_lines = _write_addmm_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.shapes[mat1_node],
                graph.shapes[mat2_node],
                graph.strides[input_node],
                graph.strides[mat1_node],
                graph.strides[mat2_node],
                graph.strides[op_node.node],
                graph.dtype,
                alpha=float(op_node.p("alpha", 1.0)),
                beta=float(op_node.p("beta", 1.0)),
            )
        elif op_node.spec.kind == "addbmm":
            input_node, batch1_node, batch2_node = op_node.inputs
            kernel_lines = _write_addbmm_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.shapes[batch1_node],
                graph.shapes[batch2_node],
                graph.strides[input_node],
                graph.strides[batch1_node],
                graph.strides[batch2_node],
                graph.strides[op_node.node],
                graph.dtype,
                alpha=float(op_node.p("alpha", 1.0)),
                beta=float(op_node.p("beta", 1.0)),
            )
        elif op_node.spec.kind == "addmv":
            input_node, mat_node, vec_node = op_node.inputs
            kernel_lines = _write_addmv_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.shapes[mat_node],
                graph.shapes[vec_node],
                graph.strides[input_node],
                graph.strides[mat_node],
                graph.strides[vec_node],
                graph.strides[op_node.node],
                graph.dtype,
                alpha=float(op_node.p("alpha", 1.0)),
                beta=float(op_node.p("beta", 1.0)),
            )
        elif op_node.spec.kind == "addr":
            input_node, vec1_node, vec2_node = op_node.inputs
            kernel_lines = _write_addr_kernel(
                index,
                op_node.spec,
                graph.shapes[input_node],
                graph.shapes[vec1_node],
                graph.shapes[vec2_node],
                graph.strides[input_node],
                graph.strides[vec1_node],
                graph.strides[vec2_node],
                graph.strides[op_node.node],
                graph.dtype,
                alpha=float(op_node.p("alpha", 1.0)),
                beta=float(op_node.p("beta", 1.0)),
            )
        else:
            lhs, rhs = op_node.inputs
            lhs_shape = graph.shapes[lhs]
            rhs_shape = graph.shapes[rhs]
            lhs_strides = graph.strides[lhs]
            rhs_strides = graph.strides[rhs]
            kernel_lines = _write_matmul_kernel(
                index,
                op_node.spec,
                lhs_shape,
                rhs_shape,
                lhs_strides,
                rhs_strides,
                graph.dtype,
            )
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
        input_names = [name_map[arg] for arg in op_node.inputs]
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
) -> _CodegenDType:
    tensor_examples = [
        example
        for example in _iter_example_tensors(example_inputs)
        if example.dtype in _CODEGEN_DTYPES
    ]
    if not tensor_examples:
        raise RefBackendError("codegen backend requires at least one example tensor input")
    non_bool_examples = [
        example for example in tensor_examples if example.dtype is not torch.bool
    ]
    if non_bool_examples:
        first_dtype = non_bool_examples[0].dtype
    else:
        first_dtype = torch.bool
    dtype_info = _CODEGEN_DTYPES.get(first_dtype)
    if dtype_info is None:
        raise RefBackendError(
            "codegen backend supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
        )
    for example in tensor_examples:
        if example.dtype is not first_dtype and example.dtype is not torch.bool:
            raise RefBackendError("codegen backend expects all tensors to share a dtype")
        if example.device.type != "cpu":
            raise RefBackendError("codegen backend supports only CPU tensors")
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
    if op_spec.kind == "binary":
        return _broadcast_output_shape(op_spec, *input_shapes)
    if op_spec.kind == "where":
        return _broadcast_output_shape(op_spec, *input_shapes)
    if op_spec.kind == "unary":
        return input_shapes[0]
    if op_spec.kind == "softmax":
        return input_shapes[0]
    if op_spec.kind == "reduction":
        return ()
    if op_spec.kind == "concat":
        raise RefBackendError("codegen cat expects a tensor list input")
    if op_spec.kind == "conv1d":
        raise RefBackendError("codegen conv1d expects convolution arguments")
    if op_spec.kind == "conv2d":
        raise RefBackendError("codegen conv2d expects convolution arguments")
    if op_spec.kind == "addmm":
        input_shape, mat1_shape, mat2_shape = input_shapes
        if len(input_shape) != 2 or len(mat1_shape) != 2 or len(mat2_shape) != 2:
            raise RefBackendError("codegen addmm expects 2D inputs")
        if mat1_shape[1] != mat2_shape[0]:
            raise RefBackendError("codegen addmm requires inner dimensions to match")
        expected_shape = (mat1_shape[0], mat2_shape[1])
        if input_shape != expected_shape:
            raise RefBackendError(
                "codegen addmm expects input shape to match matmul output"
            )
        return expected_shape
    if op_spec.kind == "addbmm":
        input_shape, batch1_shape, batch2_shape = input_shapes
        if (
            len(input_shape) != 2
            or len(batch1_shape) != 3
            or len(batch2_shape) != 3
        ):
            raise RefBackendError("codegen addbmm expects 2D input and 3D batches")
        if batch1_shape[0] != batch2_shape[0]:
            raise RefBackendError("codegen addbmm requires batch dimensions to match")
        if batch1_shape[2] != batch2_shape[1]:
            raise RefBackendError("codegen addbmm requires inner dimensions to match")
        expected_shape = (batch1_shape[1], batch2_shape[2])
        if input_shape != expected_shape:
            raise RefBackendError(
                "codegen addbmm expects input shape to match bmm output"
            )
        return expected_shape
    if op_spec.kind == "addmv":
        input_shape, mat_shape, vec_shape = input_shapes
        if len(input_shape) != 1 or len(mat_shape) != 2 or len(vec_shape) != 1:
            raise RefBackendError(
                "codegen addmv expects 1D input and 2D matrix/1D vector"
            )
        if mat_shape[1] != vec_shape[0]:
            raise RefBackendError("codegen addmv requires inner dimensions to match")
        expected_shape = (mat_shape[0],)
        if input_shape != expected_shape:
            raise RefBackendError(
                "codegen addmv expects input shape to match mat-vec output"
            )
        return expected_shape
    if op_spec.kind == "addr":
        input_shape, vec1_shape, vec2_shape = input_shapes
        if len(input_shape) != 2 or len(vec1_shape) != 1 or len(vec2_shape) != 1:
            raise RefBackendError("codegen addr expects 2D input and 1D vectors")
        expected_shape = (vec1_shape[0], vec2_shape[0])
        if input_shape != expected_shape:
            raise RefBackendError(
                "codegen addr expects input shape to match outer product output"
            )
        return expected_shape
    a_shape, b_shape = input_shapes
    if op_spec.name == "matmul":
        if len(a_shape) == 1 and len(b_shape) == 1:
            if a_shape[0] != b_shape[0]:
                raise RefBackendError(
                    "codegen matmul requires inner dimensions to match"
                )
            return ()
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise RefBackendError("codegen matmul requires 1D or 2D inputs")
        if a_shape[1] != b_shape[0]:
            raise RefBackendError("codegen matmul requires inner dimensions to match")
        return (a_shape[0], b_shape[1])
    if len(a_shape) != 3 or len(b_shape) != 3:
        raise RefBackendError("codegen bmm requires 3D inputs")
    if a_shape[0] != b_shape[0]:
        raise RefBackendError("codegen bmm requires batch dimensions to match")
    if a_shape[2] != b_shape[1]:
        raise RefBackendError("codegen bmm requires inner dimensions to match")
    return (a_shape[0], a_shape[1], b_shape[2])


def _normalize_reduction_dims(
    op_name: str, dim: object | None, rank: int
) -> Tuple[int, ...]:
    if dim is None:
        return tuple(range(rank))
    if isinstance(dim, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_name} expects dim to be an int or tuple of ints"
        )
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


def _normalize_conv1d_param(name: str, value: object) -> int:
    if isinstance(value, int):
        return value
    if (
        isinstance(value, (tuple, list))
        and len(value) == 1
        and all(isinstance(item, int) for item in value)
    ):
        return value[0]
    raise RefBackendError(f"codegen conv1d expects {name} to be an int or 1-item tuple")


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
            extra = set(node.kwargs) - {"dim", "unbiased", "keepdim"}
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
    if isinstance(dim, torch.fx.Node):
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, (tuple, list)):
        raise RefBackendError(f"codegen {op_name} expects dim to be an int")
    try:
        dim_value = operator.index(dim)
    except TypeError as exc:
        raise RefBackendError(f"codegen {op_name} expects dim to be an int") from exc
    if dim_value < 0:
        dim_value += len(input_shape)
    if dim_value < 0 or dim_value >= len(input_shape):
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
    raise RefBackendError(f"Unsupported parametric op: {op_name}")


def _parse_softmax_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, object | None]:
    if not node.args:
        raise RefBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 3:
        raise RefBackendError(f"codegen {op_name} expects at most three inputs")
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
        is_log=op_spec.name == "log_softmax",
    )
    return rendered.strip().splitlines()
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
        if not _is_contiguous(bias_shape, strides[bias_node]):
            raise RefBackendError("codegen conv1d requires contiguous bias")
    if len(input_shape) != 3 or len(weight_shape) != 3:
        raise RefBackendError("codegen conv1d requires 3D input and weight tensors")
    if not _is_contiguous(input_shape, strides[input_arg]) or not _is_contiguous(
        weight_shape, strides[weight_arg]
    ):
        raise RefBackendError("codegen conv1d requires contiguous tensors")
    stride_value = _normalize_conv1d_param("stride", stride)
    padding_value = _normalize_conv1d_param("padding", padding)
    dilation_value = _normalize_conv1d_param("dilation", dilation)
    if stride_value <= 0 or dilation_value <= 0 or padding_value < 0:
        raise RefBackendError(
            "codegen conv1d expects stride and dilation to be positive and padding to be non-negative"
        )
    if not isinstance(groups, int) or groups <= 0:
        raise RefBackendError("codegen conv1d requires positive groups")
    output_shape = _conv1d_output_shape_from_shapes(
        input_shape,
        weight_shape,
        stride_value,
        padding_value,
        dilation_value,
        groups,
    )
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
    stride_pair = _normalize_conv2d_pair("stride", stride)
    dilation_pair = _normalize_conv2d_pair("dilation", dilation)
    if isinstance(padding, str):
        padding_mode = padding.lower()
        if padding_mode not in ("same", "valid"):
            raise RefBackendError(
                "codegen conv2d expects padding to be an int, a pair of ints, or 'same'/'valid'"
            )
        if padding_mode == "valid":
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
        padding_pair = _normalize_conv2d_pair("padding", padding)
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


def _handle_pool2d_node(
    node: torch.fx.Node,
    op_spec: _OpSpec,
    dtype_info: _CodegenDType,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    strides: Dict[torch.fx.Node, Tuple[int, ...]],
    dtypes: Dict[torch.fx.Node, torch.dtype],
) -> _OpNode:
    if op_spec.name == "max_pool2d":
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
    kernel_pair = _normalize_pool2d_param("kernel_size", kernel_size)
    if stride is None:
        stride_pair = kernel_pair
    else:
        stride_pair = _normalize_pool2d_param("stride", stride)
    padding_pair = _normalize_pool2d_param("padding", padding)
    dilation_pair = _normalize_pool2d_param("dilation", dilation)
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
    if dtype_info.torch_dtype is not torch.float32:
        raise RefBackendError(
            f"codegen {op_name} supports only torch.float32 tensors"
        )
    if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
        raise RefBackendError(
            f"codegen {op_name} expects inputs to share the graph dtype"
        )
    output_shape = _infer_output_shape(op_spec, input_shapes)
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


def _analyze_generic_graph(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> _GenericGraph:
    dtype_info = _validate_example_inputs(example_inputs)
    output_node = None
    placeholders: List[torch.fx.Node] = []
    tensor_placeholders: List[torch.fx.Node] = []
    op_nodes: List[_OpNode] = []
    shapes: Dict[torch.fx.Node, Tuple[int, ...]] = {}
    strides: Dict[torch.fx.Node, Tuple[int, ...]] = {}
    dtypes: Dict[torch.fx.Node, torch.dtype] = {}
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
                if example.dtype not in _CODEGEN_DTYPES and example.numel() == 1:
                    continue
                shapes[node] = tuple(example.shape)
                strides[node] = tuple(example.stride())
                dtypes[node] = example.dtype
                tensor_placeholders.append(node)
            continue
        if node.op in {"call_function", "call_method"}:
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
            if op_spec.kind == "concat":
                op_nodes.append(
                    _handle_concat_node(
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
            reduction_dims: Tuple[int, ...] | None = None
            keepdim = False
            reduce_all = False
            param_values: Dict[str, object] = {}
            if op_spec.kind in {"reduction", "arg_reduction"}:
                if len(node.args) < 1:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects one input"
                    )
                args_to_check = node.args[:1]
            elif op_spec.kind == "unary" and op_spec.name in _PARAMETRIC_UNARY_OPS:
                input_node, param_values = _parse_parametric_unary_args(
                    op_spec.name, node
                )
                args_to_check = (input_node,)
            else:
                if node.kwargs:
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
                if len(node.args) != expected_arity:
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
                args_to_check = node.args
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
                    ) = _parse_norm_args(op_spec.name, node, input_shapes[0])
                    param_values["norm_p"] = norm_p
                else:
                    (
                        reduction_dims,
                        keepdim,
                        reduce_all,
                        unbiased,
                    ) = _parse_reduction_args(op_spec.name, node, input_shapes[0])
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
                    input_shapes[0],
                    reduction_dims,
                    keepdim,
                    reduce_all=reduce_all,
                )
            elif op_spec.kind == "arg_reduction":
                (
                    reduction_dims,
                    keepdim,
                    reduce_all,
                ) = _parse_argminmax_args(op_spec.name, node, input_shapes[0])
                reduction_count = 1
                if reduce_all:
                    for size in input_shapes[0]:
                        reduction_count *= size
                else:
                    for dim in reduction_dims:
                        reduction_count *= input_shapes[0][dim]
                if reduction_count == 0:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects a non-empty reduction dimension"
                    )
                output_shape = _infer_reduction_output_shape(
                    input_shapes[0],
                    reduction_dims,
                    keepdim,
                    reduce_all=reduce_all,
                )
            else:
                output_shape = _infer_output_shape(op_spec, input_shapes)
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
    if not tensor_placeholders:
        raise RefBackendError("codegen backend requires at least one tensor input")
    output_value, output_structure = _unwrap_output_node(output_node)
    if output_value not in shapes:
        raise RefBackendError("codegen backend expects a single output node")
    if output_value not in {op.node for op in op_nodes}:
        raise RefBackendError("codegen backend output must be an operator result")

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
        output_inplace_input=output_inplace_input,
        output_structure=output_structure,
        shapes=shapes,
        strides=strides,
        dtypes=dtypes,
        dtype=dtype_info,
    )


def _compile_generic_library(graph: _GenericGraph) -> _GenericLibrary:
    source = _write_generic_source(graph)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _LIBRARY_CACHE.get(digest)
    if cached is not None:
        return cached

    build_dir = Path(tempfile.mkdtemp(prefix="codegen_generic_"))
    c_path = build_dir / "ref_codegen_generic.c"
    so_path = build_dir / "ref_codegen_generic.so"
    c_path.write_text(source, encoding="utf-8")

    cmd = [
        "cc",
        "-shared",
        "-O3",
        "-fPIC",
        "-I",
        str(_C_SRC_DIR),
        str(c_path),
        "-o",
        str(so_path),
    ]
    subprocess.check_call(cmd)

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
    conv2d_contiguous_indices = tuple(
        sorted(
            {
                graph.tensor_placeholders.index(input_node)
                for op_node in graph.op_nodes
                if op_node.spec.kind == "conv2d"
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
        if conv2d_contiguous_indices:
            for index in conv2d_contiguous_indices:
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
            out = torch.empty(
                lib.output_shape,
                dtype=output_dtype,
                device=contiguous_inputs[0].device,
            )
            lib.run(contiguous_inputs, out)
            env[output_value] = out
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
