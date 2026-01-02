import hashlib
import operator
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from c_ref_backend.cffi_bindings import RefBackendError


@dataclass(frozen=True)
class _OpSpec:
    name: str
    kind: str
    symbol: str | None
    supported_targets: set


def _binary_spec(name: str, targets: Iterable[object], symbol: str | None) -> _OpSpec:
    return _OpSpec(
        name=name,
        kind="binary",
        symbol=symbol,
        supported_targets=set(targets),
    )


def _unary_spec(name: str, targets: Iterable[object]) -> _OpSpec:
    return _OpSpec(
        name=name,
        kind="unary",
        symbol=None,
        supported_targets=set(targets),
    )


SUPPORTED_OPS = {
    "add": _binary_spec(
        "add",
        (
            operator.add,
            torch.add,
            torch.ops.prims.add,
            torch.ops.prims.add.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        ),
        "+",
    ),
    "sub": _binary_spec(
        "sub",
        (
            operator.sub,
            torch.sub,
            torch.ops.prims.sub,
            torch.ops.prims.sub.default,
            torch.ops.aten.sub.Tensor,
        ),
        "-",
    ),
    "mul": _binary_spec(
        "mul",
        (
            operator.mul,
            torch.mul,
            torch.ops.prims.mul,
            torch.ops.prims.mul.default,
            torch.ops.aten.mul.Tensor,
        ),
        "*",
    ),
    "div": _binary_spec(
        "div",
        (
            operator.truediv,
            torch.div,
            torch.true_divide,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.div,
        ),
        "/",
    ),
    "maximum": _binary_spec(
        "maximum",
        (
            torch.maximum,
            torch.ops.aten.maximum.default,
            torch.ops.aten.maximum,
        ),
        None,
    ),
    "minimum": _binary_spec(
        "minimum",
        (
            torch.minimum,
            torch.ops.aten.minimum.default,
            torch.ops.aten.minimum,
        ),
        None,
    ),
    "atan2": _binary_spec(
        "atan2",
        (
            torch.atan2,
            torch.ops.aten.atan2.default,
            torch.ops.aten.atan2,
        ),
        None,
    ),
    "pow": _binary_spec(
        "pow",
        (
            operator.pow,
            torch.pow,
            torch.ops.aten.pow.Tensor_Tensor,
        ),
        None,
    ),
    "remainder": _binary_spec(
        "remainder",
        (
            torch.remainder,
            torch.ops.aten.remainder.Tensor,
            torch.ops.aten.remainder,
        ),
        None,
    ),
    "fmod": _binary_spec(
        "fmod",
        (
            torch.fmod,
            torch.ops.aten.fmod.Tensor,
            torch.ops.aten.fmod,
        ),
        None,
    ),
    "floor_divide": _binary_spec(
        "floor_divide",
        (
            torch.floor_divide,
            torch.ops.aten.floor_divide.default,
            torch.ops.aten.floor_divide,
        ),
        None,
    ),
    "fmax": _binary_spec(
        "fmax",
        (
            torch.fmax,
            torch.ops.aten.fmax.default,
            torch.ops.aten.fmax,
        ),
        None,
    ),
    "fmin": _binary_spec(
        "fmin",
        (
            torch.fmin,
            torch.ops.aten.fmin.default,
            torch.ops.aten.fmin,
        ),
        None,
    ),
    "copysign": _binary_spec(
        "copysign",
        (
            torch.copysign,
            torch.ops.aten.copysign.default,
            torch.ops.aten.copysign,
        ),
        None,
    ),
    "hypot": _binary_spec(
        "hypot",
        (
            torch.hypot,
            torch.ops.aten.hypot.default,
            torch.ops.aten.hypot,
        ),
        None,
    ),
    "logaddexp": _binary_spec(
        "logaddexp",
        (
            torch.logaddexp,
            torch.ops.aten.logaddexp.default,
            torch.ops.aten.logaddexp,
        ),
        None,
    ),
    "nextafter": _binary_spec(
        "nextafter",
        (
            torch.nextafter,
            torch.ops.aten.nextafter.default,
            torch.ops.aten.nextafter,
        ),
        None,
    ),
    "xlogy": _binary_spec(
        "xlogy",
        (
            torch.xlogy,
            torch.ops.aten.xlogy.Tensor,
            torch.ops.aten.xlogy,
        ),
        None,
    ),
    "heaviside": _binary_spec(
        "heaviside",
        (
            torch.heaviside,
            torch.ops.aten.heaviside.default,
            torch.ops.aten.heaviside,
        ),
        None,
    ),
    "ldexp": _binary_spec(
        "ldexp",
        (
            torch.ldexp,
            torch.ops.aten.ldexp.default,
            torch.ops.aten.ldexp,
        ),
        None,
    ),
    "clamp_min": _binary_spec(
        "clamp_min",
        (
            torch.clamp_min,
            torch.ops.aten.clamp_min.default,
            torch.ops.aten.clamp_min,
        ),
        None,
    ),
    "clamp_max": _binary_spec(
        "clamp_max",
        (
            torch.clamp_max,
            torch.ops.aten.clamp_max.default,
            torch.ops.aten.clamp_max,
        ),
        None,
    ),
    "neg": _unary_spec(
        "neg",
        (
            operator.neg,
            torch.neg,
            torch.ops.aten.neg.default,
            torch.ops.aten.neg,
        ),
    ),
    "exp": _unary_spec(
        "exp",
        (
            torch.exp,
            torch.ops.aten.exp.default,
            torch.ops.aten.exp,
        ),
    ),
    "abs": _unary_spec(
        "abs",
        (
            torch.abs,
            torch.ops.aten.abs.default,
            torch.ops.aten.abs,
        ),
    ),
    "sqrt": _unary_spec(
        "sqrt",
        (
            torch.sqrt,
            torch.ops.aten.sqrt.default,
            torch.ops.aten.sqrt,
        ),
    ),
    "log": _unary_spec(
        "log",
        (
            torch.log,
            torch.ops.aten.log.default,
            torch.ops.aten.log,
        ),
    ),
    "sin": _unary_spec(
        "sin",
        (
            torch.sin,
            torch.ops.aten.sin.default,
            torch.ops.aten.sin,
        ),
    ),
    "cos": _unary_spec(
        "cos",
        (
            torch.cos,
            torch.ops.aten.cos.default,
            torch.ops.aten.cos,
        ),
    ),
    "acos": _unary_spec(
        "acos",
        (
            torch.acos,
            torch.ops.aten.acos.default,
            torch.ops.aten.acos,
        ),
    ),
    "acosh": _unary_spec(
        "acosh",
        (
            torch.acosh,
            torch.ops.aten.acosh.default,
            torch.ops.aten.acosh,
        ),
    ),
    "asin": _unary_spec(
        "asin",
        (
            torch.asin,
            torch.ops.aten.asin.default,
            torch.ops.aten.asin,
        ),
    ),
    "asinh": _unary_spec(
        "asinh",
        (
            torch.asinh,
            torch.ops.aten.asinh.default,
            torch.ops.aten.asinh,
        ),
    ),
    "atan": _unary_spec(
        "atan",
        (
            torch.atan,
            torch.ops.aten.atan.default,
            torch.ops.aten.atan,
        ),
    ),
    "atanh": _unary_spec(
        "atanh",
        (
            torch.atanh,
            torch.ops.aten.atanh.default,
            torch.ops.aten.atanh,
            torch.ops.aten.atanh_.default,
            torch.ops.aten.atanh_,
        ),
    ),
    "cosh": _unary_spec(
        "cosh",
        (
            torch.cosh,
            torch.ops.aten.cosh.default,
            torch.ops.aten.cosh,
        ),
    ),
    "sinh": _unary_spec(
        "sinh",
        (
            torch.sinh,
            torch.ops.aten.sinh.default,
            torch.ops.aten.sinh,
        ),
    ),
    "tan": _unary_spec(
        "tan",
        (
            torch.tan,
            torch.ops.aten.tan.default,
            torch.ops.aten.tan,
        ),
    ),
    "erf": _unary_spec(
        "erf",
        (
            torch.erf,
            torch.ops.aten.erf.default,
            torch.ops.aten.erf,
        ),
    ),
    "erfc": _unary_spec(
        "erfc",
        (
            torch.erfc,
            torch.ops.aten.erfc.default,
            torch.ops.aten.erfc,
        ),
    ),
    "expm1": _unary_spec(
        "expm1",
        (
            torch.expm1,
            torch.ops.aten.expm1.default,
            torch.ops.aten.expm1,
        ),
    ),
    "log1p": _unary_spec(
        "log1p",
        (
            torch.log1p,
            torch.ops.aten.log1p.default,
            torch.ops.aten.log1p,
        ),
    ),
    "log2": _unary_spec(
        "log2",
        (
            torch.log2,
            torch.ops.aten.log2.default,
            torch.ops.aten.log2,
        ),
    ),
    "log10": _unary_spec(
        "log10",
        (
            torch.log10,
            torch.ops.aten.log10.default,
            torch.ops.aten.log10,
        ),
    ),
    "rsqrt": _unary_spec(
        "rsqrt",
        (
            torch.rsqrt,
            torch.ops.aten.rsqrt.default,
            torch.ops.aten.rsqrt,
        ),
    ),
    "sigmoid": _unary_spec(
        "sigmoid",
        (
            torch.sigmoid,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.sigmoid,
        ),
    ),
    "sign": _unary_spec(
        "sign",
        (
            torch.sign,
            torch.ops.aten.sign.default,
            torch.ops.aten.sign,
        ),
    ),
    "round": _unary_spec(
        "round",
        (
            torch.round,
            torch.ops.aten.round.default,
            torch.ops.aten.round,
        ),
    ),
    "trunc": _unary_spec(
        "trunc",
        (
            torch.trunc,
            torch.ops.aten.trunc.default,
            torch.ops.aten.trunc,
        ),
    ),
    "tanh": _unary_spec(
        "tanh",
        (
            torch.tanh,
            torch.ops.aten.tanh.default,
            torch.ops.aten.tanh,
        ),
    ),
    "floor": _unary_spec(
        "floor",
        (
            torch.floor,
            torch.ops.aten.floor.default,
            torch.ops.aten.floor,
        ),
    ),
    "ceil": _unary_spec(
        "ceil",
        (
            torch.ceil,
            torch.ops.aten.ceil.default,
            torch.ops.aten.ceil,
        ),
    ),
    "reciprocal": _unary_spec(
        "reciprocal",
        (
            torch.reciprocal,
            torch.ops.aten.reciprocal.default,
            torch.ops.aten.reciprocal,
        ),
    ),
    "relu": _unary_spec(
        "relu",
        (
            torch.relu,
            torch.ops.aten.relu.default,
            torch.ops.aten.relu,
        ),
    ),
    "angle": _unary_spec(
        "angle",
        (
            torch.angle,
            torch.ops.aten.angle.default,
            torch.ops.aten.angle,
        ),
    ),
    "conj": _unary_spec(
        "conj",
        (
            torch.conj,
            torch.ops.aten.conj.default,
            torch.ops.aten.conj,
        ),
    ),
    "conj_physical": _unary_spec(
        "conj_physical",
        (
            torch.conj_physical,
            torch.ops.aten.conj_physical.default,
            torch.ops.aten.conj_physical,
        ),
    ),
    "deg2rad": _unary_spec(
        "deg2rad",
        (
            torch.deg2rad,
            torch.ops.aten.deg2rad.default,
            torch.ops.aten.deg2rad,
        ),
    ),
    "digamma": _unary_spec(
        "digamma",
        (
            torch.digamma,
            torch.ops.aten.digamma.default,
            torch.ops.aten.digamma,
        ),
    ),
    "erfinv": _unary_spec(
        "erfinv",
        (
            torch.erfinv,
            torch.ops.aten.erfinv.default,
            torch.ops.aten.erfinv,
        ),
    ),
    "exp2": _unary_spec(
        "exp2",
        (
            torch.exp2,
            torch.ops.aten.exp2.default,
            torch.ops.aten.exp2,
        ),
    ),
    "frac": _unary_spec(
        "frac",
        (
            torch.frac,
            torch.ops.aten.frac.default,
            torch.ops.aten.frac,
        ),
    ),
    "i0": _unary_spec(
        "i0",
        (
            torch.i0,
            torch.ops.aten.i0.default,
            torch.ops.aten.i0,
        ),
    ),
    "lgamma": _unary_spec(
        "lgamma",
        (
            torch.lgamma,
            torch.ops.aten.lgamma.default,
            torch.ops.aten.lgamma,
        ),
    ),
    "logit": _unary_spec(
        "logit",
        (
            torch.logit,
            torch.ops.aten.logit.default,
            torch.ops.aten.logit,
        ),
    ),
    "nan_to_num": _unary_spec(
        "nan_to_num",
        (
            torch.nan_to_num,
            torch.ops.aten.nan_to_num.default,
            torch.ops.aten.nan_to_num,
        ),
    ),
    "positive": _unary_spec(
        "positive",
        (
            torch.positive,
            torch.ops.aten.positive.default,
            torch.ops.aten.positive,
        ),
    ),
    "rad2deg": _unary_spec(
        "rad2deg",
        (
            torch.rad2deg,
            torch.ops.aten.rad2deg.default,
            torch.ops.aten.rad2deg,
        ),
    ),
    "real": _unary_spec(
        "real",
        (
            torch.real,
            torch.ops.aten.real.default,
            torch.ops.aten.real,
        ),
    ),
    "sgn": _unary_spec(
        "sgn",
        (
            torch.sgn,
            torch.ops.aten.sgn.default,
            torch.ops.aten.sgn,
        ),
    ),
    "sinc": _unary_spec(
        "sinc",
        (
            torch.sinc,
            torch.ops.aten.sinc.default,
            torch.ops.aten.sinc,
        ),
    ),
    "square": _unary_spec(
        "square",
        (
            torch.square,
            torch.ops.aten.square.default,
            torch.ops.aten.square,
        ),
    ),
    "matmul": _OpSpec(
        name="matmul",
        kind="matmul",
        symbol=None,
        supported_targets={
            operator.matmul,
            torch.matmul,
            torch.ops.aten.mm,
            torch.ops.aten.mm.default,
            torch.ops.aten.matmul,
            torch.ops.aten.matmul.default,
        },
    ),
    "bmm": _OpSpec(
        name="bmm",
        kind="matmul",
        symbol=None,
        supported_targets={
            torch.bmm,
            torch.ops.aten.bmm,
            torch.ops.aten.bmm.default,
        },
    ),
}


TARGET_TO_OP: Dict[object, _OpSpec] = {
    target: spec
    for spec in SUPPORTED_OPS.values()
    for target in spec.supported_targets
}

INPLACE_TARGETS = {
    torch.ops.aten.atanh_.default: 0,
    torch.ops.aten.atanh_: 0,
    torch.ops.aten.add_.Tensor: 0,
    torch.ops.aten.add_: 0,
}


@dataclass
class _OpNode:
    node: torch.fx.Node
    spec: _OpSpec
    inputs: Tuple[torch.fx.Node, ...]
    output_shape: Tuple[int, ...]
    inplace_input: int | None = None


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


@dataclass
class _GenericLibrary:
    so_path: Path
    lib: object
    input_shapes: Tuple[Tuple[int, ...], ...]
    output_shape: Tuple[int, ...]

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = getattr(self.lib, "ref_codegen_main_f32")
        args = [tensor.data_ptr() for tensor in inputs]
        args.append(out.data_ptr())
        fn(*args)


_LIBRARY_CACHE: Dict[str, object] = {}
_C_SRC_DIR = Path(__file__).resolve().parents[2] / "csrc"


def _format_array_suffix(shape: Sequence[int]) -> str:
    return "".join(f"[{dim}]" for dim in shape) or "[1]"


def _write_elementwise_kernel(
    node_index: int, op_spec: _OpSpec, shape: Sequence[int]
) -> List[str]:
    array_suffix = _format_array_suffix(shape)
    if op_spec.kind == "binary":
        signature = (
            f"void node{node_index}_{op_spec.name}_f32(const float a{array_suffix}, "
            f"const float b{array_suffix}, float out{array_suffix}) {{"
        )
    else:
        signature = (
            f"void node{node_index}_{op_spec.name}_f32(const float a{array_suffix}, "
            f"float out{array_suffix}) {{"
        )
    lines = [signature]
    indent = "    "
    if shape:
        for dim, size in enumerate(shape):
            lines.append(
                f"{indent}for (int64_t i{dim} = 0; i{dim} < {size}; ++i{dim}) {{"
            )
            indent += "    "
    index_expr = "".join(f"[i{dim}]" for dim in range(len(shape))) or "[0]"
    scalar_fn = f"ref_scalar_f32_{op_spec.name}"
    if op_spec.kind == "binary":
        lines.append(
            f"{indent}out{index_expr} = {scalar_fn}(a{index_expr}, b{index_expr});"
        )
    else:
        lines.append(f"{indent}out{index_expr} = {scalar_fn}(a{index_expr});")
    if shape:
        for _ in range(len(shape)):
            indent = indent[:-4]
            lines.append(f"{indent}}}")
    lines.append("}")
    return lines


def _write_matmul_kernel(
    node_index: int,
    op_spec: _OpSpec,
    a_shape: Sequence[int],
    b_shape: Sequence[int],
) -> List[str]:
    if op_spec.name == "matmul":
        m, k = a_shape
        _, n = b_shape
        a_suffix = _format_array_suffix((m, k))
        b_suffix = _format_array_suffix((k, n))
        out_suffix = _format_array_suffix((m, n))
        lines = [
            f"void node{node_index}_{op_spec.name}_f32(const float a{a_suffix}, const float b{b_suffix}, float out{out_suffix}) {{",
            f"    for (int64_t i = 0; i < {m}; ++i) {{",
            f"        for (int64_t j = 0; j < {n}; ++j) {{",
            "            float acc = 0.0f;",
            f"            for (int64_t t = 0; t < {k}; ++t) {{",
            "                acc += a[i][t] * b[t][j];",
            "            }",
            "            out[i][j] = acc;",
            "        }",
            "    }",
            "}",
        ]
        return lines
    batch, m, k = a_shape
    _, _, n = b_shape
    a_suffix = _format_array_suffix((batch, m, k))
    b_suffix = _format_array_suffix((batch, k, n))
    out_suffix = _format_array_suffix((batch, m, n))
    lines = [
        f"void node{node_index}_{op_spec.name}_f32(const float a{a_suffix}, const float b{b_suffix}, float out{out_suffix}) {{",
        f"    for (int64_t b_idx = 0; b_idx < {batch}; ++b_idx) {{",
        f"        for (int64_t i = 0; i < {m}; ++i) {{",
        f"            for (int64_t j = 0; j < {n}; ++j) {{",
        "                float acc = 0.0f;",
        f"                for (int64_t t = 0; t < {k}; ++t) {{",
        "                    acc += a[b_idx][i][t] * b[b_idx][t][j];",
        "                }",
        "                out[b_idx][i][j] = acc;",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return lines


def _write_generic_source(graph: _GenericGraph) -> str:
    placeholders = graph.tensor_placeholders
    op_nodes = graph.op_nodes
    lines = [
        "#include <stdint.h>",
        "#include \"ops_scalar_f32.h\"",
        "",
    ]
    for index, op_node in enumerate(op_nodes, start=1):
        if op_node.spec.kind in {"binary", "unary"}:
            lines.extend(_write_elementwise_kernel(index, op_node.spec, op_node.output_shape))
        else:
            lhs, rhs = op_node.inputs
            lhs_shape = graph.shapes[lhs]
            rhs_shape = graph.shapes[rhs]
            lines.extend(
                _write_matmul_kernel(index, op_node.spec, lhs_shape, rhs_shape)
            )
        lines.append("")
    input_args = ", ".join(
        [
            f"const float input_{idx}{_format_array_suffix(graph.shapes[node])}"
            for idx, node in enumerate(placeholders)
        ]
    )
    input_args = f"{input_args}, " if input_args else ""
    lines.append(
        "void ref_codegen_main_f32("
        f"{input_args}float out{_format_array_suffix(graph.shapes[graph.output_value])}) {{"
    )
    name_map: Dict[torch.fx.Node, str] = {}
    for idx, placeholder in enumerate(placeholders):
        name_map[placeholder] = f"input_{idx}"
    temp_index = 0
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
        lines.append(
            f"    float {temp_name}{_format_array_suffix(op_node.output_shape)};"
        )
    for index, op_node in enumerate(op_nodes, start=1):
        input_names = [name_map[arg] for arg in op_node.inputs]
        output_name = name_map[op_node.node]
        args = ", ".join([*input_names, output_name])
        lines.append(f"    node{index}_{op_node.spec.name}_f32({args});")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _validate_example_inputs(example_inputs: Sequence[torch.Tensor]) -> None:
    tensor_examples = [example for example in example_inputs if isinstance(example, torch.Tensor)]
    if not tensor_examples:
        raise RefBackendError("codegen backend requires at least one example tensor input")
    for example in tensor_examples:
        if example.dtype is not torch.float32:
            raise RefBackendError("codegen backend supports only torch.float32 tensors")
        if example.device.type != "cpu":
            raise RefBackendError("codegen backend supports only CPU tensors")


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
        a_shape, b_shape = input_shapes
        if a_shape != b_shape:
            raise RefBackendError(
                f"codegen {op_spec.name} requires inputs to have identical shapes"
            )
        return a_shape
    if op_spec.kind == "unary":
        return input_shapes[0]
    a_shape, b_shape = input_shapes
    if op_spec.name == "matmul":
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise RefBackendError("codegen matmul requires 2D inputs")
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


def _analyze_generic_graph(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> _GenericGraph:
    _validate_example_inputs(example_inputs)
    output_node = None
    placeholders: List[torch.fx.Node] = []
    tensor_placeholders: List[torch.fx.Node] = []
    op_nodes: List[_OpNode] = []
    shapes: Dict[torch.fx.Node, Tuple[int, ...]] = {}
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
                shapes[node] = tuple(example.shape)
                tensor_placeholders.append(node)
            continue
        if node.op == "call_function":
            if node.kwargs:
                raise RefBackendError("codegen backend expects positional args only")
            op_spec = TARGET_TO_OP.get(node.target)
            if op_spec is None:
                raise RefBackendError(f"Unsupported call_function: {node.target}")
            inplace_input = INPLACE_TARGETS.get(node.target)
            expected_arity = 1 if op_spec.kind == "unary" else 2
            if len(node.args) != expected_arity:
                if expected_arity == 1:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects one input"
                    )
                raise RefBackendError(
                    f"codegen {op_spec.name} expects exactly two inputs"
                )
            input_nodes: List[torch.fx.Node] = []
            input_shapes: List[Tuple[int, ...]] = []
            for arg in node.args:
                if not isinstance(arg, torch.fx.Node):
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects tensor inputs only"
                    )
                if arg not in shapes:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects tensor inputs only"
                    )
                input_nodes.append(arg)
                input_shapes.append(shapes[arg])
            output_shape = _infer_output_shape(op_spec, input_shapes)
            shapes[node] = output_shape
            op_nodes.append(
                _OpNode(
                    node=node,
                    spec=op_spec,
                    inputs=tuple(input_nodes),
                    output_shape=output_shape,
                    inplace_input=inplace_input,
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
    lib.ref_codegen_main_f32.argtypes = argtypes
    lib.ref_codegen_main_f32.restype = None

    input_shapes = tuple(graph.shapes[node] for node in graph.tensor_placeholders)
    compiled = _GenericLibrary(
        so_path=so_path,
        lib=lib,
        input_shapes=input_shapes,
        output_shape=graph.shapes[graph.output_value],
    )
    _LIBRARY_CACHE[digest] = compiled
    return compiled


def _validate_runtime_inputs(inputs: Iterable[torch.Tensor]) -> None:
    for tensor in inputs:
        if tensor.dtype is not torch.float32:
            raise RefBackendError("codegen backend supports only torch.float32 tensors")
        if tensor.device.type != "cpu":
            raise RefBackendError("codegen backend supports only CPU tensors")


def _compile_graph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    graph = _analyze_generic_graph(gm, example_inputs)
    lib = _compile_generic_library(graph)
    output_structure = graph.output_structure
    output_value = graph.output_value
    output_inplace_input = graph.output_inplace_input

    def resolve_output(value: object, env: Dict[torch.fx.Node, object]) -> object:
        if isinstance(value, torch.fx.Node):
            return env[value]
        if isinstance(value, (list, tuple, immutable_list)):
            resolved = [resolve_output(item, env) for item in value]
            return type(value)(resolved)
        return value

    def compiled(*args: object) -> object:
        if len(args) != len(graph.placeholders):
            raise RefBackendError(
                f"codegen backend expects {len(graph.placeholders)} inputs, got {len(args)}"
            )
        env: Dict[torch.fx.Node, object] = {}
        input_tensors = []
        for node, value in zip(graph.placeholders, args):
            env[node] = value
            if node in graph.tensor_placeholders:
                if not isinstance(value, torch.Tensor):
                    raise RefBackendError("codegen backend expects tensor inputs only")
                input_tensors.append(value)
        _validate_runtime_inputs(input_tensors)
        expected_shapes = lib.input_shapes
        for tensor, expected in zip(input_tensors, expected_shapes):
            if tuple(tensor.shape) != expected:
                raise RefBackendError(
                    f"codegen backend requires inputs to have shapes {expected_shapes}"
                )
        contiguous_inputs = [tensor.contiguous() for tensor in input_tensors]
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
            out = torch.empty(
                lib.output_shape,
                dtype=contiguous_inputs[0].dtype,
                device=contiguous_inputs[0].device,
            )
            lib.run(contiguous_inputs, out)
            env[output_value] = out
        return resolve_output(output_structure, env)

    return compiled


def get_generic_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> str:
    graph = _analyze_generic_graph(gm, example_inputs)
    return _write_generic_source(graph)


def codegen_generic_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    return _compile_graph(gm, example_inputs)
