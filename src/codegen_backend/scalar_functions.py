from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Set

import torch

from codegen_backend.dtypes import _CODEGEN_DTYPES
from codegen_backend.errors import CodegenBackendError


@dataclass(frozen=True)
class _ScalarTypeInfo:
    torch_dtype: torch.dtype
    c_type: str
    prefix: str
    suffix: str
    is_float: bool
    is_bool: bool
    is_signed: bool
    is_small_int: bool
    bits: int | None


@dataclass(frozen=True)
class _GeneratedScalar:
    lines: List[str]
    deps: Set[str]
    includes: Set[str]


class ScalarType(str, Enum):
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    BOOL = "bool"

    @classmethod
    def from_torch_dtype(cls, dtype: torch.dtype) -> "ScalarType":
        mapping = {
            torch.float32: cls.F32,
            torch.float64: cls.F64,
            torch.int8: cls.I8,
            torch.int16: cls.I16,
            torch.int32: cls.I32,
            torch.int64: cls.I64,
            torch.uint8: cls.U8,
            torch.uint16: cls.U16,
            torch.uint32: cls.U32,
            torch.uint64: cls.U64,
            torch.bool: cls.BOOL,
        }
        try:
            return mapping[dtype]
        except KeyError as exc:
            raise CodegenBackendError(
                f"unsupported torch dtype for scalar functions: {dtype}"
            ) from exc


class ScalarFunction(str, Enum):
    ABS = "abs"
    ABSOLUTE = "absolute"
    ACOS = "acos"
    ACOSH = "acosh"
    ADD = "add"
    ANGLE = "angle"
    ARCCOS = "arccos"
    ARCSIN = "arcsin"
    ARCSINH = "arcsinh"
    ARCTAN = "arctan"
    ASIN = "asin"
    ASINH = "asinh"
    ATAN = "atan"
    ATAN2 = "atan2"
    ATANH = "atanh"
    BITWISE_AND = "bitwise_and"
    BITWISE_LEFT_SHIFT = "bitwise_left_shift"
    BITWISE_NOT = "bitwise_not"
    BITWISE_OR = "bitwise_or"
    BITWISE_RIGHT_SHIFT = "bitwise_right_shift"
    BITWISE_XOR = "bitwise_xor"
    CBRT = "cbrt"
    CEIL = "ceil"
    CLAMP_MAX = "clamp_max"
    CLAMP_MIN = "clamp_min"
    CONJ = "conj"
    CONJ_PHYSICAL = "conj_physical"
    COPYSIGN = "copysign"
    COS = "cos"
    COSH = "cosh"
    DEG2RAD = "deg2rad"
    DIGAMMA = "digamma"
    DIV = "div"
    ELU = "elu"
    EQ = "eq"
    ERF = "erf"
    ERFC = "erfc"
    ERFINV = "erfinv"
    EXP = "exp"
    EXP2 = "exp2"
    EXPM1 = "expm1"
    FLOOR = "floor"
    FLOOR_DIVIDE = "floor_divide"
    FMAX = "fmax"
    FMIN = "fmin"
    FMOD = "fmod"
    FRAC = "frac"
    GE = "ge"
    GELU = "gelu"
    GT = "gt"
    HARDSIGMOID = "hardsigmoid"
    HARDSWISH = "hardswish"
    HEAVISIDE = "heaviside"
    HYPOT = "hypot"
    I0 = "i0"
    ISFINITE = "isfinite"
    ISINF = "isinf"
    ISNAN = "isnan"
    ISNEGINF = "isneginf"
    ISPOSINF = "isposinf"
    LDEXP = "ldexp"
    LE = "le"
    LEAKY_RELU = "leaky_relu"
    LGAMMA = "lgamma"
    LOG = "log"
    LOG10 = "log10"
    LOG1P = "log1p"
    LOG2 = "log2"
    LOG_SIGMOID = "log_sigmoid"
    LOGADDEXP = "logaddexp"
    LOGADDEXP2 = "logaddexp2"
    LOGICAL_AND = "logical_and"
    LOGICAL_NOT = "logical_not"
    LOGICAL_OR = "logical_or"
    LOGICAL_XOR = "logical_xor"
    LOGIT = "logit"
    LT = "lt"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MISH = "mish"
    MUL = "mul"
    NAN_TO_NUM = "nan_to_num"
    NE = "ne"
    NEG = "neg"
    NEXTAFTER = "nextafter"
    POSITIVE = "positive"
    POW = "pow"
    RAD2DEG = "rad2deg"
    REAL = "real"
    RECIPROCAL = "reciprocal"
    RELU = "relu"
    RELU6 = "relu6"
    REMAINDER = "remainder"
    ROUND = "round"
    RSQRT = "rsqrt"
    SELU = "selu"
    SGN = "sgn"
    SIGMOID = "sigmoid"
    SIGN = "sign"
    SILU = "silu"
    SIN = "sin"
    SINC = "sinc"
    SINH = "sinh"
    SOFTPLUS = "softplus"
    SQRT = "sqrt"
    SQUARE = "square"
    SUB = "sub"
    TAN = "tan"
    TANH = "tanh"
    TRUNC = "trunc"
    XLOGY = "xlogy"
    CONVERT_FROM_F32 = "convert_from_f32"
    CONVERT_FROM_F64 = "convert_from_f64"
    CONVERT_FROM_I8 = "convert_from_i8"
    CONVERT_FROM_I16 = "convert_from_i16"
    CONVERT_FROM_I32 = "convert_from_i32"
    CONVERT_FROM_I64 = "convert_from_i64"
    CONVERT_FROM_U8 = "convert_from_u8"
    CONVERT_FROM_U16 = "convert_from_u16"
    CONVERT_FROM_U32 = "convert_from_u32"
    CONVERT_FROM_U64 = "convert_from_u64"
    CONVERT_FROM_BOOL = "convert_from_bool"

    @classmethod
    def from_op_name(cls, op_name: str) -> "ScalarFunction":
        try:
            return cls(op_name)
        except ValueError as exc:
            raise CodegenBackendError(
                f"unknown scalar function op name: {op_name}"
            ) from exc


@dataclass(frozen=True)
class ScalarFunctionKey:
    function: ScalarFunction
    return_type: ScalarType

    @classmethod
    def for_torch_dtype(
        cls, function: ScalarFunction, dtype: torch.dtype
    ) -> "ScalarFunctionKey":
        return cls(function=function, return_type=ScalarType.from_torch_dtype(dtype))


FLOAT_ONLY_OPS = frozenset(
    [
        ScalarFunction.ABS,
        ScalarFunction.ABSOLUTE,
        ScalarFunction.ADD,
        ScalarFunction.SUB,
        ScalarFunction.MUL,
        ScalarFunction.DIV,
        ScalarFunction.MAXIMUM,
        ScalarFunction.MINIMUM,
        ScalarFunction.LE,
        ScalarFunction.LT,
        ScalarFunction.GE,
        ScalarFunction.GT,
        ScalarFunction.EQ,
        ScalarFunction.NE,
        ScalarFunction.LOGICAL_OR,
        ScalarFunction.LOGICAL_AND,
        ScalarFunction.LOGICAL_XOR,
        ScalarFunction.LOGICAL_NOT,
        ScalarFunction.FMAX,
        ScalarFunction.FMIN,
        ScalarFunction.COPYSIGN,
        ScalarFunction.HYPOT,
        ScalarFunction.ATAN2,
        ScalarFunction.POW,
        ScalarFunction.FMOD,
        ScalarFunction.REMAINDER,
        ScalarFunction.FLOOR_DIVIDE,
        ScalarFunction.LOGADDEXP,
        ScalarFunction.LOGADDEXP2,
        ScalarFunction.NEXTAFTER,
        ScalarFunction.XLOGY,
        ScalarFunction.HEAVISIDE,
        ScalarFunction.LDEXP,
        ScalarFunction.CLAMP_MIN,
        ScalarFunction.CLAMP_MAX,
        ScalarFunction.NEG,
        ScalarFunction.RECIPROCAL,
        ScalarFunction.RELU,
        ScalarFunction.CEIL,
        ScalarFunction.FLOOR,
        ScalarFunction.SIN,
        ScalarFunction.COS,
        ScalarFunction.SQRT,
        ScalarFunction.CBRT,
        ScalarFunction.EXP,
        ScalarFunction.TANH,
        ScalarFunction.LOG,
        ScalarFunction.ACOS,
        ScalarFunction.ARCCOS,
        ScalarFunction.ACOSH,
        ScalarFunction.ASIN,
        ScalarFunction.ARCSIN,
        ScalarFunction.ASINH,
        ScalarFunction.ARCSINH,
        ScalarFunction.ATAN,
        ScalarFunction.ARCTAN,
        ScalarFunction.ATANH,
        ScalarFunction.COSH,
        ScalarFunction.SINH,
        ScalarFunction.TAN,
        ScalarFunction.ERF,
        ScalarFunction.ERFC,
        ScalarFunction.EXPM1,
        ScalarFunction.LOG1P,
        ScalarFunction.LOG2,
        ScalarFunction.LOG10,
        ScalarFunction.ISFINITE,
        ScalarFunction.RSQRT,
        ScalarFunction.SIGMOID,
        ScalarFunction.LOG_SIGMOID,
        ScalarFunction.GELU,
        ScalarFunction.ELU,
        ScalarFunction.LEAKY_RELU,
        ScalarFunction.SOFTPLUS,
        ScalarFunction.SILU,
        ScalarFunction.MISH,
        ScalarFunction.SELU,
        ScalarFunction.RELU6,
        ScalarFunction.HARDSIGMOID,
        ScalarFunction.HARDSWISH,
        ScalarFunction.SIGN,
        ScalarFunction.ROUND,
        ScalarFunction.TRUNC,
        ScalarFunction.ANGLE,
        ScalarFunction.CONJ,
        ScalarFunction.CONJ_PHYSICAL,
        ScalarFunction.DEG2RAD,
        ScalarFunction.DIGAMMA,
        ScalarFunction.ERFINV,
        ScalarFunction.EXP2,
        ScalarFunction.FRAC,
        ScalarFunction.I0,
        ScalarFunction.LGAMMA,
        ScalarFunction.LOGIT,
        ScalarFunction.ISNAN,
        ScalarFunction.ISINF,
        ScalarFunction.ISNEGINF,
        ScalarFunction.ISPOSINF,
        ScalarFunction.NAN_TO_NUM,
        ScalarFunction.POSITIVE,
        ScalarFunction.RAD2DEG,
        ScalarFunction.REAL,
        ScalarFunction.SGN,
        ScalarFunction.SINC,
        ScalarFunction.SQUARE,
    ]
)


COMMON_BASE_OPS = frozenset(
    [
        "from_f32",
        "abs",
        "add",
        "sub",
        "mul",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
        "div",
        "maximum",
        "minimum",
        "le",
        "lt",
        "ge",
        "gt",
        "eq",
        "ne",
        "logical_or",
        "logical_and",
        "logical_xor",
        "logical_not",
        "fmax",
        "fmin",
        "copysign",
        "fmod",
        "remainder",
        "floor_divide",
        "clamp_min",
        "clamp_max",
        "neg",
        "reciprocal",
        "relu",
        "ceil",
        "floor",
        "round",
    ]
)


INT_ONLY_OPS = frozenset(
    [
        "bitwise_left_shift",
        "bitwise_right_shift",
    ]
)


SIGNED_ONLY_OPS = frozenset(
    [
        "trunc",
        "frac",
        "sign",
        "conj",
        "conj_physical",
        "positive",
        "real",
        "sgn",
        "square",
    ]
)


UNSIGNED_ONLY_OPS = frozenset()


BOOL_ONLY_OPS = frozenset(["to_f32"])


SIGNED_INT_BASE_OPS = COMMON_BASE_OPS | INT_ONLY_OPS | SIGNED_ONLY_OPS


UNSIGNED_INT_BASE_OPS = COMMON_BASE_OPS | INT_ONLY_OPS | UNSIGNED_ONLY_OPS


BOOL_BASE_OPS = COMMON_BASE_OPS | BOOL_ONLY_OPS


COMMON_UNARY_FROM_F32_OPS = frozenset(
    [
        "acos",
        "acosh",
        "angle",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cbrt",
        "cos",
        "cosh",
        "deg2rad",
        "digamma",
        "erf",
        "erfc",
        "erfinv",
        "exp",
        "exp2",
        "expm1",
        "i0",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "isfinite",
        "isnan",
        "logit",
        "log_sigmoid",
        "gelu",
        "elu",
        "leaky_relu",
        "softplus",
        "isinf",
        "isneginf",
        "isposinf",
        "nan_to_num",
        "rad2deg",
        "rsqrt",
        "sigmoid",
        "selu",
        "relu6",
        "hardsigmoid",
        "silu",
        "mish",
        "hardswish",
        "sin",
        "sinc",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ]
)


BOOL_ONLY_UNARY_FROM_F32_OPS = frozenset(
    [
        "abs",
        "neg",
        "reciprocal",
        "relu",
        "ceil",
        "floor",
        "sign",
        "round",
        "trunc",
        "conj",
        "conj_physical",
        "frac",
        "positive",
        "real",
        "sgn",
        "square",
    ]
)


INT_UNARY_FROM_F32_OPS = COMMON_UNARY_FROM_F32_OPS


BOOL_UNARY_FROM_F32_OPS = COMMON_UNARY_FROM_F32_OPS | BOOL_ONLY_UNARY_FROM_F32_OPS


COMMON_BINARY_FROM_F32_OPS = frozenset(
    [
        "atan2",
        "heaviside",
        "hypot",
        "ldexp",
        "logaddexp",
        "logaddexp2",
        "nextafter",
        "pow",
        "xlogy",
    ]
)


BOOL_ONLY_BINARY_FROM_F32_OPS = frozenset(
    [
        "add",
        "sub",
        "mul",
        "div",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "copysign",
        "fmod",
        "remainder",
        "floor_divide",
        "clamp_min",
        "clamp_max",
    ]
)


INT_BINARY_FROM_F32_OPS = COMMON_BINARY_FROM_F32_OPS


BOOL_BINARY_FROM_F32_OPS = COMMON_BINARY_FROM_F32_OPS | BOOL_ONLY_BINARY_FROM_F32_OPS


_OP_ALIASES = {
    "absolute": "abs",
    "arccos": "acos",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
}


_NO_SUFFIX_MATH = {"isfinite", "isnan", "isinf", "signbit"}


def _float_literal(value: float, dtype_info: _ScalarTypeInfo) -> str:
    if dtype_info.suffix == "f32":
        if value == int(value):
            return f"{int(value)}.0f"
        literal = f"{value}"
        if "e" in literal or "E" in literal:
            return f"{literal}f"
        if "." not in literal:
            literal = f"{literal}.0"
        return f"{literal}f"
    if value == int(value):
        return f"{int(value)}.0"
    literal = f"{value}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return literal


def _math_fn(base: str, dtype_info: _ScalarTypeInfo) -> str:
    if dtype_info.suffix == "f32" and base not in _NO_SUFFIX_MATH:
        return f"{base}f"
    return base


def _normalize_op_name(op_name: str) -> str:
    return _OP_ALIASES.get(op_name, op_name)


def _cast_value(expr: str, dtype_info: _ScalarTypeInfo) -> str:
    if dtype_info.is_small_int:
        return f"({dtype_info.c_type})({expr})"
    return expr


def _simple_unary(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _simple_binary(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_unary_math(dtype_info: _ScalarTypeInfo, name: str, base: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, f"{_math_fn(base, dtype_info)}(a)")


def _float_binary_math(dtype_info: _ScalarTypeInfo, name: str, base: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, f"{_math_fn(base, dtype_info)}(a, b)")


def _float_isfinite(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "isfinite", f"isfinite(a) ? {one} : {zero}")


def _float_isnan(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "isnan", f"isnan(a) ? {one} : {zero}")


def _float_isinf(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "isinf", f"isinf(a) ? {one} : {zero}")


def _float_isneginf(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(
        dtype_info, "isneginf", f"(isinf(a) && signbit(a)) ? {one} : {zero}"
    )


def _float_isposinf(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(
        dtype_info, "isposinf", f"(isinf(a) && !signbit(a)) ? {one} : {zero}"
    )


def _float_comparison(
    dtype_info: _ScalarTypeInfo, name: str, op: str
) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_binary(dtype_info, name, f"a {op} b ? {one} : {zero}")


def _float_logical_binary(
    dtype_info: _ScalarTypeInfo, name: str, expr: str
) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_binary(dtype_info, name, f"{expr} ? {one} : {zero}")


def _float_logical_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "logical_not", f"a == {zero} ? {one} : {zero}")


def _float_remainder(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    nan = "NAN"
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}remainder({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        f"        return {nan};",
        "    }",
        f"    if (b == {_float_literal(0.0, dtype_info)}) {{",
        f"        return {nan};",
        "    }",
        f"    {dtype_info.c_type} mod = {_math_fn('fmod', dtype_info)}(a, b);",
        f"    if (mod == {_float_literal(0.0, dtype_info)}) {{",
        "        return mod;",
        "    }",
        "    if ((mod < 0) != (b < 0)) {",
        "        mod += b;",
        "    }",
        "    return mod;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_floor_divide(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(
        dtype_info, "floor_divide", f"{_math_fn('floor', dtype_info)}(a / b)"
    )


def _float_logaddexp(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}logaddexp({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        "        return NAN;",
        "    }",
        f"    {dtype_info.c_type} max_val = {_math_fn('fmax', dtype_info)}(a, b);",
        f"    {dtype_info.c_type} min_val = {_math_fn('fmin', dtype_info)}(a, b);",
        "    if (max_val == -INFINITY) {",
        "        return -INFINITY;",
        "    }",
        f"    return max_val + {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(min_val - max_val));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_logaddexp2(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}logaddexp2({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        "        return NAN;",
        "    }",
        f"    {dtype_info.c_type} max_val = {_math_fn('fmax', dtype_info)}(a, b);",
        f"    {dtype_info.c_type} min_val = {_math_fn('fmin', dtype_info)}(a, b);",
        "    if (max_val == -INFINITY) {",
        "        return -INFINITY;",
        "    }",
        f"    return max_val + {_math_fn('log2', dtype_info)}({one} + {_math_fn('exp2', dtype_info)}(min_val - max_val));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_xlogy(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}xlogy({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        "        return NAN;",
        "    }",
        f"    if (a == {zero}) {{",
        f"        return {zero};",
        "    }",
        f"    return a * {_math_fn('log', dtype_info)}(b);",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_heaviside(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}heaviside({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        f"    if (a > {zero}) {{",
        f"        return {one};",
        "    }",
        f"    if (a == {zero}) {{",
        "        return b;",
        "    }",
        f"    return {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_ldexp(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(
        dtype_info, "ldexp", f"a * {_math_fn('exp2', dtype_info)}(b)"
    )


def _float_reciprocal(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "reciprocal", f"{one} / a")


def _float_relu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "relu", f"a > {zero} ? a : {zero}")


def _float_rsqrt(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "rsqrt", f"{one} / {_math_fn('sqrt', dtype_info)}(a)")


def _float_sigmoid(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "sigmoid", f"{one} / ({one} + {_math_fn('exp', dtype_info)}(-a))")


def _float_log_sigmoid(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}log_sigmoid({dtype_info.c_type} a) {{",
        f"    if (a >= {zero}) {{",
        f"        return -{_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(-a));",
        "    }",
        f"    return a - {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(a));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_gelu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    inv_sqrt2 = _float_literal(0.7071067811865475, dtype_info)
    half = _float_literal(0.5, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}gelu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} inv_sqrt2 = {inv_sqrt2};",
        f"    return {half} * a * ({one} + {_math_fn('erf', dtype_info)}(a * inv_sqrt2));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_elu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}elu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {one};",
        f"    const {dtype_info.c_type} scale = {one};",
        f"    const {dtype_info.c_type} input_scale = {one};",
        "    if (a > 0) {",
        "        return scale * a;",
        "    }",
        f"    return scale * alpha * ({_math_fn('exp', dtype_info)}(input_scale * a) - {one});",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_leaky_relu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    negative_slope = _float_literal(0.01, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}leaky_relu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} negative_slope = {negative_slope};",
        "    return a > 0 ? a : negative_slope * a;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_softplus(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    beta = _float_literal(1.0, dtype_info)
    threshold = _float_literal(20.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}softplus({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} beta = {beta};",
        f"    const {dtype_info.c_type} threshold = {threshold};",
        "    if (beta * a > threshold) {",
        "        return a;",
        "    }",
        f"    return {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(beta * a)) / beta;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_silu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "silu", f"a / ({one} + {_math_fn('exp', dtype_info)}(-a))")


def _float_mish(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    twenty = _float_literal(20.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}mish({dtype_info.c_type} a) {{",
        f"    if (a > {twenty}) {{",
        "        return a;",
        "    }",
        f"    if (a < -{twenty}) {{",
        f"        {dtype_info.c_type} exp_a = {_math_fn('exp', dtype_info)}(a);",
        "        return a * exp_a;",
        "    }",
        f"    {dtype_info.c_type} softplus = {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(a));",
        f"    return a * {_math_fn('tanh', dtype_info)}(softplus);",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_selu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    alpha = "1.6732632423543772848170429916717"
    scale = "1.0507009873554804934193349852946"
    alpha_literal = f"{alpha}f" if dtype_info.suffix == "f32" else alpha
    scale_literal = f"{scale}f" if dtype_info.suffix == "f32" else scale
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}selu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha_literal};",
        f"    const {dtype_info.c_type} scale = {scale_literal};",
        "    if (a > 0) {",
        "        return scale * a;",
        "    }",
        f"    return scale * alpha * ({_math_fn('exp', dtype_info)}(a) - {one});",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_relu6(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    six = _float_literal(6.0, dtype_info)
    return _simple_unary(
        dtype_info,
        "relu6",
        f"{_math_fn('fmin', dtype_info)}({six}, {_math_fn('fmax', dtype_info)}({zero}, a))",
    )


def _float_hardsigmoid(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    three = _float_literal(3.0, dtype_info)
    six = _float_literal(6.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}hardsigmoid({dtype_info.c_type} a) {{",
        f"    {dtype_info.c_type} shifted = a + {three};",
        f"    {dtype_info.c_type} clamped = {_math_fn('fmin', dtype_info)}({six}, {_math_fn('fmax', dtype_info)}({zero}, shifted));",
        f"    return clamped / {six};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_hardswish(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    three = _float_literal(3.0, dtype_info)
    six = _float_literal(6.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}hardswish({dtype_info.c_type} a) {{",
        f"    {dtype_info.c_type} shifted = a + {three};",
        f"    {dtype_info.c_type} clamped = {_math_fn('fmin', dtype_info)}({six}, {_math_fn('fmax', dtype_info)}({zero}, shifted));",
        f"    return a * clamped / {six};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_sign(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    minus_one = _float_literal(-1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sign({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        "        return a;",
        "    }",
        f"    if (a > {zero}) {{",
        f"        return {one};",
        "    }",
        f"    if (a < {zero}) {{",
        f"        return {minus_one};",
        "    }",
        f"    return {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_round(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _float_unary_math(dtype_info, "round", "round")


def _float_trunc(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _float_unary_math(dtype_info, "trunc", "trunc")


def _float_angle(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}angle({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        "        return a;",
        "    }",
        f"    return a < {zero} ? {pi} : {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_conj(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, "a")


def _float_deg2rad(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    one_eighty = _float_literal(180.0, dtype_info)
    return _simple_unary(dtype_info, "deg2rad", f"a * ({pi} / {one_eighty})")


def _float_rad2deg(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    one_eighty = _float_literal(180.0, dtype_info)
    return _simple_unary(dtype_info, "rad2deg", f"a * ({one_eighty} / {pi})")


def _float_digamma_f64() -> _GeneratedScalar:
    lines = [
        "static inline double ref_scalar_f64_digamma(double x) {",
        "    if (isnan(x) || isinf(x)) {",
        "        return x;",
        "    }",
        "    if (x <= 0.0) {",
        "        double frac = x - floor(x);",
        "        if (frac == 0.0) {",
        "            return NAN;",
        "        }",
        "        return ref_scalar_f64_digamma(1.0 - x) - REF_PI_D / tan(REF_PI_D * x);",
        "    }",
        "    double result = 0.0;",
        "    while (x < 10.0) {",
        "        result -= 1.0 / x;",
        "        x += 1.0;",
        "    }",
        "    double inv = 1.0 / x;",
        "    double inv2 = inv * inv;",
        "    result += log(x) - 0.5 * inv",
        "        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0",
        "        - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0",
        "        - inv2 * (1.0 / 132.0 - inv2 * (691.0 / 32760.0))))));",
        "    return result;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_digamma(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if dtype_info.suffix == "f64":
        return _float_digamma_f64()
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}digamma({dtype_info.c_type} x) {{",
        "    return (float)ref_scalar_f64_digamma((double)x);",
        "}",
    ]
    deps = {"ref_scalar_f64_digamma"}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _float_erfinv(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    two = _float_literal(2.0, dtype_info)
    a_literal = _float_literal(0.147, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}erfinv({dtype_info.c_type} x) {{",
        "    if (isnan(x)) {",
        "        return x;",
        "    }",
        f"    if (x <= -{one}) {{",
        f"        return x == -{one} ? -INFINITY : NAN;",
        "    }",
        f"    if (x >= {one}) {{",
        f"        return x == {one} ? INFINITY : NAN;",
        "    }",
        f"    if (x == {zero}) {{",
        f"        return {zero};",
        "    }",
        f"    {dtype_info.c_type} a = {a_literal};",
        f"    {dtype_info.c_type} ln = {_math_fn('log', dtype_info)}({one} - x * x);",
        f"    {dtype_info.c_type} term = {two} / ({pi} * a) + ln / {two};",
        f"    {dtype_info.c_type} inner = term * term - ln / a;",
        f"    {dtype_info.c_type} approx = {_math_fn('sqrt', dtype_info)}({_math_fn('fmax', dtype_info)}({zero}, {_math_fn('sqrt', dtype_info)}(inner) - term));",
        f"    if (x < {zero}) {{",
        "        approx = -approx;",
        "    }",
        "    for (int i = 0; i < 2; ++i) {",
        f"        {dtype_info.c_type} err = {_math_fn('erf', dtype_info)}(approx) - x;",
        f"        {dtype_info.c_type} deriv = {two} / {_math_fn('sqrt', dtype_info)}({pi}) * {_math_fn('exp', dtype_info)}(-approx * approx);",
        "        approx -= err / deriv;",
        "    }",
        "    return approx;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_frac(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "frac", f"a - {_math_fn('trunc', dtype_info)}(a)")


def _float_i0(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    three_seven_five = _float_literal(3.75, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}i0({dtype_info.c_type} x) {{",
        f"    {dtype_info.c_type} ax = {_math_fn('fabs', dtype_info)}(x);",
        f"    if (ax < {three_seven_five}) {{",
        f"        {dtype_info.c_type} y = x / {three_seven_five};",
        "        y *= y;",
        f"        return { _float_literal(1.0, dtype_info)} + y * ({_float_literal(3.5156229, dtype_info)} + y * ({_float_literal(3.0899424, dtype_info)} + y * ({_float_literal(1.2067492, dtype_info)}",
        f"            + y * ({_float_literal(0.2659732, dtype_info)} + y * ({_float_literal(0.0360768, dtype_info)} + y * {_float_literal(0.0045813, dtype_info)})))));",
        "    }",
        f"    {dtype_info.c_type} y = {three_seven_five} / ax;",
        f"    return ({_math_fn('exp', dtype_info)}(ax) / {_math_fn('sqrt', dtype_info)}(ax)) * ({_float_literal(0.39894228, dtype_info)} + y * ({_float_literal(0.01328592, dtype_info)}",
        f"        + y * ({_float_literal(0.00225319, dtype_info)} + y * ({_float_literal(-0.00157565, dtype_info)} + y * ({_float_literal(0.00916281, dtype_info)}",
        f"        + y * ({_float_literal(-0.02057706, dtype_info)} + y * ({_float_literal(0.02635537, dtype_info)}",
        f"        + y * ({_float_literal(-0.01647633, dtype_info)} + y * {_float_literal(0.00392377, dtype_info)}))))))));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_logit(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}logit({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        "        return a;",
        "    }",
        f"    if (a == {zero}) {{",
        "        return -INFINITY;",
        "    }",
        f"    if (a == {one}) {{",
        "        return INFINITY;",
        "    }",
        f"    if (a < {zero} || a > {one}) {{",
        "        return NAN;",
        "    }",
        f"    return {_math_fn('log', dtype_info)}(a / ({one} - a));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_nan_to_num(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}nan_to_num({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        f"        return {zero};",
        "    }",
        "    if (isinf(a)) {",
        "        return signbit(a) ? -FLT_MAX : FLT_MAX;",
        "    }",
        "    return a;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_sgn(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    minus_one = _float_literal(-1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sgn({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        f"        return {zero};",
        "    }",
        f"    if (a > {zero}) {{",
        f"        return {one};",
        "    }",
        f"    if (a < {zero}) {{",
        f"        return {minus_one};",
        "    }",
        f"    return {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_sinc(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sinc({dtype_info.c_type} a) {{",
        f"    if (a == {zero}) {{",
        f"        return {one};",
        "    }",
        f"    {dtype_info.c_type} x = {pi} * a;",
        f"    return {_math_fn('sin', dtype_info)}(x) / x;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_square(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "square", "a * a")


def _float_positive(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "positive", "a")


def _float_clamp_min(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(dtype_info, "clamp_min", f"{_math_fn('fmax', dtype_info)}(a, b)")


def _float_clamp_max(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(dtype_info, "clamp_max", f"{_math_fn('fmin', dtype_info)}(a, b)")


def _float_from_ops(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    canonical_name = _normalize_op_name(name)
    if canonical_name != name:
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
            f"    return {dtype_info.prefix}{canonical_name}(a);",
            "}",
        ]
        deps = {f"{dtype_info.prefix}{canonical_name}"}
        return _GeneratedScalar(lines=lines, deps=deps, includes=set())
    name = canonical_name
    if name == "abs":
        return _float_unary_math(dtype_info, "abs", "fabs")
    if name in {"add", "sub", "mul", "div"}:
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[name]
        return _simple_binary(dtype_info, name, f"a {op} b")
    if name in {"maximum", "fmax"}:
        return _float_binary_math(dtype_info, name, "fmax")
    if name in {"minimum", "fmin"}:
        return _float_binary_math(dtype_info, name, "fmin")
    if name in {"le", "lt", "ge", "gt", "eq", "ne"}:
        op = {
            "le": "<=",
            "lt": "<",
            "ge": ">=",
            "gt": ">",
            "eq": "==",
            "ne": "!=",
        }[name]
        return _float_comparison(dtype_info, name, op)
    if name == "logical_or":
        zero = _float_literal(0.0, dtype_info)
        return _float_logical_binary(dtype_info, name, f"(a != {zero} || b != {zero})")
    if name == "logical_and":
        zero = _float_literal(0.0, dtype_info)
        return _float_logical_binary(dtype_info, name, f"(a != {zero} && b != {zero})")
    if name == "logical_xor":
        zero = _float_literal(0.0, dtype_info)
        return _float_logical_binary(dtype_info, name, f"((a != {zero}) != (b != {zero}))")
    if name == "logical_not":
        return _float_logical_not(dtype_info)
    if name == "copysign":
        return _float_binary_math(dtype_info, name, "copysign")
    if name == "hypot":
        return _float_binary_math(dtype_info, name, "hypot")
    if name == "atan2":
        return _float_binary_math(dtype_info, name, "atan2")
    if name == "pow":
        return _float_binary_math(dtype_info, name, "pow")
    if name == "fmod":
        return _float_binary_math(dtype_info, name, "fmod")
    if name == "remainder":
        return _float_remainder(dtype_info)
    if name == "floor_divide":
        return _float_floor_divide(dtype_info)
    if name == "logaddexp":
        return _float_logaddexp(dtype_info)
    if name == "logaddexp2":
        return _float_logaddexp2(dtype_info)
    if name == "nextafter":
        return _float_binary_math(dtype_info, name, "nextafter")
    if name == "xlogy":
        return _float_xlogy(dtype_info)
    if name == "heaviside":
        return _float_heaviside(dtype_info)
    if name == "ldexp":
        return _float_ldexp(dtype_info)
    if name == "clamp_min":
        return _float_clamp_min(dtype_info)
    if name == "clamp_max":
        return _float_clamp_max(dtype_info)
    if name == "neg":
        return _simple_unary(dtype_info, "neg", "-a")
    if name == "reciprocal":
        return _float_reciprocal(dtype_info)
    if name == "relu":
        return _float_relu(dtype_info)
    if name == "ceil":
        return _float_unary_math(dtype_info, "ceil", "ceil")
    if name == "floor":
        return _float_unary_math(dtype_info, "floor", "floor")
    if name in {"sin", "cos", "sqrt", "cbrt", "exp", "tanh", "log"}:
        return _float_unary_math(dtype_info, name, name)
    if name in {"acos", "acosh", "asin", "asinh", "atan", "atanh", "cosh", "sinh", "tan"}:
        return _float_unary_math(dtype_info, name, name)
    if name in {"erf", "erfc", "expm1", "log1p", "log2", "log10", "exp2", "lgamma"}:
        return _float_unary_math(dtype_info, name, name)
    if name == "isfinite":
        return _float_isfinite(dtype_info)
    if name == "rsqrt":
        return _float_rsqrt(dtype_info)
    if name == "sigmoid":
        return _float_sigmoid(dtype_info)
    if name == "log_sigmoid":
        return _float_log_sigmoid(dtype_info)
    if name == "gelu":
        return _float_gelu(dtype_info)
    if name == "elu":
        return _float_elu(dtype_info)
    if name == "leaky_relu":
        return _float_leaky_relu(dtype_info)
    if name == "softplus":
        return _float_softplus(dtype_info)
    if name == "silu":
        return _float_silu(dtype_info)
    if name == "mish":
        return _float_mish(dtype_info)
    if name == "selu":
        return _float_selu(dtype_info)
    if name == "relu6":
        return _float_relu6(dtype_info)
    if name == "hardsigmoid":
        return _float_hardsigmoid(dtype_info)
    if name == "hardswish":
        return _float_hardswish(dtype_info)
    if name == "sign":
        return _float_sign(dtype_info)
    if name == "round":
        return _float_round(dtype_info)
    if name == "trunc":
        return _float_trunc(dtype_info)
    if name == "angle":
        return _float_angle(dtype_info)
    if name == "conj":
        return _float_conj(dtype_info, "conj")
    if name == "conj_physical":
        return _float_conj(dtype_info, "conj_physical")
    if name == "deg2rad":
        return _float_deg2rad(dtype_info)
    if name == "digamma":
        return _float_digamma(dtype_info)
    if name == "erfinv":
        return _float_erfinv(dtype_info)
    if name == "frac":
        return _float_frac(dtype_info)
    if name == "i0":
        return _float_i0(dtype_info)
    if name == "logit":
        return _float_logit(dtype_info)
    if name == "isnan":
        return _float_isnan(dtype_info)
    if name == "isinf":
        return _float_isinf(dtype_info)
    if name == "isneginf":
        return _float_isneginf(dtype_info)
    if name == "isposinf":
        return _float_isposinf(dtype_info)
    if name == "nan_to_num":
        return _float_nan_to_num(dtype_info)
    if name == "positive":
        return _float_positive(dtype_info)
    if name == "rad2deg":
        return _float_rad2deg(dtype_info)
    if name == "real":
        return _simple_unary(dtype_info, "real", "a")
    if name == "sgn":
        return _float_sgn(dtype_info)
    if name == "sinc":
        return _float_sinc(dtype_info)
    if name == "square":
        return _float_square(dtype_info)
    raise CodegenBackendError(f"unsupported float scalar op: {name}")


def _int_from_f32(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines: List[str] = []
    includes: Set[str] = {"#include <math.h>", "#include <stdint.h>"}
    if dtype_info.is_signed:
        includes.add("#include <limits.h>")
        min_name = f"INT{dtype_info.bits}_MIN"
        max_name = f"INT{dtype_info.bits}_MAX"
        lines.extend(
            [
                f"static inline {dtype_info.c_type} {dtype_info.prefix}from_f32(float value) {{",
                "    if (!isfinite(value)) {",
                f"        return {min_name};",
                "    }",
                f"    if (value > (float){max_name}) {{",
                f"        return {max_name};",
                "    }",
                f"    if (value < (float){min_name}) {{",
                f"        return {min_name};",
                "    }",
                f"    return ({dtype_info.c_type})value;",
                "}",
            ]
        )
        return _GeneratedScalar(lines=lines, deps=set(), includes=includes)
    max_name = f"UINT{dtype_info.bits}_MAX"
    if dtype_info.bits in {32, 64}:
        lines.extend(
            [
                f"static inline {dtype_info.c_type} {dtype_info.prefix}from_f32(float value) {{",
                "    if (!isfinite(value)) {",
                "        return 0;",
                "    }",
                "    if (value <= 0.0f) {",
                "        return 0;",
                "    }",
                f"    if (value >= (float){max_name}) {{",
                f"        return {max_name};",
                "    }",
                f"    return ({dtype_info.c_type})value;",
                "}",
            ]
        )
        return _GeneratedScalar(lines=lines, deps=set(), includes=includes)
    lines.extend(
        [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}from_f32(float value) {{",
            "    if (!isfinite(value)) {",
            "        return 0;",
            "    }",
            f"    return ({dtype_info.c_type})value;",
            "}",
        ]
    )
    return _GeneratedScalar(lines=lines, deps=set(), includes=includes)


def _int_unary_from_f32(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
        f"    return {dtype_info.prefix}from_f32(ref_scalar_f32_{name}((float)a));",
        "}",
    ]
    deps = {f"{dtype_info.prefix}from_f32", f"ref_scalar_f32_{name}"}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_binary_from_f32(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        f"    return {dtype_info.prefix}from_f32(ref_scalar_f32_{name}((float)a, (float)b));",
        "}",
    ]
    deps = {f"{dtype_info.prefix}from_f32", f"ref_scalar_f32_{name}"}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_bool_literal(dtype_info: _ScalarTypeInfo, value: bool) -> str:
    if dtype_info.is_small_int:
        return f"({dtype_info.c_type}){1 if value else 0}"
    return "1" if value else "0"


def _int_binary_op(dtype_info: _ScalarTypeInfo, name: str, op: str) -> _GeneratedScalar:
    expr = _cast_value(f"a {op} b", dtype_info)
    return _simple_binary(dtype_info, name, expr)


def _int_unary_op(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, _cast_value(expr, dtype_info))


def _int_comparison(dtype_info: _ScalarTypeInfo, name: str, op: str) -> _GeneratedScalar:
    one = _int_bool_literal(dtype_info, True)
    zero = _int_bool_literal(dtype_info, False)
    return _simple_binary(dtype_info, name, f"a {op} b ? {one} : {zero}")


def _int_logical(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    one = _int_bool_literal(dtype_info, True)
    zero = _int_bool_literal(dtype_info, False)
    return _simple_binary(dtype_info, name, f"{expr} ? {one} : {zero}")


def _int_logical_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _int_bool_literal(dtype_info, False)
    one = _int_bool_literal(dtype_info, True)
    return _simple_unary(dtype_info, "logical_not", f"a == {zero} ? {one} : {zero}")


def _int_abs(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if not dtype_info.is_signed:
        return _simple_unary(dtype_info, "abs", "a")
    min_name = f"INT{dtype_info.bits}_MIN"
    includes = {"#include <limits.h>"}
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}abs({dtype_info.c_type} a) {{",
        f"    if (a == {min_name}) {{",
        f"        return {min_name};",
        "    }",
        "    return a < 0 ? -a : a;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=includes)


def _int_absolute(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}absolute({dtype_info.c_type} a) {{",
        f"    return {dtype_info.prefix}abs(a);",
        "}",
    ]
    deps = {f"{dtype_info.prefix}abs"}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_div(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a / b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}div({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_fmod(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a % b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}fmod({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_remainder(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a % b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}remainder({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
        f"    {dtype_info.c_type} mod = {expr};",
        "    if (mod == 0) {",
        "        return mod;",
        "    }",
        "    if ((mod < 0) != (b < 0)) {",
        "        mod += b;",
        "    }",
        "    return mod;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_floor_divide(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a / b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}floor_divide({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
    ]
    if dtype_info.is_signed:
        lines.extend(
            [
                f"    {dtype_info.c_type} quo = a / b;",
                f"    {dtype_info.c_type} rem = a % b;",
                "    if (rem != 0 && ((rem < 0) != (b < 0))) {",
                "        quo -= 1;",
                "    }",
                "    return quo;",
                "}",
            ]
        )
    else:
        lines.extend(
            [
                f"    return {expr};",
                "}",
            ]
        )
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_copysign(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if not dtype_info.is_signed:
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}copysign({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
            f"    {dtype_info.c_type} magnitude = {dtype_info.prefix}abs(a);",
            "    return b < 0 ? -magnitude : magnitude;",
            "}",
        ]
    else:
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}copysign({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
            f"    {dtype_info.c_type} magnitude = {dtype_info.prefix}abs(a);",
            "    return b < 0 ? -magnitude : magnitude;",
            "}",
        ]
    deps = {f"{dtype_info.prefix}abs"}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_neg(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if dtype_info.is_signed:
        min_name = f"INT{dtype_info.bits}_MIN"
        includes = {"#include <limits.h>"}
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}neg({dtype_info.c_type} a) {{",
            f"    if (a == {min_name}) {{",
            f"        return {min_name};",
            "    }",
            "    return -a;",
            "}",
        ]
        return _GeneratedScalar(lines=lines, deps=set(), includes=includes)
    expr = _cast_value("0 - a", dtype_info)
    return _simple_unary(dtype_info, "neg", expr)


def _int_reciprocal(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("1 / a", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}reciprocal({dtype_info.c_type} a) {{",
        "    if (a == 0) {",
        "        return 0;",
        "    }",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_relu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if dtype_info.is_signed:
        return _simple_unary(dtype_info, "relu", "a > 0 ? a : 0")
    return _simple_unary(dtype_info, "relu", "a")


def _int_ceil_floor(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, "a")


def _int_round(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "round", "a")


def _int_trunc(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "trunc", "a")


def _int_frac(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}frac({dtype_info.c_type} a) {{",
        "    (void)a;",
        "    return 0;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_sign(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if not dtype_info.is_signed:
        return _simple_unary(dtype_info, "sign", "a > 0 ? 1 : 0")
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sign({dtype_info.c_type} a) {{",
        "    if (a > 0) {",
        "        return 1;",
        "    }",
        "    if (a < 0) {",
        "        return -1;",
        "    }",
        "    return 0;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_conj(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, "a")


def _int_positive(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "positive", "a")


def _int_sgn(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if not dtype_info.is_signed:
        return _simple_unary(dtype_info, "sgn", "a > 0 ? 1 : 0")
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sgn({dtype_info.c_type} a) {{",
        "    if (a > 0) {",
        "        return 1;",
        "    }",
        "    if (a < 0) {",
        "        return -1;",
        "    }",
        "    return 0;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_square(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "square", _cast_value("a * a", dtype_info))


def _int_from_ops(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    if name == "from_f32":
        return _int_from_f32(dtype_info)
    if name in INT_UNARY_FROM_F32_OPS:
        return _int_unary_from_f32(dtype_info, name)
    if name in INT_BINARY_FROM_F32_OPS:
        return _int_binary_from_f32(dtype_info, name)
    if name == "abs":
        return _int_abs(dtype_info)
    if name == "absolute":
        return _int_absolute(dtype_info)
    if name in {"add", "sub", "mul"}:
        op = {"add": "+", "sub": "-", "mul": "*"}[name]
        return _int_binary_op(dtype_info, name, op)
    if name == "bitwise_and":
        return _int_binary_op(dtype_info, name, "&")
    if name == "bitwise_or":
        return _int_binary_op(dtype_info, name, "|")
    if name == "bitwise_xor":
        return _int_binary_op(dtype_info, name, "^")
    if name == "bitwise_left_shift":
        return _int_binary_op(dtype_info, name, "<<")
    if name == "bitwise_right_shift":
        return _int_binary_op(dtype_info, name, ">>")
    if name == "bitwise_not":
        return _int_unary_op(dtype_info, name, "~a")
    if name == "div":
        return _int_div(dtype_info)
    if name == "maximum":
        return _simple_binary(dtype_info, name, "a > b ? a : b")
    if name == "minimum":
        return _simple_binary(dtype_info, name, "a < b ? a : b")
    if name in {"le", "lt", "ge", "gt", "eq", "ne"}:
        op = {
            "le": "<=",
            "lt": "<",
            "ge": ">=",
            "gt": ">",
            "eq": "==",
            "ne": "!=",
        }[name]
        return _int_comparison(dtype_info, name, op)
    if name == "logical_or":
        return _int_logical(dtype_info, name, "(a != 0 || b != 0)")
    if name == "logical_and":
        return _int_logical(dtype_info, name, "(a != 0 && b != 0)")
    if name == "logical_xor":
        return _int_logical(dtype_info, name, "((a != 0) != (b != 0))")
    if name == "logical_not":
        return _int_logical_not(dtype_info)
    if name == "fmax":
        return _simple_binary(dtype_info, name, "a > b ? a : b")
    if name == "fmin":
        return _simple_binary(dtype_info, name, "a < b ? a : b")
    if name == "copysign":
        return _int_copysign(dtype_info)
    if name == "fmod":
        return _int_fmod(dtype_info)
    if name == "remainder":
        return _int_remainder(dtype_info)
    if name == "floor_divide":
        return _int_floor_divide(dtype_info)
    if name == "clamp_min":
        return _simple_binary(dtype_info, name, "a > b ? a : b")
    if name == "clamp_max":
        return _simple_binary(dtype_info, name, "a < b ? a : b")
    if name == "neg":
        return _int_neg(dtype_info)
    if name == "reciprocal":
        return _int_reciprocal(dtype_info)
    if name == "relu":
        return _int_relu(dtype_info)
    if name == "ceil":
        return _int_ceil_floor(dtype_info, "ceil")
    if name == "floor":
        return _int_ceil_floor(dtype_info, "floor")
    if name == "round":
        return _int_round(dtype_info)
    if name == "trunc":
        return _int_trunc(dtype_info)
    if name == "frac":
        return _int_frac(dtype_info)
    if name == "sign":
        return _int_sign(dtype_info)
    if name == "conj":
        return _int_conj(dtype_info, "conj")
    if name == "conj_physical":
        return _int_conj(dtype_info, "conj_physical")
    if name == "positive":
        return _int_positive(dtype_info)
    if name == "real":
        return _simple_unary(dtype_info, "real", "a")
    if name == "sgn":
        return _int_sgn(dtype_info)
    if name == "square":
        return _int_square(dtype_info)
    raise CodegenBackendError(f"unsupported int scalar op: {name}")


def _bool_to_f32() -> _GeneratedScalar:
    lines = [
        "static inline float ref_scalar_bool_to_f32(bool value) {",
        "    return value ? 1.0f : 0.0f;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _bool_from_f32() -> _GeneratedScalar:
    lines = [
        "static inline bool ref_scalar_bool_from_f32(float value) {",
        "    return value != 0.0f;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _bool_bitwise(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, expr)


def _bool_bitwise_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "bitwise_not", "!a")


def _bool_logical(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, expr)


def _bool_logical_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "logical_not", "!a")


def _bool_comparison(dtype_info: _ScalarTypeInfo, name: str, op: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, f"a {op} b")


def _bool_unary_from_f32(name: str) -> _GeneratedScalar:
    lines = [
        f"static inline bool ref_scalar_bool_{name}(bool a) {{",
        f"    return ref_scalar_bool_from_f32(ref_scalar_f32_{name}(ref_scalar_bool_to_f32(a)));",
        "}",
    ]
    deps = {"ref_scalar_bool_from_f32", "ref_scalar_bool_to_f32", f"ref_scalar_f32_{name}"}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _bool_binary_from_f32(name: str) -> _GeneratedScalar:
    lines = [
        f"static inline bool ref_scalar_bool_{name}(bool a, bool b) {{",
        "    return ref_scalar_bool_from_f32(",
        f"        ref_scalar_f32_{name}(ref_scalar_bool_to_f32(a), ref_scalar_bool_to_f32(b))",
        "    );",
        "}",
    ]
    deps = {"ref_scalar_bool_from_f32", "ref_scalar_bool_to_f32", f"ref_scalar_f32_{name}"}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _bool_from_ops(name: str) -> _GeneratedScalar:
    if name == "to_f32":
        return _bool_to_f32()
    if name == "from_f32":
        return _bool_from_f32()
    if name == "bitwise_and":
        return _simple_binary(_SCALAR_TYPES[torch.bool], name, "a & b")
    if name == "bitwise_or":
        return _simple_binary(_SCALAR_TYPES[torch.bool], name, "a | b")
    if name == "bitwise_xor":
        return _simple_binary(_SCALAR_TYPES[torch.bool], name, "a ^ b")
    if name == "bitwise_not":
        return _bool_bitwise_not(_SCALAR_TYPES[torch.bool])
    if name == "logical_or":
        return _bool_logical(_SCALAR_TYPES[torch.bool], name, "a || b")
    if name == "logical_and":
        return _bool_logical(_SCALAR_TYPES[torch.bool], name, "a && b")
    if name == "logical_xor":
        return _bool_logical(_SCALAR_TYPES[torch.bool], name, "a != b")
    if name == "logical_not":
        return _bool_logical_not(_SCALAR_TYPES[torch.bool])
    if name in {"le", "lt", "ge", "gt", "eq", "ne"}:
        op = {
            "le": "<=",
            "lt": "<",
            "ge": ">=",
            "gt": ">",
            "eq": "==",
            "ne": "!=",
        }[name]
        return _bool_comparison(_SCALAR_TYPES[torch.bool], name, op)
    if name in BOOL_UNARY_FROM_F32_OPS:
        return _bool_unary_from_f32(name)
    if name in BOOL_BINARY_FROM_F32_OPS:
        return _bool_binary_from_f32(name)
    raise CodegenBackendError(f"unsupported bool scalar op: {name}")


_SCALAR_TYPES: Dict[torch.dtype, _ScalarTypeInfo] = {
    torch.float32: _ScalarTypeInfo(
        torch_dtype=torch.float32,
        c_type=_CODEGEN_DTYPES[torch.float32].c_type,
        prefix=_CODEGEN_DTYPES[torch.float32].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.float32].suffix,
        is_float=True,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=None,
    ),
    torch.float64: _ScalarTypeInfo(
        torch_dtype=torch.float64,
        c_type=_CODEGEN_DTYPES[torch.float64].c_type,
        prefix=_CODEGEN_DTYPES[torch.float64].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.float64].suffix,
        is_float=True,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=None,
    ),
    torch.int8: _ScalarTypeInfo(
        torch_dtype=torch.int8,
        c_type=_CODEGEN_DTYPES[torch.int8].c_type,
        prefix=_CODEGEN_DTYPES[torch.int8].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.int8].suffix,
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=True,
        bits=8,
    ),
    torch.int16: _ScalarTypeInfo(
        torch_dtype=torch.int16,
        c_type=_CODEGEN_DTYPES[torch.int16].c_type,
        prefix=_CODEGEN_DTYPES[torch.int16].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.int16].suffix,
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=True,
        bits=16,
    ),
    torch.int32: _ScalarTypeInfo(
        torch_dtype=torch.int32,
        c_type=_CODEGEN_DTYPES[torch.int32].c_type,
        prefix=_CODEGEN_DTYPES[torch.int32].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.int32].suffix,
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=32,
    ),
    torch.int64: _ScalarTypeInfo(
        torch_dtype=torch.int64,
        c_type=_CODEGEN_DTYPES[torch.int64].c_type,
        prefix=_CODEGEN_DTYPES[torch.int64].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.int64].suffix,
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=64,
    ),
    torch.uint8: _ScalarTypeInfo(
        torch_dtype=torch.uint8,
        c_type=_CODEGEN_DTYPES[torch.uint8].c_type,
        prefix=_CODEGEN_DTYPES[torch.uint8].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.uint8].suffix,
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=True,
        bits=8,
    ),
    torch.uint16: _ScalarTypeInfo(
        torch_dtype=torch.uint16,
        c_type=_CODEGEN_DTYPES[torch.uint16].c_type,
        prefix=_CODEGEN_DTYPES[torch.uint16].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.uint16].suffix,
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=True,
        bits=16,
    ),
    torch.uint32: _ScalarTypeInfo(
        torch_dtype=torch.uint32,
        c_type=_CODEGEN_DTYPES[torch.uint32].c_type,
        prefix=_CODEGEN_DTYPES[torch.uint32].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.uint32].suffix,
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=False,
        bits=32,
    ),
    torch.uint64: _ScalarTypeInfo(
        torch_dtype=torch.uint64,
        c_type=_CODEGEN_DTYPES[torch.uint64].c_type,
        prefix=_CODEGEN_DTYPES[torch.uint64].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.uint64].suffix,
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=False,
        bits=64,
    ),
    torch.bool: _ScalarTypeInfo(
        torch_dtype=torch.bool,
        c_type=_CODEGEN_DTYPES[torch.bool].c_type,
        prefix=_CODEGEN_DTYPES[torch.bool].scalar_prefix,
        suffix=_CODEGEN_DTYPES[torch.bool].suffix,
        is_float=False,
        is_bool=True,
        is_signed=False,
        is_small_int=False,
        bits=None,
    ),
}


_SCALAR_TYPE_BY_ENUM: Mapping[ScalarType, _ScalarTypeInfo] = {
    ScalarType.F32: _SCALAR_TYPES[torch.float32],
    ScalarType.F64: _SCALAR_TYPES[torch.float64],
    ScalarType.I8: _SCALAR_TYPES[torch.int8],
    ScalarType.I16: _SCALAR_TYPES[torch.int16],
    ScalarType.I32: _SCALAR_TYPES[torch.int32],
    ScalarType.I64: _SCALAR_TYPES[torch.int64],
    ScalarType.U8: _SCALAR_TYPES[torch.uint8],
    ScalarType.U16: _SCALAR_TYPES[torch.uint16],
    ScalarType.U32: _SCALAR_TYPES[torch.uint32],
    ScalarType.U64: _SCALAR_TYPES[torch.uint64],
    ScalarType.BOOL: _SCALAR_TYPES[torch.bool],
}


_CONVERSION_SOURCE_BY_FUNCTION: Mapping[ScalarFunction, ScalarType] = {
    ScalarFunction.CONVERT_FROM_F32: ScalarType.F32,
    ScalarFunction.CONVERT_FROM_F64: ScalarType.F64,
    ScalarFunction.CONVERT_FROM_I8: ScalarType.I8,
    ScalarFunction.CONVERT_FROM_I16: ScalarType.I16,
    ScalarFunction.CONVERT_FROM_I32: ScalarType.I32,
    ScalarFunction.CONVERT_FROM_I64: ScalarType.I64,
    ScalarFunction.CONVERT_FROM_U8: ScalarType.U8,
    ScalarFunction.CONVERT_FROM_U16: ScalarType.U16,
    ScalarFunction.CONVERT_FROM_U32: ScalarType.U32,
    ScalarFunction.CONVERT_FROM_U64: ScalarType.U64,
    ScalarFunction.CONVERT_FROM_BOOL: ScalarType.BOOL,
}


def _supported_ops(dtype_info: _ScalarTypeInfo) -> Set[str]:
    if dtype_info.is_float:
        return {op.value for op in FLOAT_ONLY_OPS}
    if dtype_info.is_bool:
        return set(BOOL_BASE_OPS | BOOL_UNARY_FROM_F32_OPS | BOOL_BINARY_FROM_F32_OPS)
    if dtype_info.is_signed:
        return set(SIGNED_INT_BASE_OPS | INT_UNARY_FROM_F32_OPS | INT_BINARY_FROM_F32_OPS)
    return set(UNSIGNED_INT_BASE_OPS | INT_UNARY_FROM_F32_OPS | INT_BINARY_FROM_F32_OPS)


def validate_scalar_function_supported_ops() -> None:
    scalar_ops = {
        function.value
        for function in ScalarFunction
        if not function.value.startswith("convert_from_")
    }
    conversion_aliases = {"from_f32", "to_f32"}
    categories = {
        "float": _supported_ops(_SCALAR_TYPES[torch.float32]),
        "bool": _supported_ops(_SCALAR_TYPES[torch.bool]),
        "signed_int": _supported_ops(_SCALAR_TYPES[torch.int8]),
        "unsigned_int": _supported_ops(_SCALAR_TYPES[torch.uint8]),
    }
    errors: List[str] = []
    for category, supported in categories.items():
        missing = sorted(supported - scalar_ops - conversion_aliases)
        if missing:
            errors.append(
                f"{category} missing ScalarFunction ops (defined in _supported_ops): {missing}"
            )
    supported_union = set().union(*categories.values()) - conversion_aliases
    unexpected_extras = sorted(scalar_ops - supported_union)
    if unexpected_extras:
        errors.append(
            "ScalarFunction ops not supported by any dtype category: "
            f"{unexpected_extras}"
        )
    if errors:
        raise AssertionError(
            "ScalarFunction/_supported_ops drift detected:\n" + "\n".join(errors)
        )


def _parse_scalar_name(function_name: str) -> tuple[_ScalarTypeInfo, str]:
    for info in _SCALAR_TYPES.values():
        if function_name.startswith(info.prefix):
            return info, function_name[len(info.prefix) :]
    raise CodegenBackendError(f"unknown scalar function requested: {function_name}")


def _generate_scalar(function_name: str) -> _GeneratedScalar:
    dtype_info, op_name = _parse_scalar_name(function_name)
    if _normalize_op_name(op_name) not in _supported_ops(dtype_info):
        raise CodegenBackendError(f"unsupported scalar op {op_name} for {dtype_info.suffix}")
    if dtype_info.is_float:
        generated = _float_from_ops(dtype_info, op_name)
    elif dtype_info.is_bool:
        generated = _bool_from_ops(op_name)
    else:
        generated = _int_from_ops(dtype_info, op_name)
    includes = set(generated.includes)
    if dtype_info.is_float:
        includes.update({"#include <math.h>", "#include <float.h>"})
    if not dtype_info.is_float and not dtype_info.is_bool:
        includes.update({"#include <stdint.h>"})
        if dtype_info.is_signed:
            includes.add("#include <limits.h>")
    if dtype_info.is_bool:
        includes.add("#include <stdbool.h>")
    return _GeneratedScalar(lines=generated.lines, deps=generated.deps, includes=includes)


def _function_name_for_key(key: ScalarFunctionKey) -> str:
    if key.function in _CONVERSION_SOURCE_BY_FUNCTION:
        source_type = _CONVERSION_SOURCE_BY_FUNCTION[key.function]
        if source_type == ScalarType.F32:
            if key.return_type in {
                ScalarType.I8,
                ScalarType.I16,
                ScalarType.I32,
                ScalarType.I64,
                ScalarType.U8,
                ScalarType.U16,
                ScalarType.U32,
                ScalarType.U64,
                ScalarType.BOOL,
            }:
                target_info = _SCALAR_TYPE_BY_ENUM[key.return_type]
                return f"{target_info.prefix}from_f32"
            raise CodegenBackendError(
                f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
            )
        if source_type == ScalarType.BOOL:
            if key.return_type == ScalarType.F32:
                source_info = _SCALAR_TYPE_BY_ENUM[source_type]
                return f"{source_info.prefix}to_f32"
            raise CodegenBackendError(
                f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
            )
        raise CodegenBackendError(
            f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
        )
    op_name = key.function.value
    dtype_info = _SCALAR_TYPE_BY_ENUM[key.return_type]
    if op_name not in _supported_ops(dtype_info):
        raise CodegenBackendError(
            f"unsupported scalar op {op_name} for {dtype_info.suffix}"
        )
    return f"{dtype_info.prefix}{op_name}"


class ScalarFunctionRegistry:
    def __init__(self) -> None:
        self._requested: List[str] = []
        self._requested_set: Set[str] = set()
        self._key_to_name: Dict[ScalarFunctionKey, str] = {}
        self._generated: Dict[str, _GeneratedScalar] = {}

    def request(self, key: ScalarFunctionKey) -> str:
        name = self._key_to_name.get(key)
        if name is None:
            name = _function_name_for_key(key)
            self._key_to_name[key] = name
        self._register_name(name)
        return name

    def _register_name(self, function_name: str) -> None:
        if function_name in self._requested_set:
            return
        _parse_scalar_name(function_name)
        self._requested.append(function_name)
        self._requested_set.add(function_name)

    def include_lines(self) -> List[str]:
        includes: Set[str] = set()
        visited: Set[str] = set()

        def collect(name: str) -> None:
            if name in visited:
                return
            self._ensure_generated(name)
            entry = self._generated[name]
            visited.add(name)
            for dep in entry.deps:
                collect(dep)
            includes.update(entry.includes)

        for name in self._requested:
            collect(name)
        ordered = sorted(includes)
        preamble = [
            "#ifndef REF_PI_F",
            "#define REF_PI_F 3.14159265358979323846f",
            "#endif",
            "#ifndef REF_PI_D",
            "#define REF_PI_D 3.14159265358979323846",
            "#endif",
        ]
        return ordered + preamble

    def render(self) -> List[str]:
        if not self._requested:
            return []
        lines: List[str] = []
        emitted: Set[str] = set()

        def emit(name: str) -> None:
            if name in emitted:
                return
            self._ensure_generated(name)
            entry = self._generated[name]
            for dep in sorted(entry.deps):
                emit(dep)
            lines.extend(entry.lines)
            lines.append("")
            emitted.add(name)

        for name in self._requested:
            emit(name)
        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _ensure_generated(self, name: str) -> None:
        if name in self._generated:
            return
        self._generated[name] = _generate_scalar(name)
