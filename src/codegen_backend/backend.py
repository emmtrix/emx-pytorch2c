import hashlib
import operator
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from importlib import resources
from jinja2 import Environment, FileSystemLoader
import torch.fx
from torch.fx.immutable_collections import immutable_list

from c_ref_backend.cffi_bindings import RefBackendError


@dataclass(frozen=True)
class _OpSpec:
    name: str
    kind: str
    symbol: str | None
    supported_targets: set
    inplace_targets: set = field(default_factory=set)
    inplace_arg_index: int | None = None


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

def _binary_spec(
    name: str,
    targets: Iterable[object],
    symbol: str | None,
    inplace_targets: Iterable[object] = (),
) -> _OpSpec:
    inplace_targets_set = set(inplace_targets)
    return _OpSpec(
        name=name,
        kind="binary",
        symbol=symbol,
        supported_targets=set(targets),
        inplace_targets=inplace_targets_set,
        inplace_arg_index=0 if inplace_targets_set else None,
    )


def _unary_spec(
    name: str, targets: Iterable[object], inplace_targets: Iterable[object] = ()
) -> _OpSpec:
    inplace_targets_set = set(inplace_targets)
    return _OpSpec(
        name=name,
        kind="unary",
        symbol=None,
        supported_targets=set(targets),
        inplace_targets=inplace_targets_set,
        inplace_arg_index=0 if inplace_targets_set else None,
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
            torch.ops.aten.add_,
        ),
        "+",
        inplace_targets=(
            torch.ops.aten.add_.Tensor,
            torch.ops.aten.add_,
        ),
    ),
    "sub": _binary_spec(
        "sub",
        (
            operator.sub,
            torch.sub,
            torch.ops.prims.sub,
            torch.ops.prims.sub.default,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.sub_.Tensor,
            torch.ops.aten.sub_,
        ),
        "-",
        inplace_targets=(
            torch.ops.aten.sub_.Tensor,
            torch.ops.aten.sub_,
        ),
    ),
    "mul": _binary_spec(
        "mul",
        (
            operator.mul,
            torch.mul,
            torch.ops.prims.mul,
            torch.ops.prims.mul.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul_.Tensor,
            torch.ops.aten.mul_,
        ),
        "*",
        inplace_targets=(
            torch.ops.aten.mul_.Tensor,
            torch.ops.aten.mul_,
        ),
    ),
    "bitwise_and": _binary_spec(
        "bitwise_and",
        (
            operator.and_,
            torch.bitwise_and,
            torch.ops.aten.bitwise_and.Tensor,
            torch.ops.aten.bitwise_and_.Tensor,
            torch.ops.aten.bitwise_and_,
            torch.ops.aten.bitwise_and,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.bitwise_and_.Tensor,
            torch.ops.aten.bitwise_and_,
        ),
    ),
    "bitwise_or": _binary_spec(
        "bitwise_or",
        (
            operator.or_,
            torch.bitwise_or,
            torch.ops.aten.bitwise_or.Tensor,
            torch.ops.aten.bitwise_or_.Tensor,
            torch.ops.aten.bitwise_or_,
            torch.ops.aten.bitwise_or,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.bitwise_or_.Tensor,
            torch.ops.aten.bitwise_or_,
        ),
    ),
    "bitwise_xor": _binary_spec(
        "bitwise_xor",
        (
            operator.xor,
            torch.bitwise_xor,
            torch.ops.aten.bitwise_xor.Tensor,
            torch.ops.aten.bitwise_xor_.Tensor,
            torch.ops.aten.bitwise_xor_,
            torch.ops.aten.bitwise_xor,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.bitwise_xor_.Tensor,
            torch.ops.aten.bitwise_xor_,
        ),
    ),
    "logical_or": _binary_spec(
        "logical_or",
        (
            torch.logical_or,
            torch.ops.aten.logical_or.default,
            torch.ops.aten.logical_or,
            torch.ops.aten.logical_or_.default,
            torch.ops.aten.logical_or_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.logical_or_.default,
            torch.ops.aten.logical_or_,
        ),
    ),
    "logical_and": _binary_spec(
        "logical_and",
        (
            torch.logical_and,
            torch.ops.aten.logical_and.default,
            torch.ops.aten.logical_and,
            torch.ops.aten.logical_and_.default,
            torch.ops.aten.logical_and_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.logical_and_.default,
            torch.ops.aten.logical_and_,
        ),
    ),
    "logical_xor": _binary_spec(
        "logical_xor",
        (
            torch.logical_xor,
            torch.ops.aten.logical_xor.default,
            torch.ops.aten.logical_xor,
            torch.ops.aten.logical_xor_.default,
            torch.ops.aten.logical_xor_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.logical_xor_.default,
            torch.ops.aten.logical_xor_,
        ),
    ),
    "logical_not": _unary_spec(
        "logical_not",
        (
            torch.logical_not,
            torch.ops.aten.logical_not.default,
            torch.ops.aten.logical_not,
            torch.ops.aten.logical_not_.default,
            torch.ops.aten.logical_not_,
        ),
        inplace_targets=(
            torch.ops.aten.logical_not_.default,
            torch.ops.aten.logical_not_,
        ),
    ),
    "bitwise_left_shift": _binary_spec(
        "bitwise_left_shift",
        (
            operator.lshift,
            torch.bitwise_left_shift,
            torch.ops.aten.bitwise_left_shift.Tensor,
            torch.ops.aten.bitwise_left_shift_.Tensor,
            torch.ops.aten.bitwise_left_shift_,
            torch.ops.aten.bitwise_left_shift,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.bitwise_left_shift_.Tensor,
            torch.ops.aten.bitwise_left_shift_,
        ),
    ),
    "bitwise_right_shift": _binary_spec(
        "bitwise_right_shift",
        (
            operator.rshift,
            torch.bitwise_right_shift,
            torch.ops.aten.bitwise_right_shift.Tensor,
            torch.ops.aten.bitwise_right_shift_.Tensor,
            torch.ops.aten.bitwise_right_shift_,
            torch.ops.aten.bitwise_right_shift,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.bitwise_right_shift_.Tensor,
            torch.ops.aten.bitwise_right_shift_,
        ),
    ),
    "bitwise_not": _unary_spec(
        "bitwise_not",
        (
            operator.invert,
            torch.bitwise_not,
            torch.ops.aten.bitwise_not.default,
            torch.ops.aten.bitwise_not,
            torch.ops.aten.bitwise_not_.default,
            torch.ops.aten.bitwise_not_,
        ),
        inplace_targets=(
            torch.ops.aten.bitwise_not_.default,
            torch.ops.aten.bitwise_not_,
        ),
    ),
    "div": _binary_spec(
        "div",
        (
            operator.truediv,
            torch.div,
            torch.true_divide,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.div,
            torch.ops.aten.div_.Tensor,
            torch.ops.aten.div_,
        ),
        "/",
        inplace_targets=(
            torch.ops.aten.div_.Tensor,
            torch.ops.aten.div_,
        ),
    ),
    "lt": _binary_spec(
        "lt",
        (
            operator.lt,
            torch.lt,
            torch.ops.aten.lt.Tensor,
            torch.ops.aten.lt.default,
            torch.ops.aten.lt,
        ),
        None,
    ),
    "le": _binary_spec(
        "le",
        (
            operator.le,
            torch.le,
            torch.ops.aten.le.Tensor,
            torch.ops.aten.le.default,
            torch.ops.aten.le,
        ),
        None,
    ),
    "gt": _binary_spec(
        "gt",
        (
            operator.gt,
            torch.gt,
            torch.ops.aten.gt.Tensor,
            torch.ops.aten.gt.default,
            torch.ops.aten.gt,
        ),
        None,
    ),
    "ge": _binary_spec(
        "ge",
        (
            operator.ge,
            torch.ge,
            torch.ops.aten.ge.Tensor,
            torch.ops.aten.ge.default,
            torch.ops.aten.ge,
        ),
        None,
    ),
    "eq": _binary_spec(
        "eq",
        (
            operator.eq,
            torch.eq,
            torch.ops.aten.eq.Tensor,
            torch.ops.aten.eq.default,
            torch.ops.aten.eq,
        ),
        None,
    ),
    "ne": _binary_spec(
        "ne",
        (
            operator.ne,
            torch.ne,
            torch.ops.aten.ne.Tensor,
            torch.ops.aten.ne.default,
            torch.ops.aten.ne,
        ),
        None,
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
            torch.ops.aten.atan2_.default,
            torch.ops.aten.atan2_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.atan2_.default,
            torch.ops.aten.atan2_,
        ),
    ),
    "pow": _binary_spec(
        "pow",
        (
            operator.pow,
            torch.pow,
            torch.ops.aten.pow.Tensor_Tensor,
            torch.ops.aten.pow_.Tensor,
            torch.ops.aten.pow_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.pow_.Tensor,
            torch.ops.aten.pow_,
        ),
    ),
    "remainder": _binary_spec(
        "remainder",
        (
            torch.remainder,
            torch.ops.aten.remainder.Tensor,
            torch.ops.aten.remainder,
            torch.ops.aten.remainder_.Tensor,
            torch.ops.aten.remainder_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.remainder_.Tensor,
            torch.ops.aten.remainder_,
        ),
    ),
    "fmod": _binary_spec(
        "fmod",
        (
            torch.fmod,
            torch.ops.aten.fmod.Tensor,
            torch.ops.aten.fmod,
            torch.ops.aten.fmod_.Tensor,
            torch.ops.aten.fmod_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.fmod_.Tensor,
            torch.ops.aten.fmod_,
        ),
    ),
    "floor_divide": _binary_spec(
        "floor_divide",
        (
            torch.floor_divide,
            torch.ops.aten.floor_divide.default,
            torch.ops.aten.floor_divide,
            torch.ops.aten.floor_divide_.Tensor,
            torch.ops.aten.floor_divide_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.floor_divide_.Tensor,
            torch.ops.aten.floor_divide_,
        ),
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
            torch.ops.aten.copysign_.Tensor,
            torch.ops.aten.copysign_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.copysign_.Tensor,
            torch.ops.aten.copysign_,
        ),
    ),
    "hypot": _binary_spec(
        "hypot",
        (
            torch.hypot,
            torch.ops.aten.hypot.default,
            torch.ops.aten.hypot,
            torch.ops.aten.hypot_.default,
            torch.ops.aten.hypot_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.hypot_.default,
            torch.ops.aten.hypot_,
        ),
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
            torch.ops.aten.nextafter_.default,
            torch.ops.aten.nextafter_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.nextafter_.default,
            torch.ops.aten.nextafter_,
        ),
    ),
    "xlogy": _binary_spec(
        "xlogy",
        (
            torch.xlogy,
            torch.ops.aten.xlogy.Tensor,
            torch.ops.aten.xlogy,
            torch.ops.aten.xlogy_.Tensor,
            torch.ops.aten.xlogy_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.xlogy_.Tensor,
            torch.ops.aten.xlogy_,
        ),
    ),
    "heaviside": _binary_spec(
        "heaviside",
        (
            torch.heaviside,
            torch.ops.aten.heaviside.default,
            torch.ops.aten.heaviside,
            torch.ops.aten.heaviside_.default,
            torch.ops.aten.heaviside_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.heaviside_.default,
            torch.ops.aten.heaviside_,
        ),
    ),
    "where": _OpSpec(
        name="where",
        kind="where",
        symbol=None,
        supported_targets={
            torch.where,
            torch.ops.aten.where.self,
        },
    ),
    "ldexp": _binary_spec(
        "ldexp",
        (
            torch.ldexp,
            torch.ops.aten.ldexp.default,
            torch.ops.aten.ldexp,
            torch.ops.aten.ldexp_.default,
            torch.ops.aten.ldexp_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.ldexp_.default,
            torch.ops.aten.ldexp_,
        ),
    ),
    "clamp_min": _binary_spec(
        "clamp_min",
        (
            torch.clamp_min,
            torch.ops.aten.clamp_min.default,
            torch.ops.aten.clamp_min,
            torch.ops.aten.clamp_min_.default,
            torch.ops.aten.clamp_min_.Tensor,
            torch.ops.aten.clamp_min_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.clamp_min_.default,
            torch.ops.aten.clamp_min_.Tensor,
            torch.ops.aten.clamp_min_,
        ),
    ),
    "clamp_max": _binary_spec(
        "clamp_max",
        (
            torch.clamp_max,
            torch.ops.aten.clamp_max.default,
            torch.ops.aten.clamp_max,
            torch.ops.aten.clamp_max_.default,
            torch.ops.aten.clamp_max_.Tensor,
            torch.ops.aten.clamp_max_,
        ),
        None,
        inplace_targets=(
            torch.ops.aten.clamp_max_.default,
            torch.ops.aten.clamp_max_.Tensor,
            torch.ops.aten.clamp_max_,
        ),
    ),
    "neg": _unary_spec(
        "neg",
        (
            operator.neg,
            torch.neg,
            torch.ops.aten.neg.default,
            torch.ops.aten.neg,
            torch.ops.aten.neg_.default,
            torch.ops.aten.neg_,
        ),
        inplace_targets=(
            torch.ops.aten.neg_.default,
            torch.ops.aten.neg_,
        ),
    ),
    "exp": _unary_spec(
        "exp",
        (
            torch.exp,
            torch.ops.aten.exp.default,
            torch.ops.aten.exp,
            torch.ops.aten.exp_.default,
            torch.ops.aten.exp_,
        ),
        inplace_targets=(
            torch.ops.aten.exp_.default,
            torch.ops.aten.exp_,
        ),
    ),
    "abs": _unary_spec(
        "abs",
        (
            torch.abs,
            torch.ops.aten.abs.default,
            torch.ops.aten.abs,
            torch.ops.aten.abs_.default,
            torch.ops.aten.abs_,
        ),
        inplace_targets=(
            torch.ops.aten.abs_.default,
            torch.ops.aten.abs_,
        ),
    ),
    "sqrt": _unary_spec(
        "sqrt",
        (
            torch.sqrt,
            torch.ops.aten.sqrt.default,
            torch.ops.aten.sqrt,
            torch.ops.aten.sqrt_.default,
            torch.ops.aten.sqrt_,
        ),
        inplace_targets=(
            torch.ops.aten.sqrt_.default,
            torch.ops.aten.sqrt_,
        ),
    ),
    "log": _unary_spec(
        "log",
        (
            torch.log,
            torch.ops.aten.log.default,
            torch.ops.aten.log,
            torch.ops.aten.log_.default,
            torch.ops.aten.log_,
        ),
        inplace_targets=(
            torch.ops.aten.log_.default,
            torch.ops.aten.log_,
        ),
    ),
    "sin": _unary_spec(
        "sin",
        (
            torch.sin,
            torch.ops.aten.sin.default,
            torch.ops.aten.sin,
            torch.ops.aten.sin_.default,
            torch.ops.aten.sin_,
        ),
        inplace_targets=(
            torch.ops.aten.sin_.default,
            torch.ops.aten.sin_,
        ),
    ),
    "cos": _unary_spec(
        "cos",
        (
            torch.cos,
            torch.ops.aten.cos.default,
            torch.ops.aten.cos,
            torch.ops.aten.cos_.default,
            torch.ops.aten.cos_,
        ),
        inplace_targets=(
            torch.ops.aten.cos_.default,
            torch.ops.aten.cos_,
        ),
    ),
    "acos": _unary_spec(
        "acos",
        (
            torch.acos,
            torch.ops.aten.acos.default,
            torch.ops.aten.acos,
            torch.ops.aten.acos_.default,
            torch.ops.aten.acos_,
        ),
        inplace_targets=(
            torch.ops.aten.acos_.default,
            torch.ops.aten.acos_,
        ),
    ),
    "acosh": _unary_spec(
        "acosh",
        (
            torch.acosh,
            torch.arccosh,
            torch.ops.aten.acosh.default,
            torch.ops.aten.acosh,
            torch.ops.aten.acosh_.default,
            torch.ops.aten.acosh_,
            torch.ops.aten.arccosh.default,
            torch.ops.aten.arccosh,
            torch.ops.aten.arccosh_.default,
            torch.ops.aten.arccosh_,
        ),
        inplace_targets=(
            torch.ops.aten.acosh_.default,
            torch.ops.aten.acosh_,
            torch.ops.aten.arccosh_.default,
            torch.ops.aten.arccosh_,
        ),
    ),
    "asin": _unary_spec(
        "asin",
        (
            torch.asin,
            torch.ops.aten.asin.default,
            torch.ops.aten.asin,
            torch.ops.aten.asin_.default,
            torch.ops.aten.asin_,
        ),
        inplace_targets=(
            torch.ops.aten.asin_.default,
            torch.ops.aten.asin_,
        ),
    ),
    "asinh": _unary_spec(
        "asinh",
        (
            torch.asinh,
            torch.ops.aten.asinh.default,
            torch.ops.aten.asinh,
            torch.ops.aten.asinh_.default,
            torch.ops.aten.asinh_,
        ),
        inplace_targets=(
            torch.ops.aten.asinh_.default,
            torch.ops.aten.asinh_,
        ),
    ),
    "atan": _unary_spec(
        "atan",
        (
            torch.atan,
            torch.ops.aten.atan.default,
            torch.ops.aten.atan,
            torch.ops.aten.atan_.default,
            torch.ops.aten.atan_,
        ),
        inplace_targets=(
            torch.ops.aten.atan_.default,
            torch.ops.aten.atan_,
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
        inplace_targets=(
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
            torch.ops.aten.cosh_.default,
            torch.ops.aten.cosh_,
        ),
        inplace_targets=(
            torch.ops.aten.cosh_.default,
            torch.ops.aten.cosh_,
        ),
    ),
    "sinh": _unary_spec(
        "sinh",
        (
            torch.sinh,
            torch.ops.aten.sinh.default,
            torch.ops.aten.sinh,
            torch.ops.aten.sinh_.default,
            torch.ops.aten.sinh_,
        ),
        inplace_targets=(
            torch.ops.aten.sinh_.default,
            torch.ops.aten.sinh_,
        ),
    ),
    "tan": _unary_spec(
        "tan",
        (
            torch.tan,
            torch.ops.aten.tan.default,
            torch.ops.aten.tan,
            torch.ops.aten.tan_.default,
            torch.ops.aten.tan_,
        ),
        inplace_targets=(
            torch.ops.aten.tan_.default,
            torch.ops.aten.tan_,
        ),
    ),
    "erf": _unary_spec(
        "erf",
        (
            torch.erf,
            torch.ops.aten.erf.default,
            torch.ops.aten.erf,
            torch.ops.aten.erf_.default,
            torch.ops.aten.erf_,
        ),
        inplace_targets=(
            torch.ops.aten.erf_.default,
            torch.ops.aten.erf_,
        ),
    ),
    "erfc": _unary_spec(
        "erfc",
        (
            torch.erfc,
            torch.ops.aten.erfc.default,
            torch.ops.aten.erfc,
            torch.ops.aten.erfc_.default,
            torch.ops.aten.erfc_,
        ),
        inplace_targets=(
            torch.ops.aten.erfc_.default,
            torch.ops.aten.erfc_,
        ),
    ),
    "expm1": _unary_spec(
        "expm1",
        (
            torch.expm1,
            torch.ops.aten.expm1.default,
            torch.ops.aten.expm1,
            torch.ops.aten.expm1_.default,
            torch.ops.aten.expm1_,
        ),
        inplace_targets=(
            torch.ops.aten.expm1_.default,
            torch.ops.aten.expm1_,
        ),
    ),
    "log1p": _unary_spec(
        "log1p",
        (
            torch.log1p,
            torch.ops.aten.log1p.default,
            torch.ops.aten.log1p,
            torch.ops.aten.log1p_.default,
            torch.ops.aten.log1p_,
        ),
        inplace_targets=(
            torch.ops.aten.log1p_.default,
            torch.ops.aten.log1p_,
        ),
    ),
    "log2": _unary_spec(
        "log2",
        (
            torch.log2,
            torch.ops.aten.log2.default,
            torch.ops.aten.log2,
            torch.ops.aten.log2_.default,
            torch.ops.aten.log2_,
        ),
        inplace_targets=(
            torch.ops.aten.log2_.default,
            torch.ops.aten.log2_,
        ),
    ),
    "log10": _unary_spec(
        "log10",
        (
            torch.log10,
            torch.ops.aten.log10.default,
            torch.ops.aten.log10,
            torch.ops.aten.log10_.default,
            torch.ops.aten.log10_,
        ),
        inplace_targets=(
            torch.ops.aten.log10_.default,
            torch.ops.aten.log10_,
        ),
    ),
    "rsqrt": _unary_spec(
        "rsqrt",
        (
            torch.rsqrt,
            torch.ops.aten.rsqrt.default,
            torch.ops.aten.rsqrt,
            torch.ops.aten.rsqrt_.default,
            torch.ops.aten.rsqrt_,
        ),
        inplace_targets=(
            torch.ops.aten.rsqrt_.default,
            torch.ops.aten.rsqrt_,
        ),
    ),
    "sigmoid": _unary_spec(
        "sigmoid",
        (
            torch.sigmoid,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.sigmoid,
            torch.ops.aten.sigmoid_.default,
            torch.ops.aten.sigmoid_,
        ),
        inplace_targets=(
            torch.ops.aten.sigmoid_.default,
            torch.ops.aten.sigmoid_,
        ),
    ),
    "mish": _unary_spec(
        "mish",
        (
            F.mish,
            torch.ops.aten.mish.default,
            torch.ops.aten.mish,
        ),
    ),
    "silu": _unary_spec(
        "silu",
        (
            F.silu,
            torch.ops.aten.silu.default,
            torch.ops.aten.silu,
            torch.ops.aten.silu_.default,
            torch.ops.aten.silu_,
        ),
        inplace_targets=(
            torch.ops.aten.silu_.default,
            torch.ops.aten.silu_,
        ),
    ),
    "hardswish": _unary_spec(
        "hardswish",
        (
            F.hardswish,
            torch.ops.aten.hardswish.default,
            torch.ops.aten.hardswish,
            torch.ops.aten.hardswish_.default,
            torch.ops.aten.hardswish_,
        ),
        inplace_targets=(
            torch.ops.aten.hardswish_.default,
            torch.ops.aten.hardswish_,
        ),
    ),
    "sign": _unary_spec(
        "sign",
        (
            torch.sign,
            torch.ops.aten.sign.default,
            torch.ops.aten.sign,
            torch.ops.aten.sign_.default,
            torch.ops.aten.sign_,
        ),
        inplace_targets=(
            torch.ops.aten.sign_.default,
            torch.ops.aten.sign_,
        ),
    ),
    "round": _unary_spec(
        "round",
        (
            torch.round,
            torch.ops.aten.round.default,
            torch.ops.aten.round,
            torch.ops.aten.round_.default,
            torch.ops.aten.round_,
        ),
        inplace_targets=(
            torch.ops.aten.round_.default,
            torch.ops.aten.round_,
        ),
    ),
    "trunc": _unary_spec(
        "trunc",
        (
            torch.trunc,
            torch.ops.aten.trunc.default,
            torch.ops.aten.trunc,
            torch.ops.aten.trunc_.default,
            torch.ops.aten.trunc_,
        ),
        inplace_targets=(
            torch.ops.aten.trunc_.default,
            torch.ops.aten.trunc_,
        ),
    ),
    "tanh": _unary_spec(
        "tanh",
        (
            torch.tanh,
            torch.ops.aten.tanh.default,
            torch.ops.aten.tanh,
            torch.ops.aten.tanh_.default,
            torch.ops.aten.tanh_,
        ),
        inplace_targets=(
            torch.ops.aten.tanh_.default,
            torch.ops.aten.tanh_,
        ),
    ),
    "floor": _unary_spec(
        "floor",
        (
            torch.floor,
            torch.ops.aten.floor.default,
            torch.ops.aten.floor,
            torch.ops.aten.floor_.default,
            torch.ops.aten.floor_,
        ),
        inplace_targets=(
            torch.ops.aten.floor_.default,
            torch.ops.aten.floor_,
        ),
    ),
    "ceil": _unary_spec(
        "ceil",
        (
            torch.ceil,
            torch.ops.aten.ceil.default,
            torch.ops.aten.ceil,
            torch.ops.aten.ceil_.default,
            torch.ops.aten.ceil_,
        ),
        inplace_targets=(
            torch.ops.aten.ceil_.default,
            torch.ops.aten.ceil_,
        ),
    ),
    "reciprocal": _unary_spec(
        "reciprocal",
        (
            torch.reciprocal,
            torch.ops.aten.reciprocal.default,
            torch.ops.aten.reciprocal,
            torch.ops.aten.reciprocal_.default,
            torch.ops.aten.reciprocal_,
        ),
        inplace_targets=(
            torch.ops.aten.reciprocal_.default,
            torch.ops.aten.reciprocal_,
        ),
    ),
    "relu": _unary_spec(
        "relu",
        (
            torch.relu,
            torch.ops.aten.relu.default,
            torch.ops.aten.relu,
            torch.ops.aten.relu_.default,
            torch.ops.aten.relu_,
        ),
        inplace_targets=(
            torch.ops.aten.relu_.default,
            torch.ops.aten.relu_,
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
            torch.ops.aten.conj_physical_.default,
            torch.ops.aten.conj_physical_,
        ),
        inplace_targets=(
            torch.ops.aten.conj_physical_.default,
            torch.ops.aten.conj_physical_,
        ),
    ),
    "deg2rad": _unary_spec(
        "deg2rad",
        (
            torch.deg2rad,
            torch.ops.aten.deg2rad.default,
            torch.ops.aten.deg2rad,
            torch.ops.aten.deg2rad_.default,
            torch.ops.aten.deg2rad_,
        ),
        inplace_targets=(
            torch.ops.aten.deg2rad_.default,
            torch.ops.aten.deg2rad_,
        ),
    ),
    "digamma": _unary_spec(
        "digamma",
        (
            torch.digamma,
            torch.ops.aten.digamma.default,
            torch.ops.aten.digamma,
            torch.ops.aten.digamma_.default,
            torch.ops.aten.digamma_,
        ),
        inplace_targets=(
            torch.ops.aten.digamma_.default,
            torch.ops.aten.digamma_,
        ),
    ),
    "erfinv": _unary_spec(
        "erfinv",
        (
            torch.erfinv,
            torch.ops.aten.erfinv.default,
            torch.ops.aten.erfinv,
            torch.ops.aten.erfinv_.default,
            torch.ops.aten.erfinv_,
        ),
        inplace_targets=(
            torch.ops.aten.erfinv_.default,
            torch.ops.aten.erfinv_,
        ),
    ),
    "exp2": _unary_spec(
        "exp2",
        (
            torch.exp2,
            torch.ops.aten.exp2.default,
            torch.ops.aten.exp2,
            torch.ops.aten.exp2_.default,
            torch.ops.aten.exp2_,
        ),
        inplace_targets=(
            torch.ops.aten.exp2_.default,
            torch.ops.aten.exp2_,
        ),
    ),
    "frac": _unary_spec(
        "frac",
        (
            torch.frac,
            torch.ops.aten.frac.default,
            torch.ops.aten.frac,
            torch.ops.aten.frac_.default,
            torch.ops.aten.frac_,
        ),
        inplace_targets=(
            torch.ops.aten.frac_.default,
            torch.ops.aten.frac_,
        ),
    ),
    "i0": _unary_spec(
        "i0",
        (
            torch.i0,
            torch.ops.aten.i0.default,
            torch.ops.aten.i0,
            torch.ops.aten.i0_.default,
            torch.ops.aten.i0_,
        ),
        inplace_targets=(
            torch.ops.aten.i0_.default,
            torch.ops.aten.i0_,
        ),
    ),
    "isinf": _unary_spec(
        "isinf",
        (
            torch.isinf,
            torch.ops.aten.isinf.default,
            torch.ops.aten.isinf,
        ),
    ),
    "lgamma": _unary_spec(
        "lgamma",
        (
            torch.lgamma,
            torch.ops.aten.lgamma.default,
            torch.ops.aten.lgamma,
            torch.ops.aten.lgamma_.default,
            torch.ops.aten.lgamma_,
        ),
        inplace_targets=(
            torch.ops.aten.lgamma_.default,
            torch.ops.aten.lgamma_,
        ),
    ),
    "logit": _unary_spec(
        "logit",
        (
            torch.logit,
            torch.ops.aten.logit.default,
            torch.ops.aten.logit,
            torch.ops.aten.logit_.default,
            torch.ops.aten.logit_,
        ),
        inplace_targets=(
            torch.ops.aten.logit_.default,
            torch.ops.aten.logit_,
        ),
    ),
    "isfinite": _unary_spec(
        "isfinite",
        (
            torch.isfinite,
            torch.ops.aten.isfinite.default,
            torch.ops.aten.isfinite,
        ),
    ),
    "nan_to_num": _unary_spec(
        "nan_to_num",
        (
            torch.nan_to_num,
            torch.ops.aten.nan_to_num.default,
            torch.ops.aten.nan_to_num,
            torch.ops.aten.nan_to_num_.default,
            torch.ops.aten.nan_to_num_,
        ),
        inplace_targets=(
            torch.ops.aten.nan_to_num_.default,
            torch.ops.aten.nan_to_num_,
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
            torch.ops.aten.rad2deg_.default,
            torch.ops.aten.rad2deg_,
        ),
        inplace_targets=(
            torch.ops.aten.rad2deg_.default,
            torch.ops.aten.rad2deg_,
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
            torch.ops.aten.sgn_.default,
            torch.ops.aten.sgn_,
        ),
        inplace_targets=(
            torch.ops.aten.sgn_.default,
            torch.ops.aten.sgn_,
        ),
    ),
    "sinc": _unary_spec(
        "sinc",
        (
            torch.sinc,
            torch.ops.aten.sinc.default,
            torch.ops.aten.sinc,
            torch.ops.aten.sinc_.default,
            torch.ops.aten.sinc_,
        ),
        inplace_targets=(
            torch.ops.aten.sinc_.default,
            torch.ops.aten.sinc_,
        ),
    ),
    "square": _unary_spec(
        "square",
        (
            torch.square,
            torch.ops.aten.square.default,
            torch.ops.aten.square,
            torch.ops.aten.square_.default,
            torch.ops.aten.square_,
        ),
        inplace_targets=(
            torch.ops.aten.square_.default,
            torch.ops.aten.square_,
        ),
    ),
    "sum": _OpSpec(
        name="sum",
        kind="reduction",
        symbol=None,
        supported_targets={
            torch.ops.aten.sum.default,
        },
    ),
    "prod": _OpSpec(
        name="prod",
        kind="reduction",
        symbol=None,
        supported_targets={
            torch.ops.aten.prod.default,
        },
    ),
    "mean": _OpSpec(
        name="mean",
        kind="reduction",
        symbol=None,
        supported_targets={
            torch.mean,
            torch.ops.aten.mean.default,
            torch.ops.aten.mean,
        },
    ),
    "std": _OpSpec(
        name="std",
        kind="reduction",
        symbol=None,
        supported_targets={
            torch.std,
            torch.ops.aten.std.default,
            torch.ops.aten.std,
        },
    ),
    "any": _OpSpec(
        name="any",
        kind="reduction",
        symbol=None,
        supported_targets={
            torch.any,
            torch.ops.aten.any.default,
            torch.ops.aten.any,
        },
    ),
    "all": _OpSpec(
        name="all",
        kind="reduction",
        symbol=None,
        supported_targets={
            torch.all,
            torch.ops.aten.all.default,
            torch.ops.aten.all,
        },
    ),
    "cat": _OpSpec(
        name="cat",
        kind="concat",
        symbol=None,
        supported_targets={
            torch.cat,
            torch.ops.aten.cat.default,
            torch.ops.aten.cat,
        },
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
    inputs: Tuple[torch.fx.Node, ...]
    output_shape: Tuple[int, ...]
    inplace_input: int | None = None
    reduction_dims: Tuple[int, ...] | None = None
    keepdim: bool = False
    std_unbiased: bool | None = None
    concat_dim: int | None = None


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
        return "uint8_t"
    raise RefBackendError(
        "codegen backend supports only torch.float32, torch.int8, torch.int32, or torch.bool tensors"
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


def emit_body(
    op_spec: _OpSpec,
    output_access: str,
    input_shapes: Sequence[Sequence[int]],
    input_strides: Sequence[Sequence[int]],
    input_dtypes: Sequence[torch.dtype],
    output_shape: Sequence[int],
    indent: str,
    dtype: _CodegenDType,
) -> List[str]:
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
    return [f"{indent}{output_access} = {scalar_fn}({input_access});"]


def emit_footer(output_shape: Sequence[int], indent: str) -> List[str]:
    lines: List[str] = []
    if output_shape:
        for _ in range(len(output_shape)):
            indent = indent[:-4]
            lines.append(f"{indent}}}")
    lines.append("}")
    return lines


def _write_elementwise_kernel(
    node_index: int,
    op_spec: _OpSpec,
    output_shape: Sequence[int],
    input_shapes: Sequence[Sequence[int]],
    input_strides: Sequence[Sequence[int]],
    input_dtypes: Sequence[torch.dtype],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    lines = [
        emit_signature(
            node_index, op_spec, output_shape, input_shapes, input_dtypes, dtype
        )
    ]
    loop_lines, indent = emit_loops(output_shape)
    lines.extend(loop_lines)
    output_access = emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    lines.extend(
        emit_body(
            op_spec,
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
    reduction_count = 1
    for dim in reduction_dims:
        reduction_count *= input_shape[dim]
    bool_reduction = config.get("bool_reduction", False)
    acc_type = "int32_t" if bool_reduction else dtype.c_type
    if bool_reduction or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        init_value = str(config["init_value"])
    else:
        init_value = f"{config['init_value']}.0f"
    post_op = None
    if config["post_op"] == "mean":
        if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
            post_op = f"acc /= {reduction_count};"
        else:
            post_op = f"acc /= (float){reduction_count};"
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
        for _ in range(len(shape)):
            indent = indent[:-4]
            lines.append(f"{indent}}}")
        offset += shape[concat_dim]
    lines.append("}")
    return lines


def _write_generic_source(graph: _GenericGraph) -> str:
    placeholders = graph.tensor_placeholders
    op_nodes = graph.op_nodes
    headers = [
        "#include <stdint.h>",
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
                op_node.spec,
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
                    unbiased=bool(op_node.std_unbiased),
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
                op_node.concat_dim if op_node.concat_dim is not None else 0,
                graph.dtype,
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
    signature = (
        f"void ref_codegen_main_{graph.dtype.suffix}("
        f"{input_args}{graph.dtype.c_type} out{_format_array_suffix(graph.shapes[graph.output_value])}) {{"
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
        temp_decls.append(
            f"{graph.dtype.c_type} {temp_name}{_format_array_suffix(op_node.output_shape)};"
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
    tensor_examples = list(_iter_example_tensors(example_inputs))
    if not tensor_examples:
        raise RefBackendError("codegen backend requires at least one example tensor input")
    non_bool_examples = [
        example for example in tensor_examples if example.dtype is not torch.bool
    ]
    if not non_bool_examples:
        raise RefBackendError(
            "codegen backend requires at least one non-boolean tensor input"
        )
    first_dtype = non_bool_examples[0].dtype
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
    if op_spec.kind == "reduction":
        return ()
    if op_spec.kind == "concat":
        raise RefBackendError("codegen cat expects a tensor list input")
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
                    raise RefBackendError(
                        "codegen std expects unbiased to be specified once"
                    )
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
    dim = node.args[1] if len(node.args) > 1 else None
    keepdim = node.args[2] if len(node.args) > 2 else False
    dtype = node.args[3] if len(node.args) > 3 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise RefBackendError(
                    f"codegen {op_name} expects dim to be specified once"
                )
            dim = node.kwargs["dim"]
        if "keepdim" in node.kwargs:
            if len(node.args) > 2:
                raise RefBackendError(
                    f"codegen {op_name} expects keepdim to be specified once"
                )
            keepdim = node.kwargs["keepdim"]
        if "dtype" in node.kwargs:
            if dtype is not None:
                raise RefBackendError(
                    f"codegen {op_name} expects dtype to be specified once"
                )
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
                raise RefBackendError("codegen cat expects dim to be specified once")
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
            raise RefBackendError("codegen cat expects tensor inputs only")
    return list(tensors_arg), dim_value


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
                shapes[node] = tuple(example.shape)
                strides[node] = tuple(example.stride())
                dtypes[node] = example.dtype
                tensor_placeholders.append(node)
            continue
        if node.op in {"call_function", "call_method"}:
            if node.op == "call_method":
                if node.target not in {"sum", "prod", "mean", "std", "any", "all"}:
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
                input_nodes, concat_dim = _parse_concat_args(node)
                input_shapes: List[Tuple[int, ...]] = []
                for arg in input_nodes:
                    if arg not in shapes:
                        raise RefBackendError("codegen cat expects tensor inputs only")
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
                    raise RefBackendError(
                        "codegen cat expects inputs to share the graph dtype"
                    )
                output_shape = list(input_shapes[0])
                output_shape[concat_dim] = sum(
                    shape[concat_dim] for shape in input_shapes
                )
                output_shape = tuple(output_shape)
                shapes[node] = output_shape
                dtypes[node] = dtype_info.torch_dtype
                strides[node] = _contiguous_strides(output_shape)
                op_nodes.append(
                    _OpNode(
                        node=node,
                        spec=op_spec,
                        inputs=tuple(input_nodes),
                        output_shape=output_shape,
                        inplace_input=None,
                        concat_dim=concat_dim,
                    )
                )
                continue
            reduction_dims: Tuple[int, ...] | None = None
            keepdim = False
            reduce_all = False
            std_unbiased = None
            if op_spec.kind == "reduction":
                if len(node.args) < 1:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects one input"
                    )
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
            input_nodes: List[torch.fx.Node] = []
            input_shapes: List[Tuple[int, ...]] = []
            args_to_check = node.args
            if op_spec.kind == "reduction":
                args_to_check = node.args[:1]
            for arg in args_to_check:
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
                (
                    reduction_dims,
                    keepdim,
                    reduce_all,
                    std_unbiased,
                ) = _parse_reduction_args(op_spec.name, node, input_shapes[0])
                output_shape = _infer_reduction_output_shape(
                    input_shapes[0],
                    reduction_dims,
                    keepdim,
                    reduce_all=reduce_all,
                )
            else:
                output_shape = _infer_output_shape(op_spec, input_shapes)
            shapes[node] = output_shape
            dtypes[node] = dtype_info.torch_dtype
            if inplace_input is not None:
                strides[node] = strides[input_nodes[inplace_input]]
            else:
                strides[node] = _contiguous_strides(output_shape)
            op_nodes.append(
                _OpNode(
                    node=node,
                    spec=op_spec,
                    inputs=tuple(input_nodes),
                    output_shape=output_shape,
                    inplace_input=inplace_input,
                    reduction_dims=reduction_dims,
                    keepdim=keepdim,
                    std_unbiased=std_unbiased,
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

        input_shapes = tuple(tuple(tensor.shape) for tensor in input_tensors)
        input_strides = tuple(tuple(tensor.stride()) for tensor in input_tensors)
        cache_key = (input_shapes, input_strides)
        cached_lib = library_cache.get(cache_key)
        if cached_lib is None:
            updated_graph = _analyze_generic_graph(gm, list(normalized_args))
            cached_lib = _compile_generic_library(updated_graph)
            library_cache[cache_key] = cached_lib
        lib = cached_lib
        contiguous_inputs = list(input_tensors)
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
                dtype=graph.dtype.torch_dtype,
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
