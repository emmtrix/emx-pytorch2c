import operator
from typing import Callable, Dict, List

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list
from torch._decomp import get_decompositions
from torch._functorch.aot_autograd import aot_module_simplified

from .cffi_bindings import (
    RefBackendError,
    run_add,
    run_abs,
    run_acos,
    run_acosh,
    run_asin,
    run_asinh,
    run_atan,
    run_atanh,
    run_bmm,
    run_broadcast_in_dim,
    run_ceil,
    run_cos,
    run_cosh,
    _conv2d_output_shape,
    _normalize_conv2d_param,
    run_conv2d,
    run_div,
    run_erf,
    run_erfc,
    run_exp,
    run_expm1,
    run_floor,
    run_log,
    run_log1p,
    run_log10,
    run_log2,
    run_matmul,
    run_maximum,
    run_minimum,
    run_mul,
    run_neg,
    run_round,
    run_reciprocal,
    run_relu,
    run_angle,
    run_conj,
    run_conj_physical,
    run_deg2rad,
    run_digamma,
    run_erfinv,
    run_exp2,
    run_frac,
    run_i0,
    run_lgamma,
    run_logit,
    run_nan_to_num,
    run_positive,
    run_rad2deg,
    run_real,
    run_sgn,
    run_sinc,
    run_square,
    run_rsqrt,
    run_sigmoid,
    run_sin,
    run_sign,
    run_sinh,
    run_sqrt,
    run_sub,
    run_tan,
    run_tanh,
    run_trunc,
)


def _run_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_add(a, b, out)
    return out


def _run_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sub(a, b, out)
    return out


def _run_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_mul(a, b, out)
    return out


def _run_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_div(a, b, out)
    return out


def _run_maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_maximum(a, b, out)
    return out


def _run_minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_minimum(a, b, out)
    return out


def _run_neg(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_neg(a, out)
    return out


def _run_exp(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_exp(a, out)
    return out


def _run_abs(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_abs(a, out)
    return out


def _run_sqrt(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sqrt(a, out)
    return out


def _run_log(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_log(a, out)
    return out


def _run_sin(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sin(a, out)
    return out


def _run_cos(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_cos(a, out)
    return out


def _run_acos(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_acos(a, out)
    return out


def _run_acosh(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_acosh(a, out)
    return out


def _run_asin(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_asin(a, out)
    return out


def _run_asinh(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_asinh(a, out)
    return out


def _run_atan(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_atan(a, out)
    return out


def _run_atanh(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_atanh(a, out)
    return out


def _run_cosh(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_cosh(a, out)
    return out


def _run_sinh(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sinh(a, out)
    return out


def _run_tan(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_tan(a, out)
    return out


def _run_erf(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_erf(a, out)
    return out


def _run_erfc(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_erfc(a, out)
    return out


def _run_expm1(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_expm1(a, out)
    return out


def _run_log1p(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_log1p(a, out)
    return out


def _run_log2(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_log2(a, out)
    return out


def _run_log10(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_log10(a, out)
    return out


def _run_rsqrt(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_rsqrt(a, out)
    return out


def _run_sigmoid(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sigmoid(a, out)
    return out


def _run_sign(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sign(a, out)
    return out


def _run_round(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_round(a, out)
    return out


def _run_trunc(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_trunc(a, out)
    return out


def _run_tanh(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_tanh(a, out)
    return out


def _run_floor(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_floor(a, out)
    return out


def _run_ceil(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_ceil(a, out)
    return out


def _run_reciprocal(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_reciprocal(a, out)
    return out


def _run_relu(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_relu(a, out)
    return out


def _run_angle(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_angle(a, out)
    return out


def _run_conj(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_conj(a, out)
    return out


def _run_conj_physical(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_conj_physical(a, out)
    return out


def _run_deg2rad(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_deg2rad(a, out)
    return out


def _run_digamma(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_digamma(a, out)
    return out


def _run_erfinv(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_erfinv(a, out)
    return out


def _run_exp2(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_exp2(a, out)
    return out


def _run_frac(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_frac(a, out)
    return out


def _run_i0(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_i0(a, out)
    return out


def _run_lgamma(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_lgamma(a, out)
    return out


def _run_logit(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_logit(a, out)
    return out


def _run_nan_to_num(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_nan_to_num(a, out)
    return out


def _run_positive(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_positive(a, out)
    return out


def _run_rad2deg(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_rad2deg(a, out)
    return out


def _run_real(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_real(a, out)
    return out


def _run_sgn(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sgn(a, out)
    return out


def _run_sinc(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_sinc(a, out)
    return out


def _run_square(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_square(a, out)
    return out


def _run_conv2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: object,
    padding: object,
    dilation: object,
    groups: int,
) -> torch.Tensor:
    stride_pair = _normalize_conv2d_param("stride", stride)
    padding_pair = _normalize_conv2d_param("padding", padding)
    dilation_pair = _normalize_conv2d_param("dilation", dilation)
    out_shape = _conv2d_output_shape(
        input_tensor, weight, stride_pair, padding_pair, dilation_pair, groups
    )
    out = torch.empty(
        out_shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
        memory_format=torch.contiguous_format,
    )
    run_conv2d(
        input_tensor,
        weight,
        out,
        stride=stride_pair,
        padding=padding_pair,
        dilation=dilation_pair,
        groups=groups,
    )
    return out


def _run_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim == 3 and b.ndim == 3:
        out = torch.empty(
            (a.shape[0], a.shape[1], b.shape[2]),
            dtype=a.dtype,
            device=a.device,
            memory_format=torch.contiguous_format,
        )
        run_bmm(a, b, out)
        return out

    out = torch.empty(
        (a.shape[0], b.shape[1]),
        dtype=a.dtype,
        device=a.device,
        memory_format=torch.contiguous_format,
    )
    run_matmul(a, b, out)
    return out


def _run_broadcast_in_dim(
    a: torch.Tensor, shape: List[int], broadcast_dimensions: List[int]
) -> torch.Tensor:
    out = torch.empty(
        tuple(int(dim) for dim in shape),
        dtype=a.dtype,
        device=a.device,
        memory_format=torch.contiguous_format,
    )
    run_broadcast_in_dim(a, out, tuple(broadcast_dimensions))
    return out


def _run_expand(a: torch.Tensor, shape: List[int]) -> torch.Tensor:
    out_rank = len(shape)
    in_rank = a.ndim
    if out_rank < in_rank:
        raise RefBackendError("expand requires output rank >= input rank")
    resolved_shape = []
    leading = out_rank - in_rank
    for idx, dim in enumerate(shape):
        if dim == -1:
            if idx < leading:
                raise RefBackendError("expand cannot infer leading broadcast dimension")
            dim = a.shape[idx - leading]
        resolved_shape.append(int(dim))
    return a.expand(*resolved_shape)


def _compile_graph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    supported_targets = {
        operator.add: ("add", _run_add),
        torch.add: ("add", _run_add),
        torch.ops.prims.add: ("add", _run_add),
        torch.ops.prims.add.default: ("add", _run_add),
        torch.ops.aten.add.Tensor: ("add", _run_add),
        operator.sub: ("sub", _run_sub),
        torch.sub: ("sub", _run_sub),
        torch.ops.prims.sub: ("sub", _run_sub),
        torch.ops.prims.sub.default: ("sub", _run_sub),
        torch.ops.aten.sub.Tensor: ("sub", _run_sub),
        operator.mul: ("mul", _run_mul),
        torch.mul: ("mul", _run_mul),
        torch.ops.prims.mul: ("mul", _run_mul),
        torch.ops.prims.mul.default: ("mul", _run_mul),
        torch.ops.aten.mul.Tensor: ("mul", _run_mul),
        operator.truediv: ("div", _run_div),
        torch.div: ("div", _run_div),
        torch.ops.aten.div.Tensor: ("div", _run_div),
        torch.ops.aten.div: ("div", _run_div),
        torch.maximum: ("maximum", _run_maximum),
        torch.ops.aten.maximum.default: ("maximum", _run_maximum),
        torch.ops.aten.maximum: ("maximum", _run_maximum),
        torch.minimum: ("minimum", _run_minimum),
        torch.ops.aten.minimum.default: ("minimum", _run_minimum),
        torch.ops.aten.minimum: ("minimum", _run_minimum),
        operator.neg: ("neg", _run_neg),
        torch.neg: ("neg", _run_neg),
        torch.ops.aten.neg.default: ("neg", _run_neg),
        torch.ops.aten.neg: ("neg", _run_neg),
        torch.exp: ("exp", _run_exp),
        torch.ops.aten.exp.default: ("exp", _run_exp),
        torch.ops.aten.exp: ("exp", _run_exp),
        torch.abs: ("abs", _run_abs),
        torch.ops.aten.abs.default: ("abs", _run_abs),
        torch.ops.aten.abs: ("abs", _run_abs),
        torch.sqrt: ("sqrt", _run_sqrt),
        torch.ops.aten.sqrt.default: ("sqrt", _run_sqrt),
        torch.ops.aten.sqrt: ("sqrt", _run_sqrt),
        torch.log: ("log", _run_log),
        torch.ops.aten.log.default: ("log", _run_log),
        torch.ops.aten.log: ("log", _run_log),
        torch.sin: ("sin", _run_sin),
        torch.ops.aten.sin.default: ("sin", _run_sin),
        torch.ops.aten.sin: ("sin", _run_sin),
        torch.cos: ("cos", _run_cos),
        torch.ops.aten.cos.default: ("cos", _run_cos),
        torch.ops.aten.cos: ("cos", _run_cos),
        torch.acos: ("acos", _run_acos),
        torch.ops.aten.acos.default: ("acos", _run_acos),
        torch.ops.aten.acos: ("acos", _run_acos),
        torch.acosh: ("acosh", _run_acosh),
        torch.ops.aten.acosh.default: ("acosh", _run_acosh),
        torch.ops.aten.acosh: ("acosh", _run_acosh),
        torch.asin: ("asin", _run_asin),
        torch.ops.aten.asin.default: ("asin", _run_asin),
        torch.ops.aten.asin: ("asin", _run_asin),
        torch.asinh: ("asinh", _run_asinh),
        torch.ops.aten.asinh.default: ("asinh", _run_asinh),
        torch.ops.aten.asinh: ("asinh", _run_asinh),
        torch.atan: ("atan", _run_atan),
        torch.ops.aten.atan.default: ("atan", _run_atan),
        torch.ops.aten.atan: ("atan", _run_atan),
        torch.atanh: ("atanh", _run_atanh),
        torch.ops.aten.atanh.default: ("atanh", _run_atanh),
        torch.ops.aten.atanh: ("atanh", _run_atanh),
        torch.cosh: ("cosh", _run_cosh),
        torch.ops.aten.cosh.default: ("cosh", _run_cosh),
        torch.ops.aten.cosh: ("cosh", _run_cosh),
        torch.sinh: ("sinh", _run_sinh),
        torch.ops.aten.sinh.default: ("sinh", _run_sinh),
        torch.ops.aten.sinh: ("sinh", _run_sinh),
        torch.tan: ("tan", _run_tan),
        torch.ops.aten.tan.default: ("tan", _run_tan),
        torch.ops.aten.tan: ("tan", _run_tan),
        torch.erf: ("erf", _run_erf),
        torch.ops.aten.erf.default: ("erf", _run_erf),
        torch.ops.aten.erf: ("erf", _run_erf),
        torch.erfc: ("erfc", _run_erfc),
        torch.ops.aten.erfc.default: ("erfc", _run_erfc),
        torch.ops.aten.erfc: ("erfc", _run_erfc),
        torch.expm1: ("expm1", _run_expm1),
        torch.ops.aten.expm1.default: ("expm1", _run_expm1),
        torch.ops.aten.expm1: ("expm1", _run_expm1),
        torch.log1p: ("log1p", _run_log1p),
        torch.ops.aten.log1p.default: ("log1p", _run_log1p),
        torch.ops.aten.log1p: ("log1p", _run_log1p),
        torch.log2: ("log2", _run_log2),
        torch.ops.aten.log2.default: ("log2", _run_log2),
        torch.ops.aten.log2: ("log2", _run_log2),
        torch.log10: ("log10", _run_log10),
        torch.ops.aten.log10.default: ("log10", _run_log10),
        torch.ops.aten.log10: ("log10", _run_log10),
        torch.rsqrt: ("rsqrt", _run_rsqrt),
        torch.ops.aten.rsqrt.default: ("rsqrt", _run_rsqrt),
        torch.ops.aten.rsqrt: ("rsqrt", _run_rsqrt),
        torch.sigmoid: ("sigmoid", _run_sigmoid),
        torch.ops.aten.sigmoid.default: ("sigmoid", _run_sigmoid),
        torch.ops.aten.sigmoid: ("sigmoid", _run_sigmoid),
        torch.sign: ("sign", _run_sign),
        torch.ops.aten.sign.default: ("sign", _run_sign),
        torch.ops.aten.sign: ("sign", _run_sign),
        torch.round: ("round", _run_round),
        torch.ops.aten.round.default: ("round", _run_round),
        torch.ops.aten.round: ("round", _run_round),
        torch.trunc: ("trunc", _run_trunc),
        torch.ops.aten.trunc.default: ("trunc", _run_trunc),
        torch.ops.aten.trunc: ("trunc", _run_trunc),
        torch.tanh: ("tanh", _run_tanh),
        torch.ops.aten.tanh.default: ("tanh", _run_tanh),
        torch.ops.aten.tanh: ("tanh", _run_tanh),
        torch.floor: ("floor", _run_floor),
        torch.ops.aten.floor.default: ("floor", _run_floor),
        torch.ops.aten.floor: ("floor", _run_floor),
        torch.ceil: ("ceil", _run_ceil),
        torch.ops.aten.ceil.default: ("ceil", _run_ceil),
        torch.ops.aten.ceil: ("ceil", _run_ceil),
        torch.reciprocal: ("reciprocal", _run_reciprocal),
        torch.ops.aten.reciprocal.default: ("reciprocal", _run_reciprocal),
        torch.ops.aten.reciprocal: ("reciprocal", _run_reciprocal),
        torch.relu: ("relu", _run_relu),
        torch.ops.aten.relu.default: ("relu", _run_relu),
        torch.ops.aten.relu: ("relu", _run_relu),
        torch.angle: ("angle", _run_angle),
        torch.ops.aten.angle.default: ("angle", _run_angle),
        torch.ops.aten.angle: ("angle", _run_angle),
        torch.conj: ("conj", _run_conj),
        torch.ops.aten.conj.default: ("conj", _run_conj),
        torch.ops.aten.conj: ("conj", _run_conj),
        torch.conj_physical: ("conj_physical", _run_conj_physical),
        torch.ops.aten.conj_physical.default: ("conj_physical", _run_conj_physical),
        torch.ops.aten.conj_physical: ("conj_physical", _run_conj_physical),
        torch.deg2rad: ("deg2rad", _run_deg2rad),
        torch.ops.aten.deg2rad.default: ("deg2rad", _run_deg2rad),
        torch.ops.aten.deg2rad: ("deg2rad", _run_deg2rad),
        torch.digamma: ("digamma", _run_digamma),
        torch.ops.aten.digamma.default: ("digamma", _run_digamma),
        torch.ops.aten.digamma: ("digamma", _run_digamma),
        torch.erfinv: ("erfinv", _run_erfinv),
        torch.ops.aten.erfinv.default: ("erfinv", _run_erfinv),
        torch.ops.aten.erfinv: ("erfinv", _run_erfinv),
        torch.exp2: ("exp2", _run_exp2),
        torch.ops.aten.exp2.default: ("exp2", _run_exp2),
        torch.ops.aten.exp2: ("exp2", _run_exp2),
        torch.frac: ("frac", _run_frac),
        torch.ops.aten.frac.default: ("frac", _run_frac),
        torch.ops.aten.frac: ("frac", _run_frac),
        torch.i0: ("i0", _run_i0),
        torch.ops.aten.i0.default: ("i0", _run_i0),
        torch.ops.aten.i0: ("i0", _run_i0),
        torch.lgamma: ("lgamma", _run_lgamma),
        torch.ops.aten.lgamma.default: ("lgamma", _run_lgamma),
        torch.ops.aten.lgamma: ("lgamma", _run_lgamma),
        torch.logit: ("logit", _run_logit),
        torch.ops.aten.logit.default: ("logit", _run_logit),
        torch.ops.aten.logit: ("logit", _run_logit),
        torch.nan_to_num: ("nan_to_num", _run_nan_to_num),
        torch.ops.aten.nan_to_num.default: ("nan_to_num", _run_nan_to_num),
        torch.ops.aten.nan_to_num: ("nan_to_num", _run_nan_to_num),
        torch.positive: ("positive", _run_positive),
        torch.ops.aten.positive.default: ("positive", _run_positive),
        torch.ops.aten.positive: ("positive", _run_positive),
        torch.rad2deg: ("rad2deg", _run_rad2deg),
        torch.ops.aten.rad2deg.default: ("rad2deg", _run_rad2deg),
        torch.ops.aten.rad2deg: ("rad2deg", _run_rad2deg),
        torch.real: ("real", _run_real),
        torch.ops.aten.real.default: ("real", _run_real),
        torch.ops.aten.real: ("real", _run_real),
        torch.sgn: ("sgn", _run_sgn),
        torch.ops.aten.sgn.default: ("sgn", _run_sgn),
        torch.ops.aten.sgn: ("sgn", _run_sgn),
        torch.sinc: ("sinc", _run_sinc),
        torch.ops.aten.sinc.default: ("sinc", _run_sinc),
        torch.ops.aten.sinc: ("sinc", _run_sinc),
        torch.square: ("square", _run_square),
        torch.ops.aten.square.default: ("square", _run_square),
        torch.ops.aten.square: ("square", _run_square),
        torch.ops.aten.convolution.default: ("conv2d", _run_conv2d),
        torch.ops.aten.convolution: ("conv2d", _run_conv2d),
        torch.ops.aten.conv2d.default: ("conv2d", _run_conv2d),
        torch.ops.aten.conv2d: ("conv2d", _run_conv2d),
        operator.matmul: ("matmul", _run_matmul),
        torch.matmul: ("matmul", _run_matmul),
        torch.ops.aten.mm.default: ("matmul", _run_matmul),
        torch.ops.aten.mm: ("matmul", _run_matmul),
        torch.bmm: ("matmul", _run_matmul),
        torch.ops.aten.bmm.default: ("matmul", _run_matmul),
        torch.ops.aten.bmm: ("matmul", _run_matmul),
        torch.ops.aten.expand.default: ("expand", _run_expand),
        torch.ops.prims.broadcast_in_dim: ("broadcast_in_dim", _run_broadcast_in_dim),
        torch.ops.prims.broadcast_in_dim.default: (
            "broadcast_in_dim",
            _run_broadcast_in_dim,
        ),
    }
    unary_ops = {
        "neg",
        "exp",
        "abs",
        "sqrt",
        "log",
        "sin",
        "cos",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cosh",
        "sinh",
        "tan",
        "erf",
        "erfc",
        "expm1",
        "log1p",
        "log2",
        "log10",
        "rsqrt",
        "sigmoid",
        "sign",
        "round",
        "trunc",
        "tanh",
        "floor",
        "ceil",
        "reciprocal",
        "relu",
        "angle",
        "conj",
        "conj_physical",
        "deg2rad",
        "digamma",
        "erfinv",
        "exp2",
        "frac",
        "i0",
        "lgamma",
        "logit",
        "nan_to_num",
        "positive",
        "rad2deg",
        "real",
        "sgn",
        "sinc",
        "square",
    }

    def compiled(*args: torch.Tensor) -> torch.Tensor:
        env: Dict[str, torch.Tensor] = {}
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(args) != len(placeholders):
            raise RefBackendError(
                f"Expected {len(placeholders)} inputs, got {len(args)}"
            )
        for node, value in zip(placeholders, args):
            env[node.name] = value

        def resolve_output(value: object) -> object:
            if isinstance(value, torch.fx.Node):
                return env[value.name]
            if isinstance(value, (list, tuple, immutable_list)):
                resolved = [resolve_output(item) for item in value]
                return type(value)(resolved)
            raise RefBackendError("Unsupported output format")

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                if node.target not in supported_targets:
                    raise RefBackendError(f"Unsupported call_function: {node.target}")
                op_name, op_fn = supported_targets[node.target]
                if op_name == "broadcast_in_dim":
                    if node.kwargs:
                        raise RefBackendError(
                            "broadcast_in_dim expects positional arguments only"
                        )
                    if len(node.args) != 3:
                        raise RefBackendError(
                            "broadcast_in_dim expects tensor, shape, and dimensions"
                        )
                    input_arg, shape, broadcast_dimensions = node.args
                    if not isinstance(input_arg, torch.fx.Node):
                        raise RefBackendError(
                            "broadcast_in_dim expects tensor input only"
                        )
                    if isinstance(shape, torch.fx.Node) or isinstance(
                        broadcast_dimensions, torch.fx.Node
                    ):
                        raise RefBackendError(
                            "broadcast_in_dim expects constant shape and dimensions"
                        )
                    result = op_fn(
                        env[input_arg.name], list(shape), list(broadcast_dimensions)
                    )
                elif op_name == "expand":
                    if node.kwargs:
                        raise RefBackendError("expand expects positional arguments only")
                    if len(node.args) != 2:
                        raise RefBackendError("expand expects tensor and shape")
                    input_arg, shape = node.args
                    if not isinstance(input_arg, torch.fx.Node):
                        raise RefBackendError("expand expects tensor input only")
                    if isinstance(shape, torch.fx.Node):
                        raise RefBackendError("expand expects constant shape")
                    result = op_fn(env[input_arg.name], list(shape))
                elif op_name == "conv2d":
                    if node.kwargs:
                        raise RefBackendError("conv2d expects positional arguments only")
                    if len(node.args) == 7:
                        (
                            input_arg,
                            weight_arg,
                            bias,
                            stride,
                            padding,
                            dilation,
                            groups,
                        ) = node.args
                        transposed = False
                        output_padding = (0, 0)
                    elif len(node.args) == 9:
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
                        ) = node.args
                    else:
                        raise RefBackendError("conv2d expects convolution arguments")
                    if not isinstance(input_arg, torch.fx.Node) or not isinstance(
                        weight_arg, torch.fx.Node
                    ):
                        raise RefBackendError("conv2d expects tensor inputs only")
                    if isinstance(bias, torch.fx.Node) or bias is not None:
                        raise RefBackendError("conv2d does not support bias")
                    if isinstance(stride, torch.fx.Node) or isinstance(
                        padding, torch.fx.Node
                    ) or isinstance(dilation, torch.fx.Node):
                        raise RefBackendError(
                            "conv2d expects constant stride, padding, and dilation"
                        )
                    if isinstance(transposed, torch.fx.Node) or transposed:
                        raise RefBackendError("conv2d does not support transposed")
                    if isinstance(output_padding, torch.fx.Node) or output_padding not in (
                        (0, 0),
                        [0, 0],
                    ):
                        raise RefBackendError(
                            "conv2d expects zero output padding"
                        )
                    if isinstance(groups, torch.fx.Node):
                        raise RefBackendError("conv2d expects constant groups")
                    result = op_fn(
                        env[input_arg.name],
                        env[weight_arg.name],
                        stride,
                        padding,
                        dilation,
                        groups,
                    )
                else:
                    args_values = []
                    for arg in node.args:
                        if not isinstance(arg, torch.fx.Node):
                            raise RefBackendError(f"{op_name} expects tensor inputs only")
                        args_values.append(env[arg.name])
                    if op_name in unary_ops:
                        if len(args_values) != 1:
                            raise RefBackendError(f"{op_name} expects exactly one input")
                        result = op_fn(*args_values)
                    else:
                        if len(args_values) != 2:
                            raise RefBackendError(f"{op_name} expects exactly two inputs")
                        result = op_fn(*args_values)
                env[node.name] = result
                continue
            if node.op == "output":
                output_val = node.args[0]
                return resolve_output(output_val)
            raise RefBackendError(f"Unsupported node op: {node.op}")
        raise RefBackendError("Graph has no output node")

    return compiled


def c_ref_backend_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    if any(
        node.op == "call_function"
        and node.target
        in (
            torch.ops.prims.broadcast_in_dim,
            torch.ops.prims.broadcast_in_dim.default,
        )
        for node in gm.graph.nodes
    ):
        return _compile_graph(gm, example_inputs)

    decompositions = get_decompositions(
        [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
        ]
    )

    def fw_compiler(
        fx_gm: torch.fx.GraphModule, fx_example_inputs: List[torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        return _compile_graph(fx_gm, fx_example_inputs)

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=fw_compiler,
        decompositions=decompositions,
    )
