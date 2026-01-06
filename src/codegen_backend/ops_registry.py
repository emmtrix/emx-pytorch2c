from __future__ import annotations

import operator

import torch
import torch.nn.functional as F

from codegen_backend.specs import _OpSpec, _binary_spec, _unary_spec

_VALID_KINDS = {
    "binary",
    "unary",
    "fill",
    "view",
    "where",
    "flip",
    "arg_reduction",
    "reduction",
    "softmax",
    "cumsum",
    "concat",
    "diagonal",
    "addmm",
    "addbmm",
    "addmv",
    "addr",
    "matmul",
    "conv1d",
    "conv2d",
    "pool1d",
    "pool2d",
    "embedding",
    "batch_norm",
    "pdist",
}


def _flatten_targets(targets: tuple[object, ...]) -> list[object]:
    if len(targets) == 1 and isinstance(targets[0], (list, tuple, set)):
        return list(targets[0])
    return list(targets)


class _OpBuilder:
    def __init__(
        self,
        registry: "_OpRegistry",
        name: str,
        kind: str,
        symbol: str | None = None,
    ) -> None:
        self._registry = registry
        self._name = name
        self._kind = kind
        self._symbol = symbol
        self._targets: list[object] = []
        self._inplace_targets: list[object] = []
        self._inplace_arg_index: int | None = None

    def targets(self, *targets: object) -> "_OpBuilder":
        self._targets = _flatten_targets(targets)
        return self

    def inplace(self, *targets: object, arg_index: int | None = None) -> "_OpBuilder":
        self._inplace_targets = _flatten_targets(targets)
        self._inplace_arg_index = arg_index
        return self

    def build(self) -> _OpSpec:
        if not self._targets:
            raise ValueError(f"No targets registered for op '{self._name}'.")
        if self._kind == "binary":
            if self._inplace_arg_index not in (None, 0):
                raise ValueError(
                    f"Binary op '{self._name}' must use inplace_arg_index=0."
                )
            spec = _binary_spec(
                self._name,
                self._targets,
                self._symbol,
                inplace_targets=self._inplace_targets,
            )
        elif self._kind == "unary":
            if self._inplace_arg_index not in (None, 0):
                raise ValueError(
                    f"Unary op '{self._name}' must use inplace_arg_index=0."
                )
            spec = _unary_spec(
                self._name,
                self._targets,
                inplace_targets=self._inplace_targets,
            )
        else:
            spec = _OpSpec(
                name=self._name,
                kind=self._kind,
                symbol=self._symbol,
                supported_targets=set(self._targets),
                inplace_targets=set(self._inplace_targets),
                inplace_arg_index=self._inplace_arg_index,
            )
        self._registry._add(spec)
        return spec


class _OpRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, _OpSpec] = {}
        self._allowed_duplicate_targets: set[object] = set()

    def allow_duplicate_targets(self, *targets: object) -> "_OpRegistry":
        self._allowed_duplicate_targets.update(_flatten_targets(targets))
        return self

    def register_unary(self, name: str) -> _OpBuilder:
        return _OpBuilder(self, name, "unary")

    def register_binary(self, name: str, symbol: str | None = None) -> _OpBuilder:
        return _OpBuilder(self, name, "binary", symbol=symbol)

    def register_op(
        self, name: str, kind: str, symbol: str | None = None
    ) -> _OpBuilder:
        return _OpBuilder(self, name, kind, symbol=symbol)

    def _add(self, spec: _OpSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate op spec registered: {spec.name}")
        self._specs[spec.name] = spec

    def build(self) -> dict[str, _OpSpec]:
        _validate_registry(self._specs, self._allowed_duplicate_targets)
        return dict(self._specs)


def _validate_registry(
    specs: dict[str, _OpSpec],
    allow_duplicate_targets: set[object],
) -> None:
    for spec in specs.values():
        if spec.kind not in _VALID_KINDS:
            raise ValueError(
                f"Unsupported op kind '{spec.kind}' for op '{spec.name}'."
            )
        if spec.inplace_targets and spec.inplace_arg_index is None:
            raise ValueError(
                f"In-place targets require inplace_arg_index for op '{spec.name}'."
            )
    seen_targets: dict[object, str] = {}
    for spec in specs.values():
        for target in spec.supported_targets:
            if target in allow_duplicate_targets:
                continue
            if target in seen_targets and seen_targets[target] != spec.name:
                raise ValueError(
                    "Duplicate target registered for ops "
                    f"'{seen_targets[target]}' and '{spec.name}'."
                )
            seen_targets[target] = spec.name


_REGISTRY = _OpRegistry()

# To add a new op, register it here via register_unary/register_binary or
# register_op, then list the explicit targets and optional in-place targets.
_REGISTRY.register_binary("add", symbol="+").targets(
    operator.add,
    torch.add,
    torch.ops.prims.add,
    torch.ops.prims.add.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.add_,
).inplace(
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.add_,
).build()
_REGISTRY.register_op("_softmax", kind="softmax").targets(
    torch.ops.aten._softmax,
    torch.ops.aten._softmax.default,
).build()
_REGISTRY.register_unary("_to_copy").targets(
    torch.ops.aten._to_copy,
    torch.ops.aten._to_copy.default,
).build()
_REGISTRY.register_binary("sub", symbol="-").targets(
    operator.sub,
    torch.sub,
    torch.ops.prims.sub,
    torch.ops.prims.sub.default,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.sub_.Tensor,
    torch.ops.aten.sub_,
).inplace(
    torch.ops.aten.sub_.Tensor,
    torch.ops.aten.sub_,
).build()
_REGISTRY.register_binary("mul", symbol="*").targets(
    operator.mul,
    torch.mul,
    torch.ops.prims.mul,
    torch.ops.prims.mul.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.mul_.Tensor,
    torch.ops.aten.mul_,
).inplace(
    torch.ops.aten.mul_.Tensor,
    torch.ops.aten.mul_,
).build()
_REGISTRY.register_binary("bitwise_and").targets(
    operator.and_,
    torch.bitwise_and,
    torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.bitwise_and.Scalar,
    torch.ops.aten.bitwise_and_.Tensor,
    torch.ops.aten.bitwise_and_,
    torch.ops.aten.bitwise_and,
).inplace(
    torch.ops.aten.bitwise_and_.Tensor,
    torch.ops.aten.bitwise_and_,
).build()
_REGISTRY.register_binary("bitwise_or").targets(
    operator.or_,
    torch.bitwise_or,
    torch.ops.aten.bitwise_or.Tensor,
    torch.ops.aten.bitwise_or.Scalar,
    torch.ops.aten.bitwise_or_.Tensor,
    torch.ops.aten.bitwise_or_,
    torch.ops.aten.bitwise_or,
).inplace(
    torch.ops.aten.bitwise_or_.Tensor,
    torch.ops.aten.bitwise_or_,
).build()
_REGISTRY.register_binary("bitwise_xor").targets(
    operator.xor,
    torch.bitwise_xor,
    torch.ops.aten.bitwise_xor.Tensor,
    torch.ops.aten.bitwise_xor.Scalar,
    torch.ops.aten.bitwise_xor_.Tensor,
    torch.ops.aten.bitwise_xor_,
    torch.ops.aten.bitwise_xor,
).inplace(
    torch.ops.aten.bitwise_xor_.Tensor,
    torch.ops.aten.bitwise_xor_,
).build()
_REGISTRY.register_binary("logical_or").targets(
    torch.logical_or,
    torch.ops.aten.logical_or.default,
    torch.ops.aten.logical_or,
    torch.ops.aten.logical_or_.default,
    torch.ops.aten.logical_or_,
).inplace(
    torch.ops.aten.logical_or_.default,
    torch.ops.aten.logical_or_,
).build()
_REGISTRY.register_binary("logical_and").targets(
    torch.logical_and,
    torch.ops.aten.logical_and.default,
    torch.ops.aten.logical_and,
    torch.ops.aten.logical_and_.default,
    torch.ops.aten.logical_and_,
).inplace(
    torch.ops.aten.logical_and_.default,
    torch.ops.aten.logical_and_,
).build()
_REGISTRY.register_binary("logical_xor").targets(
    torch.logical_xor,
    torch.ops.aten.logical_xor.default,
    torch.ops.aten.logical_xor,
    torch.ops.aten.logical_xor_.default,
    torch.ops.aten.logical_xor_,
).inplace(
    torch.ops.aten.logical_xor_.default,
    torch.ops.aten.logical_xor_,
).build()
_REGISTRY.register_unary("logical_not").targets(
    torch.logical_not,
    torch.ops.aten.logical_not.default,
    torch.ops.aten.logical_not,
    torch.ops.aten.logical_not_.default,
    torch.ops.aten.logical_not_,
).inplace(
    torch.ops.aten.logical_not_.default,
    torch.ops.aten.logical_not_,
).build()
_REGISTRY.register_binary("bitwise_left_shift").targets(
    operator.lshift,
    torch.bitwise_left_shift,
    torch.ops.aten.bitwise_left_shift.Tensor,
    torch.ops.aten.bitwise_left_shift_.Tensor,
    torch.ops.aten.bitwise_left_shift_,
    torch.ops.aten.bitwise_left_shift,
).inplace(
    torch.ops.aten.bitwise_left_shift_.Tensor,
    torch.ops.aten.bitwise_left_shift_,
).build()
_REGISTRY.register_binary("bitwise_right_shift").targets(
    operator.rshift,
    torch.bitwise_right_shift,
    torch.ops.aten.bitwise_right_shift.Tensor,
    torch.ops.aten.bitwise_right_shift_.Tensor,
    torch.ops.aten.bitwise_right_shift_,
    torch.ops.aten.bitwise_right_shift,
).inplace(
    torch.ops.aten.bitwise_right_shift_.Tensor,
    torch.ops.aten.bitwise_right_shift_,
).build()
_REGISTRY.register_unary("bitwise_not").targets(
    operator.invert,
    torch.bitwise_not,
    torch.ops.aten.bitwise_not.default,
    torch.ops.aten.bitwise_not,
    torch.ops.aten.bitwise_not_.default,
    torch.ops.aten.bitwise_not_,
).inplace(
    torch.ops.aten.bitwise_not_.default,
    torch.ops.aten.bitwise_not_,
).build()
_REGISTRY.register_binary("div", symbol="/").targets(
    operator.truediv,
    torch.div,
    torch.true_divide,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
    torch.ops.aten.div.Tensor_mode,
    torch.ops.aten.div.Scalar_mode,
    torch.ops.aten.div,
    torch.ops.aten.div_.Tensor,
    torch.ops.aten.div_,
).inplace(
    torch.ops.aten.div_.Tensor,
    torch.ops.aten.div_,
).build()
_REGISTRY.register_binary("lt").targets(
    operator.lt,
    torch.lt,
    torch.ops.aten.lt.Tensor,
    torch.ops.aten.lt.Scalar,
    torch.ops.aten.lt.default,
    torch.ops.aten.lt,
).build()
_REGISTRY.register_binary("le").targets(
    operator.le,
    torch.le,
    torch.ops.aten.le.Tensor,
    torch.ops.aten.le.Scalar,
    torch.ops.aten.le.default,
    torch.ops.aten.le,
).build()
_REGISTRY.register_binary("gt").targets(
    operator.gt,
    torch.gt,
    torch.ops.aten.gt.Tensor,
    torch.ops.aten.gt.Scalar,
    torch.ops.aten.gt.default,
    torch.ops.aten.gt,
).build()
_REGISTRY.register_binary("ge").targets(
    operator.ge,
    torch.ge,
    torch.ops.aten.ge.Tensor,
    torch.ops.aten.ge.Scalar,
    torch.ops.aten.ge.default,
    torch.ops.aten.ge,
).build()
_REGISTRY.register_binary("eq").targets(
    operator.eq,
    torch.eq,
    torch.ops.aten.eq.Tensor,
    torch.ops.aten.eq.Scalar,
    torch.ops.aten.eq.default,
    torch.ops.aten.eq,
).build()
_REGISTRY.register_binary("ne").targets(
    operator.ne,
    torch.ne,
    torch.ops.aten.ne.Tensor,
    torch.ops.aten.ne.Scalar,
    torch.ops.aten.ne.default,
    torch.ops.aten.ne,
).build()
_REGISTRY.register_binary("maximum").targets(
    torch.maximum,
    torch.ops.aten.maximum.default,
    torch.ops.aten.maximum,
).build()
_REGISTRY.register_binary("minimum").targets(
    torch.minimum,
    torch.ops.aten.minimum.default,
    torch.ops.aten.minimum,
).build()
_REGISTRY.register_binary("atan2").targets(
    torch.atan2,
    torch.ops.aten.atan2.default,
    torch.ops.aten.atan2,
    torch.ops.aten.atan2_.default,
    torch.ops.aten.atan2_,
).inplace(
    torch.ops.aten.atan2_.default,
    torch.ops.aten.atan2_,
).build()
_REGISTRY.register_binary("pow").targets(
    operator.pow,
    torch.pow,
    torch.ops.aten.pow.Tensor_Tensor,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.pow.Scalar,
    torch.ops.aten.pow_.Tensor,
    torch.ops.aten.pow_.Scalar,
    torch.ops.aten.pow_,
).inplace(
    torch.ops.aten.pow_.Tensor,
    torch.ops.aten.pow_.Scalar,
    torch.ops.aten.pow_,
).build()
_REGISTRY.register_binary("remainder").targets(
    torch.remainder,
    torch.ops.aten.remainder.Tensor,
    torch.ops.aten.remainder.Scalar,
    torch.ops.aten.remainder,
    torch.ops.aten.remainder_.Tensor,
    torch.ops.aten.remainder_,
).inplace(
    torch.ops.aten.remainder_.Tensor,
    torch.ops.aten.remainder_,
).build()
_REGISTRY.register_binary("fmod").targets(
    torch.fmod,
    torch.ops.aten.fmod.Tensor,
    torch.ops.aten.fmod.Scalar,
    torch.ops.aten.fmod,
    torch.ops.aten.fmod_.Tensor,
    torch.ops.aten.fmod_,
).inplace(
    torch.ops.aten.fmod_.Tensor,
    torch.ops.aten.fmod_,
).build()
_REGISTRY.register_binary("floor_divide").targets(
    torch.floor_divide,
    torch.ops.aten.floor_divide.default,
    torch.ops.aten.floor_divide.Scalar,
    torch.ops.aten.floor_divide,
    torch.ops.aten.floor_divide_.Tensor,
    torch.ops.aten.floor_divide_,
).inplace(
    torch.ops.aten.floor_divide_.Tensor,
    torch.ops.aten.floor_divide_,
).build()
_REGISTRY.register_binary("fmax").targets(
    torch.fmax,
    torch.ops.aten.fmax.default,
    torch.ops.aten.fmax,
).build()
_REGISTRY.register_binary("fmin").targets(
    torch.fmin,
    torch.ops.aten.fmin.default,
    torch.ops.aten.fmin,
).build()
_REGISTRY.register_binary("copysign").targets(
    torch.copysign,
    torch.ops.aten.copysign.default,
    torch.ops.aten.copysign.Tensor,
    torch.ops.aten.copysign.Scalar,
    torch.ops.aten.copysign,
    torch.ops.aten.copysign_.Tensor,
    torch.ops.aten.copysign_,
).inplace(
    torch.ops.aten.copysign_.Tensor,
    torch.ops.aten.copysign_,
).build()
_REGISTRY.register_binary("hypot").targets(
    torch.hypot,
    torch.ops.aten.hypot.default,
    torch.ops.aten.hypot,
    torch.ops.aten.hypot_.default,
    torch.ops.aten.hypot_,
).inplace(
    torch.ops.aten.hypot_.default,
    torch.ops.aten.hypot_,
).build()
_REGISTRY.register_binary("logaddexp").targets(
    torch.logaddexp,
    torch.ops.aten.logaddexp.default,
    torch.ops.aten.logaddexp,
).build()
_REGISTRY.register_binary("logaddexp2").targets(
    torch.logaddexp2,
    torch.ops.aten.logaddexp2.default,
    torch.ops.aten.logaddexp2,
).build()
_REGISTRY.register_binary("nextafter").targets(
    torch.nextafter,
    torch.ops.aten.nextafter.default,
    torch.ops.aten.nextafter,
    torch.ops.aten.nextafter_.default,
    torch.ops.aten.nextafter_,
).inplace(
    torch.ops.aten.nextafter_.default,
    torch.ops.aten.nextafter_,
).build()
_REGISTRY.register_binary("xlogy").targets(
    torch.xlogy,
    torch.ops.aten.xlogy.Tensor,
    torch.ops.aten.xlogy,
    torch.ops.aten.xlogy_.Tensor,
    torch.ops.aten.xlogy_,
).inplace(
    torch.ops.aten.xlogy_.Tensor,
    torch.ops.aten.xlogy_,
).build()
_REGISTRY.register_binary("heaviside").targets(
    torch.heaviside,
    torch.ops.aten.heaviside.default,
    torch.ops.aten.heaviside,
    torch.ops.aten.heaviside_.default,
    torch.ops.aten.heaviside_,
).inplace(
    torch.ops.aten.heaviside_.default,
    torch.ops.aten.heaviside_,
).build()
_REGISTRY.register_op("where", kind="where").targets(
    torch.where,
    torch.ops.aten.where.self,
    torch.ops.aten.where.Scalar,
).build()
_REGISTRY.register_op("flip", kind="flip").targets(
    torch.flip,
    torch.ops.aten.flip.default,
    torch.ops.aten.flip,
).build()
_REGISTRY.register_binary("ldexp").targets(
    torch.ldexp,
    torch.ops.aten.ldexp.default,
    torch.ops.aten.ldexp,
    torch.ops.aten.ldexp_.default,
    torch.ops.aten.ldexp_,
).inplace(
    torch.ops.aten.ldexp_.default,
    torch.ops.aten.ldexp_,
).build()
_REGISTRY.register_binary("clamp_min").targets(
    torch.clamp_min,
    torch.ops.aten.clamp_min.Tensor,
    torch.ops.aten.clamp_min.default,
    torch.ops.aten.clamp_min,
    torch.ops.aten.clamp_min_.default,
    torch.ops.aten.clamp_min_.Tensor,
    torch.ops.aten.clamp_min_,
).inplace(
    torch.ops.aten.clamp_min_.default,
    torch.ops.aten.clamp_min_.Tensor,
    torch.ops.aten.clamp_min_,
).build()
_REGISTRY.register_binary("clamp_max").targets(
    torch.clamp_max,
    torch.ops.aten.clamp_max.Tensor,
    torch.ops.aten.clamp_max.default,
    torch.ops.aten.clamp_max,
    torch.ops.aten.clamp_max_.default,
    torch.ops.aten.clamp_max_.Tensor,
    torch.ops.aten.clamp_max_,
).inplace(
    torch.ops.aten.clamp_max_.default,
    torch.ops.aten.clamp_max_.Tensor,
    torch.ops.aten.clamp_max_,
).build()
_REGISTRY.register_unary("clamp").targets(
    torch.clamp,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp,
    torch.ops.aten.clamp_.default,
    torch.ops.aten.clamp_,
).inplace(
    torch.ops.aten.clamp_.default,
    torch.ops.aten.clamp_,
).build()
_REGISTRY.register_unary("neg").targets(
    operator.neg,
    torch.neg,
    torch.ops.aten.neg.default,
    torch.ops.aten.neg,
    torch.ops.aten.neg_.default,
    torch.ops.aten.neg_,
).inplace(
    torch.ops.aten.neg_.default,
    torch.ops.aten.neg_,
).build()
_REGISTRY.register_unary("exp").targets(
    torch.exp,
    torch.ops.aten.exp.default,
    torch.ops.aten.exp,
    torch.ops.aten.exp_.default,
    torch.ops.aten.exp_,
).inplace(
    torch.ops.aten.exp_.default,
    torch.ops.aten.exp_,
).build()
_REGISTRY.register_unary("abs").targets(
    torch.abs,
    torch.ops.aten.abs.default,
    torch.ops.aten.abs,
    torch.ops.aten.abs_.default,
    torch.ops.aten.abs_,
).inplace(
    torch.ops.aten.abs_.default,
    torch.ops.aten.abs_,
).build()
_REGISTRY.register_op("cumsum", kind="cumsum").targets(
    torch.cumsum,
    torch.ops.aten.cumsum.default,
    torch.ops.aten.cumsum,
).build()
_REGISTRY.register_unary("absolute").targets(
    torch.absolute,
    torch.ops.aten.absolute.default,
    torch.ops.aten.absolute,
    torch.ops.aten.absolute_.default,
    torch.ops.aten.absolute_,
).inplace(
    torch.ops.aten.absolute_.default,
    torch.ops.aten.absolute_,
).build()
_REGISTRY.register_unary("sqrt").targets(
    torch.sqrt,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.sqrt,
    torch.ops.aten.sqrt_.default,
    torch.ops.aten.sqrt_,
).inplace(
    torch.ops.aten.sqrt_.default,
    torch.ops.aten.sqrt_,
).build()
_cbrt_targets = []
cbrt_op = getattr(torch, "cbrt", None)
if cbrt_op is not None:
    _cbrt_targets.append(cbrt_op)
aten_cbrt = getattr(torch.ops.aten, "cbrt", None)
if aten_cbrt is not None:
    _cbrt_targets.extend([aten_cbrt.default, aten_cbrt])
aten_cbrt_inplace = getattr(torch.ops.aten, "cbrt_", None)
_cbrt_inplace_targets = []
if aten_cbrt_inplace is not None:
    _cbrt_inplace_targets.extend(
        [aten_cbrt_inplace.default, aten_cbrt_inplace]
    )
if _cbrt_targets:
    _REGISTRY.register_unary("cbrt").targets(_cbrt_targets).inplace(
        _cbrt_inplace_targets
    ).build()
_REGISTRY.register_unary("log").targets(
    torch.log,
    torch.ops.aten.log.default,
    torch.ops.aten.log,
    torch.ops.aten.log_.default,
    torch.ops.aten.log_,
).inplace(
    torch.ops.aten.log_.default,
    torch.ops.aten.log_,
).build()
_REGISTRY.register_unary("sin").targets(
    torch.sin,
    torch.ops.aten.sin.default,
    torch.ops.aten.sin,
    torch.ops.aten.sin_.default,
    torch.ops.aten.sin_,
).inplace(
    torch.ops.aten.sin_.default,
    torch.ops.aten.sin_,
).build()
_REGISTRY.register_unary("cos").targets(
    torch.cos,
    torch.ops.aten.cos.default,
    torch.ops.aten.cos,
    torch.ops.aten.cos_.default,
    torch.ops.aten.cos_,
).inplace(
    torch.ops.aten.cos_.default,
    torch.ops.aten.cos_,
).build()
_REGISTRY.register_unary("acos").targets(
    torch.acos,
    torch.ops.aten.acos.default,
    torch.ops.aten.acos,
    torch.ops.aten.acos_.default,
    torch.ops.aten.acos_,
).inplace(
    torch.ops.aten.acos_.default,
    torch.ops.aten.acos_,
).build()
_REGISTRY.register_unary("arccos").targets(
    torch.arccos,
    torch.ops.aten.arccos.default,
    torch.ops.aten.arccos,
    torch.ops.aten.arccos_.default,
    torch.ops.aten.arccos_,
).inplace(
    torch.ops.aten.arccos_.default,
    torch.ops.aten.arccos_,
).build()
_REGISTRY.register_unary("acosh").targets(
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
).inplace(
    torch.ops.aten.acosh_.default,
    torch.ops.aten.acosh_,
    torch.ops.aten.arccosh_.default,
    torch.ops.aten.arccosh_,
).build()
_REGISTRY.register_unary("asin").targets(
    torch.asin,
    torch.ops.aten.asin.default,
    torch.ops.aten.asin,
    torch.ops.aten.asin_.default,
    torch.ops.aten.asin_,
).inplace(
    torch.ops.aten.asin_.default,
    torch.ops.aten.asin_,
).build()
_REGISTRY.register_unary("arcsin").targets(
    torch.arcsin,
    torch.ops.aten.arcsin.default,
    torch.ops.aten.arcsin,
    torch.ops.aten.arcsin_.default,
    torch.ops.aten.arcsin_,
).inplace(
    torch.ops.aten.arcsin_.default,
    torch.ops.aten.arcsin_,
).build()
_REGISTRY.register_unary("asinh").targets(
    torch.asinh,
    torch.ops.aten.asinh.default,
    torch.ops.aten.asinh,
    torch.ops.aten.asinh_.default,
    torch.ops.aten.asinh_,
).inplace(
    torch.ops.aten.asinh_.default,
    torch.ops.aten.asinh_,
).build()
_REGISTRY.register_unary("arcsinh").targets(
    torch.arcsinh,
    torch.ops.aten.arcsinh.default,
    torch.ops.aten.arcsinh,
    torch.ops.aten.arcsinh_.default,
    torch.ops.aten.arcsinh_,
).inplace(
    torch.ops.aten.arcsinh_.default,
    torch.ops.aten.arcsinh_,
).build()
_REGISTRY.register_unary("atan").targets(
    torch.atan,
    torch.ops.aten.atan.default,
    torch.ops.aten.atan,
    torch.ops.aten.atan_.default,
    torch.ops.aten.atan_,
).inplace(
    torch.ops.aten.atan_.default,
    torch.ops.aten.atan_,
).build()
_REGISTRY.register_unary("arctan").targets(
    torch.arctan,
    torch.ops.aten.arctan.default,
    torch.ops.aten.arctan,
    torch.ops.aten.arctan_.default,
    torch.ops.aten.arctan_,
).inplace(
    torch.ops.aten.arctan_.default,
    torch.ops.aten.arctan_,
).build()
_REGISTRY.register_unary("atanh").targets(
    torch.atanh,
    torch.ops.aten.atanh.default,
    torch.ops.aten.atanh,
    torch.ops.aten.atanh_.default,
    torch.ops.aten.atanh_,
).inplace(
    torch.ops.aten.atanh_.default,
    torch.ops.aten.atanh_,
).build()
_REGISTRY.register_unary("cosh").targets(
    torch.cosh,
    torch.ops.aten.cosh.default,
    torch.ops.aten.cosh,
    torch.ops.aten.cosh_.default,
    torch.ops.aten.cosh_,
).inplace(
    torch.ops.aten.cosh_.default,
    torch.ops.aten.cosh_,
).build()
_REGISTRY.register_unary("sinh").targets(
    torch.sinh,
    torch.ops.aten.sinh.default,
    torch.ops.aten.sinh,
    torch.ops.aten.sinh_.default,
    torch.ops.aten.sinh_,
).inplace(
    torch.ops.aten.sinh_.default,
    torch.ops.aten.sinh_,
).build()
_REGISTRY.register_unary("tan").targets(
    torch.tan,
    torch.ops.aten.tan.default,
    torch.ops.aten.tan,
    torch.ops.aten.tan_.default,
    torch.ops.aten.tan_,
).inplace(
    torch.ops.aten.tan_.default,
    torch.ops.aten.tan_,
).build()
_REGISTRY.register_unary("erf").targets(
    torch.erf,
    torch.ops.aten.erf.default,
    torch.ops.aten.erf,
    torch.ops.aten.erf_.default,
    torch.ops.aten.erf_,
).inplace(
    torch.ops.aten.erf_.default,
    torch.ops.aten.erf_,
).build()
_REGISTRY.register_unary("erfc").targets(
    torch.erfc,
    torch.ops.aten.erfc.default,
    torch.ops.aten.erfc,
    torch.ops.aten.erfc_.default,
    torch.ops.aten.erfc_,
).inplace(
    torch.ops.aten.erfc_.default,
    torch.ops.aten.erfc_,
).build()
_REGISTRY.register_unary("expm1").targets(
    torch.expm1,
    torch.ops.aten.expm1.default,
    torch.ops.aten.expm1,
    torch.ops.aten.expm1_.default,
    torch.ops.aten.expm1_,
).inplace(
    torch.ops.aten.expm1_.default,
    torch.ops.aten.expm1_,
).build()
_REGISTRY.register_unary("log1p").targets(
    torch.log1p,
    torch.ops.aten.log1p.default,
    torch.ops.aten.log1p,
    torch.ops.aten.log1p_.default,
    torch.ops.aten.log1p_,
).inplace(
    torch.ops.aten.log1p_.default,
    torch.ops.aten.log1p_,
).build()
_REGISTRY.register_unary("log2").targets(
    torch.log2,
    torch.ops.aten.log2.default,
    torch.ops.aten.log2,
    torch.ops.aten.log2_.default,
    torch.ops.aten.log2_,
).inplace(
    torch.ops.aten.log2_.default,
    torch.ops.aten.log2_,
).build()
_REGISTRY.register_unary("log10").targets(
    torch.log10,
    torch.ops.aten.log10.default,
    torch.ops.aten.log10,
    torch.ops.aten.log10_.default,
    torch.ops.aten.log10_,
).inplace(
    torch.ops.aten.log10_.default,
    torch.ops.aten.log10_,
).build()
_REGISTRY.register_unary("rsqrt").targets(
    torch.rsqrt,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.rsqrt,
    torch.ops.aten.rsqrt_.default,
    torch.ops.aten.rsqrt_,
).inplace(
    torch.ops.aten.rsqrt_.default,
    torch.ops.aten.rsqrt_,
).build()
_REGISTRY.register_unary("sigmoid").targets(
    torch.sigmoid,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.sigmoid,
    torch.ops.aten.sigmoid_.default,
    torch.ops.aten.sigmoid_,
).inplace(
    torch.ops.aten.sigmoid_.default,
    torch.ops.aten.sigmoid_,
).build()
_REGISTRY.register_unary("log_sigmoid").targets(
    F.logsigmoid,
    torch.ops.aten.log_sigmoid.default,
    torch.ops.aten.log_sigmoid,
).build()
_REGISTRY.register_unary("gelu").targets(
    F.gelu,
    torch.ops.aten.gelu.default,
    torch.ops.aten.gelu,
    torch.ops.aten.gelu_.default,
    torch.ops.aten.gelu_,
).inplace(
    torch.ops.aten.gelu_.default,
    torch.ops.aten.gelu_,
).build()
_REGISTRY.register_unary("elu").targets(
    F.elu,
    torch.ops.aten.elu.default,
    torch.ops.aten.elu,
    torch.ops.aten.elu_.default,
    torch.ops.aten.elu_,
).inplace(
    torch.ops.aten.elu_.default,
    torch.ops.aten.elu_,
).build()
_REGISTRY.register_unary("leaky_relu").targets(
    F.leaky_relu,
    torch.ops.aten.leaky_relu.default,
    torch.ops.aten.leaky_relu,
    torch.ops.aten.leaky_relu_.default,
    torch.ops.aten.leaky_relu_,
).inplace(
    torch.ops.aten.leaky_relu_.default,
    torch.ops.aten.leaky_relu_,
).build()
_REGISTRY.register_unary("softplus").targets(
    F.softplus,
    torch.ops.aten.softplus.default,
    torch.ops.aten.softplus,
).build()
_REGISTRY.register_unary("selu").targets(
    F.selu,
    torch.ops.aten.selu.default,
    torch.ops.aten.selu,
    torch.ops.aten.selu_.default,
    torch.ops.aten.selu_,
).inplace(
    torch.ops.aten.selu_.default,
    torch.ops.aten.selu_,
).build()
_REGISTRY.register_unary("relu6").targets(
    F.relu6,
    torch.ops.aten.relu6.default,
    torch.ops.aten.relu6,
).build()
_REGISTRY.register_unary("hardsigmoid").targets(
    F.hardsigmoid,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardsigmoid,
).build()
_REGISTRY.register_unary("hardtanh").targets(
    F.hardtanh,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardtanh,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.hardtanh_,
).inplace(
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.hardtanh_,
).build()
_REGISTRY.register_unary("mish").targets(
    F.mish,
    torch.ops.aten.mish.default,
    torch.ops.aten.mish,
    torch.ops.aten.mish_.default,
    torch.ops.aten.mish_,
).inplace(
    torch.ops.aten.mish_.default,
    torch.ops.aten.mish_,
).build()
_REGISTRY.register_unary("silu").targets(
    F.silu,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu,
    torch.ops.aten.silu_.default,
    torch.ops.aten.silu_,
).inplace(
    torch.ops.aten.silu_.default,
    torch.ops.aten.silu_,
).build()
_REGISTRY.register_unary("hardswish").targets(
    F.hardswish,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.hardswish_,
).inplace(
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.hardswish_,
).build()
_REGISTRY.register_unary("sign").targets(
    torch.sign,
    torch.ops.aten.sign.default,
    torch.ops.aten.sign,
    torch.ops.aten.sign_.default,
    torch.ops.aten.sign_,
).inplace(
    torch.ops.aten.sign_.default,
    torch.ops.aten.sign_,
).build()
_REGISTRY.register_unary("round").targets(
    torch.round,
    torch.ops.aten.round.default,
    torch.ops.aten.round,
    torch.ops.aten.round_.default,
    torch.ops.aten.round_,
).inplace(
    torch.ops.aten.round_.default,
    torch.ops.aten.round_,
).build()
_REGISTRY.register_unary("trunc").targets(
    torch.trunc,
    torch.ops.aten.trunc.default,
    torch.ops.aten.trunc,
    torch.ops.aten.trunc_.default,
    torch.ops.aten.trunc_,
).inplace(
    torch.ops.aten.trunc_.default,
    torch.ops.aten.trunc_,
).build()
_REGISTRY.register_unary("tanh").targets(
    torch.tanh,
    torch.ops.aten.tanh.default,
    torch.ops.aten.tanh,
    torch.ops.aten.tanh_.default,
    torch.ops.aten.tanh_,
).inplace(
    torch.ops.aten.tanh_.default,
    torch.ops.aten.tanh_,
).build()
_REGISTRY.register_unary("floor").targets(
    torch.floor,
    torch.ops.aten.floor.default,
    torch.ops.aten.floor,
    torch.ops.aten.floor_.default,
    torch.ops.aten.floor_,
).inplace(
    torch.ops.aten.floor_.default,
    torch.ops.aten.floor_,
).build()
_REGISTRY.register_unary("ceil").targets(
    torch.ceil,
    torch.ops.aten.ceil.default,
    torch.ops.aten.ceil,
    torch.ops.aten.ceil_.default,
    torch.ops.aten.ceil_,
).inplace(
    torch.ops.aten.ceil_.default,
    torch.ops.aten.ceil_,
).build()
_REGISTRY.register_unary("reciprocal").targets(
    torch.reciprocal,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.reciprocal,
    torch.ops.aten.reciprocal_.default,
    torch.ops.aten.reciprocal_,
).inplace(
    torch.ops.aten.reciprocal_.default,
    torch.ops.aten.reciprocal_,
).build()
_REGISTRY.register_unary("relu").targets(
    torch.relu,
    torch.ops.aten.relu.default,
    torch.ops.aten.relu,
    torch.ops.aten.relu_.default,
    torch.ops.aten.relu_,
).inplace(
    torch.ops.aten.relu_.default,
    torch.ops.aten.relu_,
).build()
_REGISTRY.register_unary("angle").targets(
    torch.angle,
    torch.ops.aten.angle.default,
    torch.ops.aten.angle,
).build()
_REGISTRY.register_unary("conj").targets(
    torch.conj,
    torch.ops.aten.conj.default,
    torch.ops.aten.conj,
).build()
_REGISTRY.register_unary("conj_physical").targets(
    torch.conj_physical,
    torch.ops.aten.conj_physical.default,
    torch.ops.aten.conj_physical,
    torch.ops.aten.conj_physical_.default,
    torch.ops.aten.conj_physical_,
).inplace(
    torch.ops.aten.conj_physical_.default,
    torch.ops.aten.conj_physical_,
).build()
_REGISTRY.register_unary("clone").targets(
    torch.clone,
    torch.ops.aten.clone.default,
    torch.ops.aten.clone,
).build()
_REGISTRY.register_unary("alias").targets(
    torch.ops.aten.alias.default,
    torch.ops.aten.alias,
).build()
_REGISTRY.register_binary("copy").targets(
    torch.ops.aten.copy.default,
    torch.ops.aten.copy,
).build()
_REGISTRY.register_unary("resize_").targets(
    torch.ops.aten.resize_.default,
).inplace(
    torch.ops.aten.resize_.default,
).build()
_REGISTRY.register_op("fill", "fill").targets(
    torch.ops.aten.fill.Scalar,
    torch.ops.aten.fill,
    torch.ops.aten.fill_.Scalar,
    torch.ops.aten.fill_,
).inplace(
    torch.ops.aten.fill_.Scalar,
    torch.ops.aten.fill_,
    arg_index=0,
).build()
_REGISTRY.register_op("full_like", "fill").targets(
    torch.ops.aten.full_like.default,
    torch.ops.aten.full_like,
).build()
_REGISTRY.register_op("as_strided", kind="view").targets(
    torch.ops.aten.as_strided.default,
    torch.ops.aten.as_strided,
).build()
_REGISTRY.register_op("squeeze", kind="view").targets(
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
).build()
_REGISTRY.register_unary("deg2rad").targets(
    torch.deg2rad,
    torch.ops.aten.deg2rad.default,
    torch.ops.aten.deg2rad,
    torch.ops.aten.deg2rad_.default,
    torch.ops.aten.deg2rad_,
).inplace(
    torch.ops.aten.deg2rad_.default,
    torch.ops.aten.deg2rad_,
).build()
_REGISTRY.register_unary("digamma").targets(
    torch.digamma,
    torch.ops.aten.digamma.default,
    torch.ops.aten.digamma,
    torch.ops.aten.digamma_.default,
    torch.ops.aten.digamma_,
).inplace(
    torch.ops.aten.digamma_.default,
    torch.ops.aten.digamma_,
).build()
_REGISTRY.register_unary("erfinv").targets(
    torch.erfinv,
    torch.ops.aten.erfinv.default,
    torch.ops.aten.erfinv,
    torch.ops.aten.erfinv_.default,
    torch.ops.aten.erfinv_,
).inplace(
    torch.ops.aten.erfinv_.default,
    torch.ops.aten.erfinv_,
).build()
_REGISTRY.register_unary("exp2").targets(
    torch.exp2,
    torch.ops.aten.exp2.default,
    torch.ops.aten.exp2,
    torch.ops.aten.exp2_.default,
    torch.ops.aten.exp2_,
).inplace(
    torch.ops.aten.exp2_.default,
    torch.ops.aten.exp2_,
).build()
_REGISTRY.register_unary("frac").targets(
    torch.frac,
    torch.ops.aten.frac.default,
    torch.ops.aten.frac,
    torch.ops.aten.frac_.default,
    torch.ops.aten.frac_,
).inplace(
    torch.ops.aten.frac_.default,
    torch.ops.aten.frac_,
).build()
_REGISTRY.register_unary("i0").targets(
    torch.i0,
    torch.ops.aten.i0.default,
    torch.ops.aten.i0,
    torch.ops.aten.i0_.default,
    torch.ops.aten.i0_,
).inplace(
    torch.ops.aten.i0_.default,
    torch.ops.aten.i0_,
).build()
_REGISTRY.register_unary("isfinite").targets(
    torch.isfinite,
    torch.ops.aten.isfinite.default,
    torch.ops.aten.isfinite,
).build()
_REGISTRY.register_unary("isinf").targets(
    torch.isinf,
    torch.ops.aten.isinf.default,
    torch.ops.aten.isinf,
).build()
_REGISTRY.register_unary("isnan").targets(
    torch.isnan,
    torch.ops.aten.isnan.default,
    torch.ops.aten.isnan,
).build()
_REGISTRY.register_unary("isneginf").targets(
    torch.isneginf,
    torch.ops.aten.isneginf.default,
    torch.ops.aten.isneginf,
).build()
_REGISTRY.register_unary("isposinf").targets(
    torch.isposinf,
    torch.ops.aten.isposinf.default,
    torch.ops.aten.isposinf,
).build()
_REGISTRY.register_unary("lgamma").targets(
    torch.lgamma,
    torch.ops.aten.lgamma.default,
    torch.ops.aten.lgamma,
    torch.ops.aten.lgamma_.default,
    torch.ops.aten.lgamma_,
).inplace(
    torch.ops.aten.lgamma_.default,
    torch.ops.aten.lgamma_,
).build()
_REGISTRY.register_unary("logit").targets(
    torch.logit,
    torch.ops.aten.logit.default,
    torch.ops.aten.logit,
    torch.ops.aten.logit_.default,
    torch.ops.aten.logit_,
).inplace(
    torch.ops.aten.logit_.default,
    torch.ops.aten.logit_,
).build()
_REGISTRY.register_unary("nan_to_num").targets(
    torch.nan_to_num,
    torch.ops.aten.nan_to_num.default,
    torch.ops.aten.nan_to_num,
    torch.ops.aten.nan_to_num_.default,
    torch.ops.aten.nan_to_num_,
).inplace(
    torch.ops.aten.nan_to_num_.default,
    torch.ops.aten.nan_to_num_,
).build()
_REGISTRY.register_unary("positive").targets(
    torch.positive,
    torch.ops.aten.positive.default,
    torch.ops.aten.positive,
).build()
_REGISTRY.register_unary("rad2deg").targets(
    torch.rad2deg,
    torch.ops.aten.rad2deg.default,
    torch.ops.aten.rad2deg,
    torch.ops.aten.rad2deg_.default,
    torch.ops.aten.rad2deg_,
).inplace(
    torch.ops.aten.rad2deg_.default,
    torch.ops.aten.rad2deg_,
).build()
_REGISTRY.register_unary("real").targets(
    torch.real,
    torch.ops.aten.real.default,
    torch.ops.aten.real,
).build()
_REGISTRY.register_unary("sgn").targets(
    torch.sgn,
    torch.ops.aten.sgn.default,
    torch.ops.aten.sgn,
    torch.ops.aten.sgn_.default,
    torch.ops.aten.sgn_,
).inplace(
    torch.ops.aten.sgn_.default,
    torch.ops.aten.sgn_,
).build()
_REGISTRY.register_unary("sinc").targets(
    torch.sinc,
    torch.ops.aten.sinc.default,
    torch.ops.aten.sinc,
    torch.ops.aten.sinc_.default,
    torch.ops.aten.sinc_,
).inplace(
    torch.ops.aten.sinc_.default,
    torch.ops.aten.sinc_,
).build()
_REGISTRY.register_unary("square").targets(
    torch.square,
    torch.ops.aten.square.default,
    torch.ops.aten.square,
    torch.ops.aten.square_.default,
    torch.ops.aten.square_,
).inplace(
    torch.ops.aten.square_.default,
    torch.ops.aten.square_,
).build()
_REGISTRY.register_op("argmax", kind="arg_reduction").targets(
    torch.argmax,
    torch.ops.aten.argmax.default,
    torch.ops.aten.argmax,
).build()
_REGISTRY.register_op("argmin", kind="arg_reduction").targets(
    torch.argmin,
    torch.ops.aten.argmin.default,
    torch.ops.aten.argmin,
).build()
_REGISTRY.register_op("sum", kind="reduction").targets(
    torch.ops.aten.sum.default,
    torch.ops.aten.sum.dim_IntList,
).build()
_REGISTRY.register_op("prod", kind="reduction").targets(
    torch.ops.aten.prod.default,
    torch.ops.aten.prod.dim_int,
).build()
_REGISTRY.register_op("mean", kind="reduction").targets(
    torch.mean,
    torch.ops.aten.mean.default,
    torch.ops.aten.mean,
    torch.ops.aten.mean.dim,
).build()
_REGISTRY.register_op("std", kind="reduction").targets(
    torch.std,
    torch.ops.aten.std.default,
    torch.ops.aten.std,
).build()
_REGISTRY.register_op("var", kind="reduction").targets(
    torch.var,
    torch.ops.aten.var.default,
    torch.ops.aten.var.dim,
).build()
_REGISTRY.register_op("norm", kind="reduction").targets(
    torch.norm,
    torch.ops.aten.norm.Scalar,
    torch.ops.aten.norm.ScalarOpt_dim,
).build()
_REGISTRY.register_op("any", kind="reduction").targets(
    torch.any,
    torch.ops.aten.any.default,
    torch.ops.aten.any.dim,
    torch.ops.aten.any.dims,
    torch.ops.aten.any,
).build()
_REGISTRY.register_op("all", kind="reduction").targets(
    torch.all,
    torch.ops.aten.all.default,
    torch.ops.aten.all,
).build()
_REGISTRY.register_op("amax", kind="reduction").targets(
    torch.amax,
    torch.ops.aten.amax.default,
    torch.ops.aten.amax,
).build()
_REGISTRY.register_op("amin", kind="reduction").targets(
    torch.amin,
    torch.ops.aten.amin.default,
    torch.ops.aten.amin,
).build()
_REGISTRY.register_op("softmax", kind="softmax").targets(
    torch.softmax,
    F.softmax,
    torch.ops.aten.softmax.int,
    torch.ops.aten.softmax,
).build()
_REGISTRY.register_op("log_softmax", kind="softmax").targets(
    torch.log_softmax,
    F.log_softmax,
    torch.ops.aten.log_softmax.int,
    torch.ops.aten.log_softmax,
).build()
_REGISTRY.register_op("_log_softmax", kind="softmax").targets(
    torch.ops.aten._log_softmax.default,
    torch.ops.aten._log_softmax,
).build()
_REGISTRY.register_op("cat", kind="concat").targets(
    torch.cat,
    torch.ops.aten.cat.default,
    torch.ops.aten.cat,
).build()
_REGISTRY.register_op("embedding", kind="embedding").targets(
    F.embedding,
    torch.ops.aten.embedding.default,
    torch.ops.aten.embedding,
).build()
_REGISTRY.register_op("diagonal", kind="diagonal").targets(
    torch.diagonal,
    torch.ops.aten.diagonal.default,
    torch.ops.aten.diagonal,
).build()
_REGISTRY.register_op("addmm", kind="addmm").targets(
    torch.addmm,
    torch.ops.aten.addmm.default,
    torch.ops.aten.addmm,
    torch.ops.aten.addmm_.default,
    torch.ops.aten.addmm_,
).inplace(
    torch.ops.aten.addmm_.default,
    torch.ops.aten.addmm_,
    arg_index=0,
).build()
_REGISTRY.register_op("addbmm", kind="addbmm").targets(
    torch.addbmm,
    torch.ops.aten.addbmm.default,
    torch.ops.aten.addbmm,
    torch.ops.aten.addbmm_.default,
    torch.ops.aten.addbmm_,
).inplace(
    torch.ops.aten.addbmm_.default,
    torch.ops.aten.addbmm_,
    arg_index=0,
).build()
_REGISTRY.register_op("addmv", kind="addmv").targets(
    torch.addmv,
    torch.ops.aten.addmv.default,
    torch.ops.aten.addmv,
    torch.ops.aten.addmv_.default,
    torch.ops.aten.addmv_,
).inplace(
    torch.ops.aten.addmv_.default,
    torch.ops.aten.addmv_,
    arg_index=0,
).build()
_REGISTRY.register_op("addr", kind="addr").targets(
    torch.addr,
    torch.ops.aten.addr.default,
    torch.ops.aten.addr,
    torch.ops.aten.addr_.default,
    torch.ops.aten.addr_,
).inplace(
    torch.ops.aten.addr_.default,
    torch.ops.aten.addr_,
    arg_index=0,
).build()
_REGISTRY.register_op("matmul", kind="matmul").targets(
    operator.matmul,
    torch.matmul,
    torch.ops.aten.mm,
    torch.ops.aten.mm.default,
    torch.ops.aten.matmul,
    torch.ops.aten.matmul.default,
).build()
_REGISTRY.register_op("bmm", kind="matmul").targets(
    torch.bmm,
    torch.ops.aten.bmm,
    torch.ops.aten.bmm.default,
).build()
_REGISTRY.register_op("conv2d", kind="conv2d").targets(
    torch.ops.aten.convolution.default,
    torch.ops.aten.convolution,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv2d,
).build()
_REGISTRY.register_op("conv1d", kind="conv1d").targets(
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv1d,
).build()
_REGISTRY.register_op("avg_pool1d", kind="pool1d").targets(
    F.avg_pool1d,
    torch.ops.aten.avg_pool1d.default,
    torch.ops.aten.avg_pool1d,
).build()
_REGISTRY.register_op("adaptive_avg_pool1d", kind="pool1d").targets(
    F.adaptive_avg_pool1d,
    torch.ops.aten.adaptive_avg_pool1d.default,
    torch.ops.aten.adaptive_avg_pool1d,
).build()
_REGISTRY.register_op("adaptive_avg_pool2d", kind="pool2d").targets(
    F.adaptive_avg_pool2d,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d,
    torch.ops.aten._adaptive_avg_pool2d.default,
    torch.ops.aten._adaptive_avg_pool2d,
).build()
_REGISTRY.register_op("max_pool1d", kind="pool1d").targets(
    F.max_pool1d,
    torch.ops.aten.max_pool1d.default,
    torch.ops.aten.max_pool1d,
).build()
_REGISTRY.register_op("avg_pool2d", kind="pool2d").targets(
    F.avg_pool2d,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.avg_pool2d,
).build()
_REGISTRY.register_op("max_pool2d", kind="pool2d").targets(
    F.max_pool2d,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.max_pool2d,
).build()
_REGISTRY.register_op("_native_batch_norm_legit_no_training", kind="batch_norm").targets(
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit_no_training.default,
).build()
_REGISTRY.register_op("_pdist_forward", kind="pdist").targets(
    torch.ops.aten._pdist_forward,
    torch.ops.aten._pdist_forward.default,
).build()

SUPPORTED_OPS = _REGISTRY.build()
