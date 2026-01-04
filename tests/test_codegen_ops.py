import pytest
import torch
from codegen_backend import codegen_generic_backend
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def _extract_tensors(sample):
    tensors = _flatten_tensors(sample.input)
    for arg in sample.args:
        tensors.extend(_flatten_tensors(arg))
    return tensors


def _update_sample(sample, updated_tensors):
    tensor_iter = iter(updated_tensors)
    if isinstance(sample.input, (list, tuple)):
        new_input = type(sample.input)(
            next(tensor_iter) if isinstance(item, torch.Tensor) else item
            for item in sample.input
        )
    else:
        new_input = next(tensor_iter)
    new_args = []
    for arg in sample.args:
        if isinstance(arg, torch.Tensor):
            new_args.append(next(tensor_iter))
        elif isinstance(arg, (list, tuple)):
            new_args.append(
                type(arg)(
                    next(tensor_iter) if isinstance(item, torch.Tensor) else item
                    for item in arg
                )
            )
        else:
            new_args.append(arg)
    return SampleInput(new_input, args=tuple(new_args), kwargs=sample.kwargs)


def _bmm_sample_filter(sample):
    tensors = _extract_tensors(sample)
    if len(tensors) != 2:
        return False
    a, b = tensors
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[0] == b.shape[0]
        and a.shape[2] == b.shape[1]
    )


def _all_same_shape(tensors):
    if not tensors:
        return True
    shape = tensors[0].shape
    return all(tensor.shape == shape for tensor in tensors[1:])


def _broadcast_shapes(*shapes):
    if not shapes:
        return ()
    max_len = max(len(shape) for shape in shapes)
    output_shape = []
    for dim in range(1, max_len + 1):
        sizes = [
            shape[-dim] if dim <= len(shape) else 1
            for shape in shapes
        ]
        max_size = max(sizes)
        if any(size not in (1, max_size) for size in sizes):
            raise ValueError("shapes are not broadcastable")
        output_shape.append(max_size)
    return tuple(reversed(output_shape))


def _broadcastable_sample_filter(sample):
    tensors = _extract_tensors(sample)
    if not tensors:
        return False
    try:
        _broadcast_shapes(*(tensor.shape for tensor in tensors))
    except ValueError:
        return False
    return True


def _concat_sample_filter(sample):
    if not isinstance(sample.input, (list, tuple)):
        return False
    tensors = _extract_tensors(sample)
    if not tensors:
        return False
    dim = sample.kwargs.get("dim", 0)
    if sample.args:
        dim = sample.args[0]
    if not isinstance(dim, int):
        return False
    rank = tensors[0].ndim
    if rank == 0:
        return False
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return False
    for tensor in tensors:
        if tensor.ndim != rank:
            return False
    base_shape = tensors[0].shape
    for tensor in tensors[1:]:
        for dim_index, size in enumerate(tensor.shape):
            if dim_index == dim:
                continue
            if size != base_shape[dim_index]:
                return False
    return True


def _sample_matches_constraints(sample, dtype, constraints):
    tensors = _extract_tensors(sample)
    if not tensors:
        return False
    max_ndim = constraints["max_ndim"]
    if max_ndim is not None and any(tensor.ndim > max_ndim for tensor in tensors):
        return False
    if not all(tensor.dtype is dtype for tensor in tensors):
        return False
    if constraints["requires_same_shape"] and not _all_same_shape(tensors):
        return False
    if constraints["requires_contiguous"] and any(
        not tensor.is_contiguous() for tensor in tensors
    ):
        return False
    sample_filter = constraints["sample_filter"]
    if sample_filter is not None and not sample_filter(sample):
        return False
    return True


def _iter_supported_samples(op, device, dtype, constraints):
    for sample in op.sample_inputs(device, dtype):
        if sample.kwargs and not constraints["allow_kwargs"]:
            continue
        if not constraints["allow_non_tensor_args"] and any(
            not isinstance(arg, torch.Tensor) for arg in sample.args
        ):
            continue
        if not _sample_matches_constraints(sample, dtype, constraints):
            continue
        yield sample

        if constraints["allow_noncontiguous"]:
            tensors = _extract_tensors(sample)
            if all(tensor.ndim >= 2 for tensor in tensors):
                transposed = [tensor.transpose(0, 1) for tensor in tensors]
                updated = _update_sample(sample, transposed)
                if _sample_matches_constraints(updated, dtype, constraints):
                    yield updated

            if all(tensor.ndim >= 1 and tensor.size(-1) > 1 for tensor in tensors):
                sliced = [tensor[..., ::2] for tensor in tensors]
                updated = _update_sample(sample, sliced)
                if _sample_matches_constraints(updated, dtype, constraints):
                    yield updated


CODEGEN_ATEN_OPS = [
    torch.ops.aten.abs.default,
    torch.ops.aten.acos.default,
    torch.ops.aten.acosh.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.all.default,
    torch.ops.aten.angle.default,
    torch.ops.aten.any.default,
    torch.ops.aten.asin.default,
    torch.ops.aten.asinh.default,
    torch.ops.aten.atan.default,
    torch.ops.aten.atan2.default,
    torch.ops.aten.atanh.default,
    torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.bitwise_left_shift.Tensor,
    torch.ops.aten.bitwise_not.default,
    torch.ops.aten.bitwise_or.Tensor,
    torch.ops.aten.bitwise_right_shift.Tensor,
    torch.ops.aten.bitwise_xor.Tensor,
    torch.ops.aten.bmm.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.ceil.default,
    torch.ops.aten.clamp_max.Tensor,
    torch.ops.aten.clamp_min.Tensor,
    torch.ops.aten.conj.default,
    torch.ops.aten.conj_physical.default,
    torch.ops.aten.copysign.Tensor,
    torch.ops.aten.cos.default,
    torch.ops.aten.cosh.default,
    torch.ops.aten.deg2rad.default,
    torch.ops.aten.digamma.default,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.erf.default,
    torch.ops.aten.erfc.default,
    torch.ops.aten.erfinv.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.exp2.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.floor.default,
    torch.ops.aten.floor_divide.default,
    torch.ops.aten.fmax.default,
    torch.ops.aten.fmin.default,
    torch.ops.aten.fmod.Tensor,
    torch.ops.aten.frac.default,
    torch.ops.aten.heaviside.default,
    torch.ops.aten.hypot.default,
    torch.ops.aten.i0.default,
    torch.ops.aten.isfinite.default,
    torch.ops.aten.isinf.default,
    torch.ops.aten.isnan.default,
    torch.ops.aten.isneginf.default,
    torch.ops.aten.isposinf.default,
    torch.ops.aten.ldexp.Tensor,
    torch.ops.aten.lt.Tensor,
    torch.ops.aten.le.Tensor,
    torch.ops.aten.gt.Tensor,
    torch.ops.aten.ge.Tensor,
    torch.ops.aten.eq.Tensor,
    torch.ops.aten.ne.Tensor,
    torch.ops.aten.logical_and.default,
    torch.ops.aten.logical_not.default,
    torch.ops.aten.logical_or.default,
    torch.ops.aten.logical_xor.default,
    torch.ops.aten.lgamma.default,
    torch.ops.aten.log.default,
    torch.ops.aten.log10.default,
    torch.ops.aten.log1p.default,
    torch.ops.aten.log2.default,
    torch.ops.aten.logaddexp.default,
    torch.ops.aten.logit.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.maximum.default,
    torch.ops.aten.minimum.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mean.default,
    torch.ops.aten.std.default,
    torch.ops.aten.nan_to_num.default,
    torch.ops.aten.neg.default,
    torch.ops.aten.nextafter.default,
    torch.ops.aten.positive.default,
    torch.ops.aten.pow.Tensor_Tensor,
    torch.ops.aten.prod.default,
    torch.ops.aten.rad2deg.default,
    torch.ops.aten.real.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.remainder.Tensor,
    torch.ops.aten.round.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sgn.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.sign.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.sinc.default,
    torch.ops.aten.sinh.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.square.default,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sum.default,
    torch.ops.aten.tan.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.select.int,
    torch.ops.aten.narrow.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.expand.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.trunc.default,
    torch.ops.aten.xlogy.Tensor,
    torch.ops.aten.where.self,
]
INPLACE_ATEN_OPS = [
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.abs_.default,
    torch.ops.aten.acos_.default,
    torch.ops.aten.acosh_.default,
    torch.ops.aten.arccosh_.default,
    torch.ops.aten.asin_.default,
    torch.ops.aten.asinh_.default,
    torch.ops.aten.atan_.default,
    torch.ops.aten.atan2_.default,
    torch.ops.aten.atanh_.default,
    torch.ops.aten.bitwise_and_.Tensor,
    torch.ops.aten.bitwise_left_shift_.Tensor,
    torch.ops.aten.bitwise_not_.default,
    torch.ops.aten.bitwise_or_.Tensor,
    torch.ops.aten.bitwise_right_shift_.Tensor,
    torch.ops.aten.bitwise_xor_.Tensor,
    torch.ops.aten.ceil_.default,
    torch.ops.aten.clamp_max_.Tensor,
    torch.ops.aten.clamp_min_.Tensor,
    torch.ops.aten.conj_physical_.default,
    torch.ops.aten.copysign_.Tensor,
    torch.ops.aten.cos_.default,
    torch.ops.aten.cosh_.default,
    torch.ops.aten.deg2rad_.default,
    torch.ops.aten.digamma_.default,
    torch.ops.aten.div_.Tensor,
    torch.ops.aten.erf_.default,
    torch.ops.aten.erfc_.default,
    torch.ops.aten.erfinv_.default,
    torch.ops.aten.exp_.default,
    torch.ops.aten.exp2_.default,
    torch.ops.aten.expm1_.default,
    torch.ops.aten.floor_.default,
    torch.ops.aten.floor_divide_.Tensor,
    torch.ops.aten.fmod_.Tensor,
    torch.ops.aten.frac_.default,
    torch.ops.aten.heaviside_.default,
    torch.ops.aten.hypot_.default,
    torch.ops.aten.i0_.default,
    torch.ops.aten.ldexp_.default,
    torch.ops.aten.lgamma_.default,
    torch.ops.aten.log_.default,
    torch.ops.aten.log10_.default,
    torch.ops.aten.log1p_.default,
    torch.ops.aten.log2_.default,
    torch.ops.aten.logit_.default,
    torch.ops.aten.mish_.default,
    torch.ops.aten.logical_and_.default,
    torch.ops.aten.logical_not_.default,
    torch.ops.aten.logical_or_.default,
    torch.ops.aten.logical_xor_.default,
    torch.ops.aten.mul_.Tensor,
    torch.ops.aten.nan_to_num_.default,
    torch.ops.aten.neg_.default,
    torch.ops.aten.nextafter_.default,
    torch.ops.aten.pow_.Tensor,
    torch.ops.aten.rad2deg_.default,
    torch.ops.aten.reciprocal_.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.remainder_.Tensor,
    torch.ops.aten.round_.default,
    torch.ops.aten.rsqrt_.default,
    torch.ops.aten.sgn_.default,
    torch.ops.aten.sigmoid_.default,
    torch.ops.aten.sign_.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.sin_.default,
    torch.ops.aten.sinc_.default,
    torch.ops.aten.silu_.default,
    torch.ops.aten.sinh_.default,
    torch.ops.aten.sqrt_.default,
    torch.ops.aten.square_.default,
    torch.ops.aten.sub_.Tensor,
    torch.ops.aten.tan_.default,
    torch.ops.aten.tanh_.default,
    torch.ops.aten.trunc_.default,
    torch.ops.aten.xlogy_.Tensor,
]


def _lookup_opinfo(aten_name, variant_test_name):
    matches = [
        op
        for op in op_db
        if op.aten_name == aten_name and op.variant_test_name == variant_test_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one OpInfo entry for {aten_name} "
            f"variant '{variant_test_name}'; found {len(matches)}"
        )
    return matches[0]


CODEGEN_OPINFO_OVERRIDES = {
    torch.ops.aten.div.Tensor: _lookup_opinfo("div", "no_rounding_mode"),
    torch.ops.aten.round.default: _lookup_opinfo("round", ""),
    torch.ops.aten.std.default: _lookup_opinfo("std", ""),
}


def _find_opinfo_for_overload(aten_overload):
    if aten_overload in CODEGEN_OPINFO_OVERRIDES:
        return CODEGEN_OPINFO_OVERRIDES[aten_overload]
    base_name = aten_overload._schema.name.split("::")[-1]
    packet = getattr(torch.ops.aten, base_name, None)
    if packet is None or aten_overload.overloadpacket is not packet:
        raise RuntimeError(
            f"Unsupported overload packet for {aten_overload}: expected aten.{base_name}"
        )
    candidates = [op for op in op_db if op.aten_name == base_name]
    if not candidates:
        candidates = [op for op in op_db if op.name == base_name]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one OpInfo entry for {aten_overload}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


CODEGEN_OPS_UNDER_TEST = [
    (aten_overload, _find_opinfo_for_overload(aten_overload))
    for aten_overload in CODEGEN_ATEN_OPS
]
CODEGEN_OPINFO_LIST = [opinfo for _, opinfo in CODEGEN_OPS_UNDER_TEST]
CODEGEN_OP_TEST_CONFIG = {
    torch.ops.aten.add.Tensor: {
        "requires_same_shape": False,
        "sample_filter": _broadcastable_sample_filter,
    },
    torch.ops.aten.bitwise_left_shift.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.bitwise_left_shift_.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.bitwise_right_shift.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.bitwise_right_shift_.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.logical_and.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.logical_not.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.logical_or.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.logical_xor.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.where.self: {
        "requires_same_shape": False,
        "sample_filter": _broadcastable_sample_filter,
    },
    torch.ops.aten.mish_.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.matmul.default: {
        "allow_noncontiguous": True,
        "requires_same_shape": False,
    },
    torch.ops.aten.bmm.default: {
        "allow_noncontiguous": True,
        "requires_same_shape": False,
        "sample_filter": _bmm_sample_filter,
    },
    torch.ops.aten.std.default: {
        "allow_non_tensor_args": True,
    },
    torch.ops.aten.transpose.int: {
        "allow_non_tensor_args": True,
    },
    torch.ops.aten.cat.default: {
        "allow_kwargs": True,
        "expand_input_list": True,
        "requires_same_shape": False,
        "sample_filter": _concat_sample_filter,
    },
}
DEFAULT_CONSTRAINTS = {
    "allowed_dtypes": (torch.float32, torch.int8, torch.int32, torch.bool),
    "allow_noncontiguous": True,
    "allow_non_tensor_args": False,
    "allow_kwargs": False,
    "expand_input_list": False,
    "max_ndim": 8,
    "requires_same_shape": True,
    "requires_contiguous": False,
    "sample_filter": None,
}


def _constraints_for_codegen(aten_overload):
    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints.update(CODEGEN_OP_TEST_CONFIG.get(aten_overload, {}))
    return constraints


def _sample_to_inputs(sample, constraints):
    inputs = []
    if constraints["expand_input_list"] and isinstance(sample.input, (list, tuple)):
        inputs.extend(sample.input)
    else:
        inputs.append(sample.input)
    inputs.extend(sample.args)
    kwargs = sample.kwargs if constraints["allow_kwargs"] else {}
    return tuple(inputs), kwargs


def _compile_codegen_op(aten_overload):
    if aten_overload is torch.ops.aten.cat.default:
        def compiled_fn(*args: torch.Tensor, **kwargs) -> torch.Tensor:
            return aten_overload(list(args), **kwargs)
    else:
        def compiled_fn(*args: torch.Tensor, **kwargs) -> torch.Tensor:
            return aten_overload(*args, **kwargs)

    return torch.compile(compiled_fn, backend=codegen_generic_backend)


def _compile_codegen_inplace_op(aten_overload):
    def compiled_fn(lhs: torch.Tensor, rhs: torch.Tensor | None = None) -> torch.Tensor:
        if rhs is None:
            return aten_overload(lhs)
        return aten_overload(lhs, rhs)

    return torch.compile(compiled_fn, backend=codegen_generic_backend)


def _sanitize_inplace_inputs(
    aten_overload: torch._ops.OpOverload, lhs: torch.Tensor, rhs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    name = aten_overload._schema.name.split("::")[-1]
    unit_range_ops = {"acos_", "asin_", "atanh_", "erfinv_"}
    ge1_ops = {"acosh_", "arccosh_"}
    positive_ops = {"digamma_", "lgamma_", "log_", "log10_", "log2_", "rsqrt_", "sqrt_"}
    log1p_ops = {"log1p_"}
    logit_ops = {"logit_"}
    reciprocal_ops = {"reciprocal_"}
    pow_ops = {"pow_"}
    rhs_nonzero_ops = {"div_", "floor_divide_", "fmod_", "remainder_"}
    xlogy_ops = {"xlogy_"}
    ldexp_ops = {"ldexp_"}

    if name in unit_range_ops:
        lhs = lhs.tanh()
    if name in ge1_ops:
        lhs = lhs.abs() + 1
    if name in positive_ops:
        lhs = lhs.abs() + 0.1
    if name in log1p_ops:
        lhs = lhs.abs()
    if name in logit_ops:
        lhs = lhs.sigmoid().clamp(1e-4, 1 - 1e-4)
    if name in reciprocal_ops:
        lhs = lhs.sign() * (lhs.abs() + 0.1)
    if name in pow_ops:
        lhs = lhs.abs() + 0.1
    if name in rhs_nonzero_ops:
        rhs = rhs.sign() * (rhs.abs() + 0.1)
    if name in xlogy_ops:
        rhs = rhs.abs() + 0.1
    if name in ldexp_ops:
        rhs = rhs.round()

    return lhs, rhs


def _reference_for_dtype(
    aten_overload: torch._ops.OpOverload,
    inputs: tuple[object, ...],
    kwargs: dict[str, object],
    dtype: torch.dtype,
) -> torch.Tensor:
    if dtype not in (torch.int8, torch.int32, torch.bool):
        return aten_overload(*inputs, **kwargs)
    try:
        expected = aten_overload(*inputs, **kwargs)
    except Exception:
        float_inputs = tuple(
            arg.to(torch.float32) if isinstance(arg, torch.Tensor) else arg
            for arg in inputs
        )
        expected = aten_overload(*float_inputs, **kwargs)
    return expected


class TestCodegenOpInfo(TestCase):
    @ops(CODEGEN_OPINFO_LIST)
    def test_codegen_backend_matches_eager(self, device, dtype, op):
        aten_overload = next(
            (
                overload
                for overload, opinfo in CODEGEN_OPS_UNDER_TEST
                if opinfo is op
            ),
            None,
        )
        if aten_overload is None:
            raise RuntimeError(f"Missing overload mapping for OpInfo {op.name}")
        constraints = _constraints_for_codegen(aten_overload)
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_codegen_op(aten_overload)
        for sample in _iter_supported_samples(op, device, dtype, constraints):
            inputs, kwargs = _sample_to_inputs(sample, constraints)
            try:
                expected = _reference_for_dtype(aten_overload, inputs, kwargs, dtype)
            except Exception:
                continue
            result = compiled(*inputs, **kwargs)
            if result.dtype is not expected.dtype:
                expected = expected.to(result.dtype)
            torch.testing.assert_close(
                result, expected, equal_nan=dtype in (torch.int8, torch.int32)
            )


class TestCodegenAliasedOps(TestCase):
    def test_codegen_arccosh_matches_eager(self):
        aten_overload = torch.ops.aten.arccosh.default
        compiled = _compile_codegen_op(aten_overload)
        for dtype in (torch.float32,):
            inputs = (torch.rand(2, 3, dtype=dtype) + 1.0,)
            expected = aten_overload(*inputs)
            result = compiled(*inputs)
            torch.testing.assert_close(result, expected)


class TestCodegenInplaceOps(TestCase):
    def test_codegen_backend_inplace_ops(self):
        device = torch.device("cpu")
        sample_shapes = (
            ((2, 3), (2, 3)),
        )

        for aten_overload in INPLACE_ATEN_OPS:
            constraints = _constraints_for_codegen(aten_overload)
            allowed_dtypes = constraints["allowed_dtypes"]
            dtype = (
                torch.float32
                if torch.float32 in allowed_dtypes
                else allowed_dtypes[0]
            )
            required_args = [
                arg
                for arg in aten_overload._schema.arguments
                if not arg.kwarg_only and not arg.has_default_value()
            ]
            requires_rhs = len(required_args) > 1
            compiled = _compile_codegen_inplace_op(aten_overload)
            for lhs_shape, rhs_shape in sample_shapes:
                op_name = aten_overload._schema.name.split("::")[-1]
                if dtype in (torch.int8, torch.int32):
                    if op_name in {"bitwise_left_shift_", "bitwise_right_shift_"}:
                        low, high = 0, 3
                    else:
                        low, high = -5, 5
                    lhs = torch.randint(
                        low, high, lhs_shape, device=device, dtype=dtype
                    )
                    rhs = torch.randint(
                        low, high, rhs_shape, device=device, dtype=dtype
                    )
                elif dtype is torch.bool:
                    lhs = torch.randint(
                        0, 2, lhs_shape, device=device, dtype=dtype
                    )
                    rhs = torch.randint(
                        0, 2, rhs_shape, device=device, dtype=dtype
                    )
                else:
                    lhs = torch.randn(lhs_shape, device=device, dtype=dtype)
                    rhs = torch.randn(rhs_shape, device=device, dtype=dtype)
                lhs, rhs = _sanitize_inplace_inputs(aten_overload, lhs, rhs)

                expected = lhs.clone()
                if requires_rhs:
                    args = (expected, rhs)
                else:
                    args = (expected,)
                try:
                    expected_result = aten_overload(*args)
                except Exception:
                    continue

                compiled_lhs = lhs.clone()
                if requires_rhs:
                    compiled_args = (compiled_lhs, rhs)
                else:
                    compiled_args = (compiled_lhs,)
                compiled_result = compiled(*compiled_args)

                torch.testing.assert_close(compiled_result, expected_result)
                torch.testing.assert_close(compiled_lhs, expected)


class TestCodegenReductionOps(TestCase):
    def test_codegen_sum_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        for tensor in cases:
            compiled = _compile_codegen_op(torch.ops.aten.sum.default)
            result = compiled(tensor)
            expected = torch.ops.aten.sum.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected)

    def test_codegen_prod_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        for tensor in cases:
            compiled = _compile_codegen_op(torch.ops.aten.prod.default)
            result = compiled(tensor)
            expected = torch.ops.aten.prod.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected)

    def test_codegen_mean_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        for tensor in cases:
            compiled = _compile_codegen_op(torch.ops.aten.mean.default)
            result = compiled(tensor)
            expected = torch.ops.aten.mean.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected, equal_nan=True)

    def test_codegen_std_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        compiled = _compile_codegen_op(torch.ops.aten.std.default)
        for tensor in cases:
            result = compiled(tensor)
            expected = torch.ops.aten.std.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected, equal_nan=True)

        for tensor in cases:
            result = compiled(tensor, False)
            expected = torch.ops.aten.std.default(tensor, False)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected, equal_nan=True)


instantiate_device_type_tests(TestCodegenOpInfo, globals(), only_for="cpu")
