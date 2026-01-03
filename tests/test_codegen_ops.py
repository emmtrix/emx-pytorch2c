import pytest
import torch
from codegen_backend import codegen_generic_backend
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase


def _extract_tensors(sample):
    tensors = [sample.input]
    tensors.extend(arg for arg in sample.args if isinstance(arg, torch.Tensor))
    return tensors


def _update_sample(sample, updated_tensors):
    tensor_iter = iter(updated_tensors)
    new_input = next(tensor_iter)
    new_args = []
    for arg in sample.args:
        if isinstance(arg, torch.Tensor):
            new_args.append(next(tensor_iter))
        else:
            new_args.append(arg)
    return SampleInput(new_input, args=tuple(new_args))


def _matmul_sample_filter(sample):
    tensors = _extract_tensors(sample)
    if len(tensors) != 2:
        return False
    a, b = tensors
    if a.ndim != b.ndim or a.ndim not in (2, 3):
        return False
    if a.ndim == 2:
        return a.shape[1] == b.shape[0]
    return a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]


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
        if sample.kwargs:
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


CODEGEN_OP_NAMES = {
    "abs",
    "acos",
    "acosh",
    "add",
    "add_",
    "angle",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bmm",
    "cbrt",
    "ceil",
    "clamp_max",
    "clamp_min",
    "conj",
    "conj_physical",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "div",
    "div_",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frac",
    "heaviside",
    "hypot",
    "i0",
    "ldexp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logit",
    "matmul",
    "maximum",
    "minimum",
    "mul",
    "mul_",
    "nan_to_num",
    "neg",
    "nextafter",
    "positive",
    "pow",
    "rad2deg",
    "real",
    "reciprocal",
    "relu",
    "remainder",
    "round",
    "rsqrt",
    "sgn",
    "sigmoid",
    "sign",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "sub",
    "sub_",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    "xlogy",
}
CODEGEN_INPLACE_OPS = (
    "add_",
    "div_",
    "mul_",
    "sub_",
)
INPLACE_ATEN_TARGETS = {
    "add_": torch.ops.aten.add_.Tensor,
    "div_": torch.ops.aten.div_.Tensor,
    "mul_": torch.ops.aten.mul_.Tensor,
    "sub_": torch.ops.aten.sub_.Tensor,
}
CODEGEN_OPS_UNDER_TEST = [op for op in op_db if op.name in CODEGEN_OP_NAMES]
CODEGEN_OP_TEST_CONFIG = {
    "add": {
        "requires_same_shape": False,
        "sample_filter": _broadcastable_sample_filter,
    },
    "matmul": {
        "allow_noncontiguous": True,
        "requires_same_shape": False,
        "sample_filter": _matmul_sample_filter,
    },
    "bmm": {
        "allow_noncontiguous": True,
        "requires_same_shape": False,
        "sample_filter": _bmm_sample_filter,
    },
    "transpose": {
        "allow_non_tensor_args": True,
    },
}
DEFAULT_CONSTRAINTS = {
    "allowed_dtypes": (torch.float32,),
    "allow_noncontiguous": True,
    "allow_non_tensor_args": False,
    "max_ndim": 8,
    "requires_same_shape": True,
    "requires_contiguous": False,
    "sample_filter": None,
}


def _constraints_for_codegen(op):
    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints.update(CODEGEN_OP_TEST_CONFIG.get(op.name, {}))
    return constraints


def _compile_codegen_op(op):
    def compiled_fn(*args: torch.Tensor) -> torch.Tensor:
        return op(*args)

    return torch.compile(compiled_fn, backend=codegen_generic_backend)


def _compile_codegen_inplace_op(op_name):
    target = INPLACE_ATEN_TARGETS[op_name]

    def compiled_fn(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return target(lhs, rhs)

    return torch.compile(compiled_fn, backend=codegen_generic_backend)


class TestCodegenOpInfo(TestCase):
    @ops(CODEGEN_OPS_UNDER_TEST)
    def test_codegen_backend_matches_eager(self, device, dtype, op):
        constraints = _constraints_for_codegen(op)
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_codegen_op(op)
        for sample in _iter_supported_samples(op, device, dtype, constraints):
            inputs = (sample.input, *sample.args)
            result = compiled(*inputs)
            torch.testing.assert_close(result, op(*inputs))


class TestCodegenInplaceOps(TestCase):
    def test_codegen_backend_inplace_ops(self):
        device = torch.device("cpu")
        dtype = torch.float32
        sample_shapes = (
            ((2, 3), (2, 3)),
        )

        for op_name in CODEGEN_INPLACE_OPS:
            compiled = _compile_codegen_inplace_op(op_name)
            for lhs_shape, rhs_shape in sample_shapes:
                lhs = torch.randn(lhs_shape, device=device, dtype=dtype)
                rhs = torch.randn(rhs_shape, device=device, dtype=dtype)

                expected = lhs.clone()
                expected_result = getattr(expected, op_name)(rhs)

                compiled_lhs = lhs.clone()
                compiled_result = compiled(compiled_lhs, rhs)

                torch.testing.assert_close(compiled_result, expected_result)
                torch.testing.assert_close(compiled_lhs, expected)


instantiate_device_type_tests(TestCodegenOpInfo, globals(), only_for="cpu")
