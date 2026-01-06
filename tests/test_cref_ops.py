import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase, run_tests

from c_ref_backend.backend import c_ref_backend_backend
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


def _expand_sample_filter(sample):
    if len(sample.args) != 1 or not isinstance(sample.args[0], tuple):
        return False
    shape = sample.args[0]
    out_rank = len(shape)
    in_rank = sample.input.ndim
    if out_rank < in_rank:
        return False
    leading = out_rank - in_rank
    for idx, dim in enumerate(shape):
        if dim == -1 and idx < leading:
            return False
    return True


OP_TEST_CONFIG = {
    "add": {
        "allowed_dtypes": (torch.float32,),
    },
    "sub": {
        "allowed_dtypes": (torch.float32,),
    },
    "mul": {
        "allowed_dtypes": (torch.float32,),
    },
    "div": {
        "allowed_dtypes": (torch.float32,),
    },
    "maximum": {
        "allowed_dtypes": (torch.float32,),
    },
    "minimum": {
        "allowed_dtypes": (torch.float32,),
    },
    "atan2": {
        "allowed_dtypes": (torch.float32,),
    },
    "pow": {
        "allowed_dtypes": (torch.float32,),
    },
    "remainder": {
        "allowed_dtypes": (torch.float32,),
    },
    "fmod": {
        "allowed_dtypes": (torch.float32,),
    },
    "floor_divide": {
        "allowed_dtypes": (torch.float32,),
    },
    "fmax": {
        "allowed_dtypes": (torch.float32,),
    },
    "fmin": {
        "allowed_dtypes": (torch.float32,),
    },
    "copysign": {
        "allowed_dtypes": (torch.float32,),
    },
    "hypot": {
        "allowed_dtypes": (torch.float32,),
    },
    "logaddexp": {
        "allowed_dtypes": (torch.float32,),
    },
    "nextafter": {
        "allowed_dtypes": (torch.float32,),
    },
    "xlogy": {
        "allowed_dtypes": (torch.float32,),
    },
    "heaviside": {
        "allowed_dtypes": (torch.float32,),
    },
    "ldexp": {
        "allowed_dtypes": (torch.float32,),
    },
    "clamp_min": {
        "allowed_dtypes": (torch.float32,),
    },
    "clamp_max": {
        "allowed_dtypes": (torch.float32,),
    },
    "neg": {
        "allowed_dtypes": (torch.float32,),
    },
    "exp": {
        "allowed_dtypes": (torch.float32,),
    },
    "abs": {
        "allowed_dtypes": (torch.float32,),
    },
    "sqrt": {
        "allowed_dtypes": (torch.float32,),
    },
    "log": {
        "allowed_dtypes": (torch.float32,),
    },
    "sin": {
        "allowed_dtypes": (torch.float32,),
    },
    "cos": {
        "allowed_dtypes": (torch.float32,),
    },
    "acos": {
        "allowed_dtypes": (torch.float32,),
    },
    "acosh": {
        "allowed_dtypes": (torch.float32,),
    },
    "asin": {
        "allowed_dtypes": (torch.float32,),
    },
    "asinh": {
        "allowed_dtypes": (torch.float32,),
    },
    "atan": {
        "allowed_dtypes": (torch.float32,),
    },
    "atanh": {
        "allowed_dtypes": (torch.float32,),
    },
    "cosh": {
        "allowed_dtypes": (torch.float32,),
    },
    "sinh": {
        "allowed_dtypes": (torch.float32,),
    },
    "tan": {
        "allowed_dtypes": (torch.float32,),
    },
    "erf": {
        "allowed_dtypes": (torch.float32,),
    },
    "erfc": {
        "allowed_dtypes": (torch.float32,),
    },
    "expm1": {
        "allowed_dtypes": (torch.float32,),
    },
    "log1p": {
        "allowed_dtypes": (torch.float32,),
    },
    "log2": {
        "allowed_dtypes": (torch.float32,),
    },
    "log10": {
        "allowed_dtypes": (torch.float32,),
    },
    "rsqrt": {
        "allowed_dtypes": (torch.float32,),
    },
    "sigmoid": {
        "allowed_dtypes": (torch.float32,),
    },
    "silu": {
        "allowed_dtypes": (torch.float32,),
    },
    "sign": {
        "allowed_dtypes": (torch.float32,),
    },
    "round": {
        "allowed_dtypes": (torch.float32,),
    },
    "trunc": {
        "allowed_dtypes": (torch.float32,),
    },
    "tanh": {
        "allowed_dtypes": (torch.float32,),
    },
    "floor": {
        "allowed_dtypes": (torch.float32,),
    },
    "ceil": {
        "allowed_dtypes": (torch.float32,),
    },
    "cbrt": {
        "allowed_dtypes": (torch.float32,),
    },
    "reciprocal": {
        "allowed_dtypes": (torch.float32,),
    },
    "relu": {
        "allowed_dtypes": (torch.float32,),
    },
    "angle": {
        "allowed_dtypes": (torch.float32,),
    },
    "conj": {
        "allowed_dtypes": (torch.float32,),
    },
    "conj_physical": {
        "allowed_dtypes": (torch.float32,),
    },
    "deg2rad": {
        "allowed_dtypes": (torch.float32,),
    },
    "digamma": {
        "allowed_dtypes": (torch.float32,),
        "rtol": 3e-5,
        "atol": 0.0,
    },
    "erfinv": {
        "allowed_dtypes": (torch.float32,),
    },
    "exp2": {
        "allowed_dtypes": (torch.float32,),
    },
    "frac": {
        "allowed_dtypes": (torch.float32,),
    },
    "i0": {
        "allowed_dtypes": (torch.float32,),
    },
    "lgamma": {
        "allowed_dtypes": (torch.float32,),
    },
    "logit": {
        "allowed_dtypes": (torch.float32,),
    },
    "nan_to_num": {
        "allowed_dtypes": (torch.float32,),
    },
    "positive": {
        "allowed_dtypes": (torch.float32,),
    },
    "rad2deg": {
        "allowed_dtypes": (torch.float32,),
    },
    "real": {
        "allowed_dtypes": (torch.float32,),
    },
    "sgn": {
        "allowed_dtypes": (torch.float32,),
    },
    "sinc": {
        "allowed_dtypes": (torch.float32,),
    },
    "square": {
        "allowed_dtypes": (torch.float32,),
    },
    "matmul": {
        "allowed_dtypes": (torch.float32,),
        "allow_noncontiguous": False,
        "requires_same_shape": False,
        "requires_contiguous": True,
        "sample_filter": _matmul_sample_filter,
    },
    "bmm": {
        "allowed_dtypes": (torch.float32,),
        "allow_noncontiguous": False,
        "requires_same_shape": False,
        "requires_contiguous": True,
        "sample_filter": _bmm_sample_filter,
    },
    "expand": {
        "allowed_dtypes": (torch.float32,),
        "allow_noncontiguous": False,
        "requires_same_shape": False,
        "requires_contiguous": True,
        "sample_filter": _expand_sample_filter,
    },
}

DEFAULT_CONSTRAINTS = {
    "allowed_dtypes": None,
    "allow_noncontiguous": True,
    "max_ndim": 8,
    "requires_same_shape": True,
    "requires_contiguous": False,
    "sample_filter": None,
    "rtol": None,
    "atol": None,
}

OPS_UNDER_TEST = [op for op in op_db if op.name in OP_TEST_CONFIG]


def _compile_op(op):
    def compiled_fn(*args: torch.Tensor) -> torch.Tensor:
        return op(*args)

    return torch.compile(compiled_fn, backend=c_ref_backend_backend)


def _constraints_for(op):
    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints.update(OP_TEST_CONFIG[op.name])
    return constraints


def _all_same_shape(tensors):
    if not tensors:
        return True
    shape = tensors[0].shape
    return all(tensor.shape == shape for tensor in tensors[1:])


def _iter_supported_samples(op, device, dtype, constraints):
    for sample in op.sample_inputs(device, dtype):
        if sample.kwargs:
            continue
        tensors = _extract_tensors(sample)
        if not tensors:
            continue
        max_ndim = constraints["max_ndim"]
        if max_ndim is not None and any(tensor.ndim > max_ndim for tensor in tensors):
            continue
        if constraints["requires_same_shape"] and not _all_same_shape(tensors):
            continue
        if not all(tensor.dtype is dtype for tensor in tensors):
            continue
        if constraints["requires_contiguous"] and any(
            not tensor.is_contiguous() for tensor in tensors
        ):
            continue
        sample_filter = constraints["sample_filter"]
        if sample_filter is not None and not sample_filter(sample):
            continue
        yield sample

        if constraints["allow_noncontiguous"]:
            if all(tensor.ndim >= 2 for tensor in tensors):
                transposed = [tensor.transpose(0, 1) for tensor in tensors]
                if not constraints["requires_same_shape"] or _all_same_shape(
                    transposed
                ):
                    yield _update_sample(sample, transposed)

            if all(tensor.ndim >= 1 and tensor.size(-1) > 1 for tensor in tensors):
                sliced = [tensor[..., ::2] for tensor in tensors]
                if not constraints["requires_same_shape"] or _all_same_shape(sliced):
                    yield _update_sample(sample, sliced)


class TestElementwiseOpInfo(TestCase):
    @ops(OPS_UNDER_TEST)
    def test_ref_backend_matches_eager(self, device, dtype, op):
        constraints = _constraints_for(op)
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_op(op)
        for sample in _iter_supported_samples(op, device, dtype, constraints):
            inputs = (sample.input, *sample.args)
            result = compiled(*inputs)
            compare_kwargs = {}
            if constraints["rtol"] is not None or constraints["atol"] is not None:
                compare_kwargs["rtol"] = constraints["rtol"] or 0.0
                compare_kwargs["atol"] = constraints["atol"] or 0.0
            torch.testing.assert_close(result, op(*inputs), **compare_kwargs)


instantiate_device_type_tests(TestElementwiseOpInfo, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
