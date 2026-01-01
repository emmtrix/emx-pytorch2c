import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase, run_tests

from ref_backend.backend import ref_backend_backend
from ref_backend.cffi_bindings import RefBackendError


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
}

DEFAULT_CONSTRAINTS = {
    "allowed_dtypes": None,
    "allow_noncontiguous": True,
    "max_ndim": 8,
    "requires_same_shape": True,
    "max_ndim_error": None,
    "shape_error": None,
    "skip_invalid_shape_tests": False,
}

OPS_UNDER_TEST = [op for op in op_db if op.name in OP_TEST_CONFIG]


def _compile_op(op):
    def compiled_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return op(a, b)

    return torch.compile(compiled_fn, backend=ref_backend_backend)


def _constraints_for(op):
    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints.update(OP_TEST_CONFIG[op.name])
    return constraints


def _iter_supported_samples(op, device, dtype, constraints):
    for sample in op.sample_inputs(device, dtype):
        if sample.kwargs:
            continue
        if len(sample.args) != 1:
            continue
        other = sample.args[0]
        if not isinstance(other, torch.Tensor):
            continue
        if sample.input.shape != other.shape:
            continue
        if sample.input.dtype is not dtype or other.dtype is not dtype:
            continue
        yield sample

        if constraints["allow_noncontiguous"]:
            if sample.input.ndim >= 2:
                a_t = sample.input.transpose(0, 1)
                b_t = other.transpose(0, 1)
                yield SampleInput(a_t, args=(b_t,))

            if sample.input.ndim >= 1 and sample.input.size(-1) > 1:
                a_s = sample.input[..., ::2]
                b_s = other[..., ::2]
                if a_s.shape == b_s.shape:
                    yield SampleInput(a_s, args=(b_s,))


class TestBinaryElementwiseOpInfo(TestCase):
    @ops(OPS_UNDER_TEST)
    def test_ref_backend_matches_eager(self, device, dtype, op):
        constraints = _constraints_for(op)
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_op(op)
        for sample in _iter_supported_samples(op, device, dtype, constraints):
            a = sample.input
            b = sample.args[0]
            result = compiled(a, b)
            torch.testing.assert_close(result, op(a, b))

    @ops(OPS_UNDER_TEST)
    def test_ref_backend_rejects_invalid_shapes(self, device, dtype, op):
        constraints = _constraints_for(op)
        if constraints["skip_invalid_shape_tests"]:
            pytest.skip("invalid-shape checks disabled by test constraints")
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_op(op)
        max_ndim = constraints["max_ndim"]
        if max_ndim is not None:
            too_many_dims = torch.randn((1,) * (max_ndim + 1), device=device, dtype=dtype)
            max_ndim_error = constraints["max_ndim_error"]
            if max_ndim_error is None:
                max_ndim_error = f"{op.name} supports at most {max_ndim} dimensions"
            with pytest.raises(RefBackendError, match=max_ndim_error):
                compiled(too_many_dims, too_many_dims)

        if constraints["requires_same_shape"]:
            a = torch.randn((2, 3), device=device, dtype=dtype)
            b = torch.randn((2, 4), device=device, dtype=dtype)
            shape_error = constraints["shape_error"]
            if shape_error is None:
                shape_error = (
                    f"{op.name} requires inputs and output to have identical shapes"
                )
            with pytest.raises(RefBackendError, match=shape_error):
                compiled(a, b)


instantiate_device_type_tests(TestBinaryElementwiseOpInfo, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
