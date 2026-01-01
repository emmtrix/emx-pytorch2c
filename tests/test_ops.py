import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase, run_tests

from ref_backend.backend import ref_backend_backend
from ref_backend.cffi_bindings import RefBackendError


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
    "matmul": {
        "allowed_dtypes": (torch.float32,),
        "allow_noncontiguous": False,
        "requires_same_shape": False,
        "requires_contiguous": True,
        "skip_invalid_shape_tests": True,
        "sample_filter": _matmul_sample_filter,
    },
    "bmm": {
        "allowed_dtypes": (torch.float32,),
        "allow_noncontiguous": False,
        "requires_same_shape": False,
        "requires_contiguous": True,
        "skip_invalid_shape_tests": True,
        "sample_filter": _bmm_sample_filter,
    },
    "expand": {
        "allowed_dtypes": (torch.float32,),
        "allow_noncontiguous": False,
        "requires_same_shape": False,
        "requires_contiguous": True,
        "skip_invalid_shape_tests": True,
        "sample_filter": _expand_sample_filter,
    },
}

DEFAULT_CONSTRAINTS = {
    "allowed_dtypes": None,
    "allow_noncontiguous": True,
    "max_ndim": 8,
    "requires_same_shape": True,
    "requires_contiguous": False,
    "max_ndim_error": None,
    "shape_error": None,
    "skip_invalid_shape_tests": False,
    "sample_filter": None,
}

OPS_UNDER_TEST = [op for op in op_db if op.name in OP_TEST_CONFIG]


def _compile_op(op):
    def compiled_fn(*args: torch.Tensor) -> torch.Tensor:
        return op(*args)

    return torch.compile(compiled_fn, backend=ref_backend_backend)


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


def _get_op_arity(op, device, dtype, constraints):
    for sample in _iter_supported_samples(op, device, dtype, constraints):
        return 1 + len(sample.args)
    return None


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
            torch.testing.assert_close(result, op(*inputs))

    @ops(OPS_UNDER_TEST)
    def test_ref_backend_rejects_invalid_shapes(self, device, dtype, op):
        constraints = _constraints_for(op)
        if constraints["skip_invalid_shape_tests"]:
            pytest.skip("invalid-shape checks disabled by test constraints")
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_op(op)
        arity = _get_op_arity(op, device, dtype, constraints)
        if arity is None:
            pytest.skip("no supported sample inputs for this dtype")
        max_ndim = constraints["max_ndim"]
        if max_ndim is not None:
            too_many_dims = torch.randn((1,) * (max_ndim + 1), device=device, dtype=dtype)
            max_ndim_error = constraints["max_ndim_error"]
            if max_ndim_error is None:
                max_ndim_error = f"{op.name} supports at most {max_ndim} dimensions"
            with pytest.raises(RefBackendError, match=max_ndim_error):
                compiled(*([too_many_dims] * arity))

        if constraints["requires_same_shape"] and arity >= 2:
            a = torch.randn((2, 3), device=device, dtype=dtype)
            b = torch.randn((2, 4), device=device, dtype=dtype)
            shape_error = constraints["shape_error"]
            if shape_error is None:
                shape_error = (
                    f"{op.name} requires inputs and output to have identical shapes"
                )
            with pytest.raises(RefBackendError, match=shape_error):
                mismatched = [a] * (arity - 1) + [b]
                compiled(*mismatched)


instantiate_device_type_tests(TestElementwiseOpInfo, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
