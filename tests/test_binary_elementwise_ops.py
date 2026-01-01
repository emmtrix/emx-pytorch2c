import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase, run_tests

from ref_backend.cffi_bindings import run_add, run_mul, run_sub


ADD_SUB_OPS = [op for op in op_db if op.name in ("add", "sub", "mul")]


def _run_ref_op(op_name: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if op_name == "add":
        run_add(a, b, out)
        return
    if op_name == "sub":
        run_sub(a, b, out)
        return
    if op_name == "mul":
        run_mul(a, b, out)
        return
    raise ValueError(f"Unsupported op name: {op_name}")


def _iter_supported_samples(op, device, dtype):
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

        if sample.input.ndim >= 2:
            a_t = sample.input.transpose(0, 1)
            b_t = other.transpose(0, 1)
            yield SampleInput(a_t, args=(b_t,))

        if sample.input.ndim >= 1 and sample.input.size(-1) > 1:
            a_s = sample.input[..., ::2]
            b_s = other[..., ::2]
            if a_s.shape == b_s.shape:
                yield SampleInput(a_s, args=(b_s,))


class TestAddSubOpInfo(TestCase):
    @ops(ADD_SUB_OPS, allowed_dtypes=(torch.float32,))
    def test_ref_backend_matches_eager(self, device, dtype, op):
        for sample in _iter_supported_samples(op, device, dtype):
            a = sample.input
            b = sample.args[0]
            out = torch.empty_like(a)
            _run_ref_op(op.name, a, b, out)
            expected = op(a, b)
            torch.testing.assert_close(out, expected)

    @ops(ADD_SUB_OPS, allowed_dtypes=(torch.float32,))
    def test_ref_backend_rejects_invalid_shapes(self, device, dtype, op):
        too_many_dims = torch.randn((1,) * 9, device=device, dtype=dtype)
        out = torch.empty_like(too_many_dims)
        with pytest.raises(RuntimeError, match=f"{op.name} supports at most 8 dimensions"):
            _run_ref_op(op.name, too_many_dims, too_many_dims, out)

        a = torch.randn((2, 3), device=device, dtype=dtype)
        b = torch.randn((2, 4), device=device, dtype=dtype)
        out = torch.empty_like(a)
        with pytest.raises(
            RuntimeError,
            match=f"{op.name} requires inputs and output to have identical shapes",
        ):
            _run_ref_op(op.name, a, b, out)


instantiate_device_type_tests(TestAddSubOpInfo, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
