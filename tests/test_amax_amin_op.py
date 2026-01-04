import pytest
import torch

from c_ref_backend.backend import c_ref_backend_backend
from c_ref_backend.cffi_bindings import RefBackendError, run_amax, run_amin


def _compile_op(op):
    def compiled_fn(*args):
        return op(*args)

    return torch.compile(compiled_fn, backend=c_ref_backend_backend)


def test_amax_matches_eager():
    input_tensor = torch.randn(2, 3, 4, dtype=torch.float32)
    compiled = _compile_op(torch.amax)
    result = compiled(input_tensor)
    torch.testing.assert_close(result, torch.amax(input_tensor))


def test_amin_matches_eager_noncontiguous():
    input_tensor = torch.randn(2, 3, 4, dtype=torch.float32).transpose(0, 2)
    out = torch.empty((), dtype=input_tensor.dtype)
    run_amin(input_tensor, out)
    torch.testing.assert_close(out, torch.amin(input_tensor))


def test_amax_propagates_nan():
    input_tensor = torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float32)
    out = torch.empty((), dtype=input_tensor.dtype)
    run_amax(input_tensor, out)
    assert torch.isnan(out)


def test_amax_rejects_empty_input():
    input_tensor = torch.empty((0,), dtype=torch.float32)
    out = torch.empty((), dtype=input_tensor.dtype)
    with pytest.raises(
        RefBackendError, match="input.numel\\(\\) > 0 when dim is not specified"
    ):
        run_amax(input_tensor, out)


def test_amin_rejects_empty_input():
    input_tensor = torch.empty((0,), dtype=torch.float32)
    out = torch.empty((), dtype=input_tensor.dtype)
    with pytest.raises(
        RefBackendError, match="input.numel\\(\\) > 0 when dim is not specified"
    ):
        run_amin(input_tensor, out)
