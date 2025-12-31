import pytest
import torch

from ref_backend.cffi_bindings import run_sub


@pytest.mark.parametrize("shape", [(2, 3), (1,), (0, 4)])
def test_sub_op_matches_eager(shape):
    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)
    out = torch.empty_like(a)
    run_sub(a, b, out)
    torch.testing.assert_close(out, a - b)


def test_sub_op_handles_non_contiguous():
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    out = torch.empty_like(a)
    run_sub(a, b, out)
    torch.testing.assert_close(out, a - b)


def test_sub_op_rejects_more_than_8d():
    shape = (1, 1, 1, 1, 1, 1, 1, 1, 1)
    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)
    out = torch.empty_like(a)
    with pytest.raises(RuntimeError, match="sub supports at most 8 dimensions"):
        run_sub(a, b, out)
