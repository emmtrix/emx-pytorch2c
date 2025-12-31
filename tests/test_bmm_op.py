import pytest
import torch

from ref_backend.cffi_bindings import RefBackendError, run_bmm


@pytest.mark.parametrize("batch,m,k,n", [(2, 3, 4, 5), (1, 2, 2, 1), (0, 3, 2, 4)])
def test_bmm_op_matches_eager(batch, m, k, n):
    a = torch.randn(batch, m, k, dtype=torch.float32)
    b = torch.randn(batch, k, n, dtype=torch.float32)
    out = torch.empty(batch, m, n, dtype=torch.float32)
    run_bmm(a, b, out)
    torch.testing.assert_close(out, torch.bmm(a, b))


def test_bmm_op_rejects_non_contiguous():
    a = torch.randn(2, 4, 3, dtype=torch.float32).transpose(1, 2)
    b = torch.randn(2, 5, 4, dtype=torch.float32).transpose(1, 2)
    out = torch.empty(2, 3, 5, dtype=torch.float32)
    with pytest.raises(RefBackendError, match="contiguous"):
        run_bmm(a, b, out)


def test_bmm_op_rejects_mismatched_batch():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    b = torch.randn(3, 4, 5, dtype=torch.float32)
    out = torch.empty(2, 3, 5, dtype=torch.float32)
    with pytest.raises(RefBackendError, match="batch dimensions"):
        run_bmm(a, b, out)
