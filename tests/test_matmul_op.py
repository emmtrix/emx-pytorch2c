import pytest
import torch

from ref_backend.cffi_bindings import RefBackendError, run_matmul


@pytest.mark.parametrize("m,k,n", [(2, 3, 4), (1, 5, 1), (0, 3, 2)])
def test_matmul_op_matches_eager(m, k, n):
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    out = torch.empty(m, n, dtype=torch.float32)
    run_matmul(a, b, out)
    torch.testing.assert_close(out, a @ b)


def test_matmul_op_rejects_non_contiguous():
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    out = torch.empty(4, 4, dtype=torch.float32)
    with pytest.raises(RefBackendError, match="contiguous"):
        run_matmul(a, b, out)
