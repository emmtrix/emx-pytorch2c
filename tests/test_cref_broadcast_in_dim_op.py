import torch

from c_ref_backend.backend import c_ref_backend_backend


def test_broadcast_in_dim_matches_eager():
    def f(a):
        return torch.ops.prims.broadcast_in_dim(a, (2, 3), (1,))

    compiled = torch.compile(f, backend=c_ref_backend_backend)
    a = torch.randn(3, dtype=torch.float32)
    result = compiled(a)
    torch.testing.assert_close(result, f(a))


def test_broadcast_in_dim_higher_rank_matches_eager():
    def f(a):
        return torch.ops.prims.broadcast_in_dim(a, (2, 4, 3), (0, 2))

    compiled = torch.compile(f, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    result = compiled(a)
    torch.testing.assert_close(result, f(a))
