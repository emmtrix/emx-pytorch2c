import pytest
import torch

from c_ref_backend.backend import c_ref_backend_backend
from c_ref_backend.cffi_bindings import RefBackendError


def f(a, b):
    return a + b


def g(a, b):
    return a @ b


def h(a, b):
    return torch.bmm(a, b)


def i(a, b):
    return a - b


def j(a, b):
    return a * b


def k(a, b):
    return torch.nn.functional.conv2d(a, b, stride=1, padding=1)


def test_torch_compile_add_matches_eager():
    compiled = torch.compile(f, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, f(a, b))


def test_torch_compile_add_broadcast_matches_eager():
    compiled = torch.compile(f, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, f(a, b))


def test_torch_compile_add_handles_non_contiguous():
    compiled = torch.compile(f, backend=c_ref_backend_backend)
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    result = compiled(a, b)
    torch.testing.assert_close(result, f(a, b))


def test_torch_compile_sub_matches_eager():
    compiled = torch.compile(i, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, i(a, b))


def test_torch_compile_sub_broadcast_matches_eager():
    compiled = torch.compile(i, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, i(a, b))


def test_torch_compile_sub_handles_non_contiguous():
    compiled = torch.compile(i, backend=c_ref_backend_backend)
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    result = compiled(a, b)
    torch.testing.assert_close(result, i(a, b))


def test_torch_compile_mul_matches_eager():
    compiled = torch.compile(j, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, j(a, b))


def test_torch_compile_mul_broadcast_matches_eager():
    compiled = torch.compile(j, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, j(a, b))


def test_torch_compile_mul_handles_non_contiguous():
    compiled = torch.compile(j, backend=c_ref_backend_backend)
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    result = compiled(a, b)
    torch.testing.assert_close(result, j(a, b))


def test_torch_compile_matmul_matches_eager():
    compiled = torch.compile(g, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, g(a, b))


def test_torch_compile_matmul_rejects_non_contiguous():
    compiled = torch.compile(g, backend=c_ref_backend_backend)
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    with pytest.raises(RefBackendError, match="contiguous"):
        compiled(a, b)


def test_torch_compile_bmm_matches_eager():
    compiled = torch.compile(h, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    b = torch.randn(2, 4, 5, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, h(a, b))


def test_torch_compile_bmm_rejects_non_contiguous():
    compiled = torch.compile(h, backend=c_ref_backend_backend)
    a = torch.randn(2, 4, 3, dtype=torch.float32).transpose(1, 2)
    b = torch.randn(2, 5, 4, dtype=torch.float32).transpose(1, 2)
    with pytest.raises(RefBackendError, match="contiguous"):
        compiled(a, b)


def test_torch_compile_conv2d_matches_eager():
    compiled = torch.compile(k, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, 8, 8, dtype=torch.float32)
    b = torch.randn(4, 3, 3, 3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, k(a, b))


def test_torch_compile_conv2d_rejects_non_contiguous():
    compiled = torch.compile(k, backend=c_ref_backend_backend)
    a = torch.randn(2, 3, 8, 8, dtype=torch.float32).transpose(2, 3)
    b = torch.randn(4, 3, 3, 3, dtype=torch.float32)
    with pytest.raises(RefBackendError, match="contiguous"):
        compiled(a, b)
