import os
from pathlib import Path

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from codegen_backend import (
    codegen_add_backend,
    codegen_bmm_backend,
    codegen_matmul_backend,
    codegen_sub_backend,
)
from codegen_backend.backend import (
    get_add_source,
    get_bmm_source,
    get_matmul_source,
    get_sub_source,
)
from c_ref_backend.cffi_bindings import RefBackendError


REFERENCE_DIR = Path(__file__).resolve().parent / "codegen_refs"


def _assert_codegen_source_matches(
    reference_name: str, source_fn, fn, example_inputs
) -> None:
    reference_path = REFERENCE_DIR / reference_name
    gm = torch.fx.symbolic_trace(fn)
    source = source_fn(gm, example_inputs).lstrip()
    if os.getenv("UPDATE_CODEGEN_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = reference_path.read_text(encoding="utf-8")
    assert source == expected


def add_fn(a, b):
    return a + b


def mul_fn(a, b):
    return a * b


def add_chain_fn(a, b, c):
    return (a + b) + c


def sub_fn(a, b):
    return a - b


def sub_chain_fn(a, b, c):
    return (a - b) - c


def matmul_fn(a, b):
    return a @ b


def bmm_fn(a, b):
    return torch.bmm(a, b)


def _matmul_inputs():
    return (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
    )


def _bmm_inputs():
    return (
        torch.randn(2, 3, 4, dtype=torch.float32),
        torch.randn(2, 4, 5, dtype=torch.float32),
    )


def _matmul_noncontig_inputs():
    a = torch.randn(3, 4, dtype=torch.float32).t()
    b = torch.randn(5, 3, dtype=torch.float32).t()
    return a, b


def _bmm_noncontig_inputs():
    a = torch.randn(2, 3, 4, dtype=torch.float32).transpose(1, 2)
    b = torch.randn(2, 5, 3, dtype=torch.float32).transpose(1, 2)
    return a, b


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_matches_eager.c", add_fn, get_add_source, codegen_add_backend),
        ("sub_matches_eager.c", sub_fn, get_sub_source, codegen_sub_backend),
        (
            "matmul_matches_eager.c",
            matmul_fn,
            get_matmul_source,
            codegen_matmul_backend,
        ),
        ("bmm_matches_eager.c", bmm_fn, get_bmm_source, codegen_bmm_backend),
    ],
)
def test_codegen_binary_matches_eager(reference_name, fn, source_fn, backend):
    if fn is matmul_fn:
        a, b = _matmul_inputs()
    elif fn is bmm_fn:
        a, b = _bmm_inputs()
    else:
        a = torch.randn(2, 3, dtype=torch.float32)
        b = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b))
    compiled = torch.compile(fn, backend=backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, fn(a, b))


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_handles_non_contiguous.c", add_fn, get_add_source, codegen_add_backend),
        ("sub_handles_non_contiguous.c", sub_fn, get_sub_source, codegen_sub_backend),
        (
            "matmul_handles_non_contiguous.c",
            matmul_fn,
            get_matmul_source,
            codegen_matmul_backend,
        ),
        ("bmm_handles_non_contiguous.c", bmm_fn, get_bmm_source, codegen_bmm_backend),
    ],
)
def test_codegen_binary_handles_non_contiguous(
    reference_name, fn, source_fn, backend
):
    if fn is matmul_fn:
        a, b = _matmul_noncontig_inputs()
    elif fn is bmm_fn:
        a, b = _bmm_noncontig_inputs()
    else:
        a = torch.randn(4, 4, dtype=torch.float32).t()
        b = torch.randn(4, 4, dtype=torch.float32).t()
    _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b))
    compiled = torch.compile(fn, backend=backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, fn(a, b))


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_rejects_other_ops.c", add_fn, get_add_source, codegen_add_backend),
        ("sub_rejects_other_ops.c", sub_fn, get_sub_source, codegen_sub_backend),
    ],
)
def test_codegen_binary_rejects_other_ops(reference_name, fn, source_fn, backend):
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b))
    compiled = torch.compile(mul_fn, backend=backend)
    with pytest.raises(BackendCompilerFailed, match="Unsupported call_function"):
        compiled(a, b)


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_chain.c", add_chain_fn, get_add_source, codegen_add_backend),
        ("sub_chain.c", sub_chain_fn, get_sub_source, codegen_sub_backend),
    ],
)
def test_codegen_binary_handles_multiple_ops(
    reference_name, fn, source_fn, backend
):
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    c = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b, c))
    compiled = torch.compile(fn, backend=backend)
    result = compiled(a, b, c)
    torch.testing.assert_close(result, fn(a, b, c))


@pytest.mark.parametrize(
    ("fn", "source_fn", "backend", "reference_name"),
    [
        (matmul_fn, get_matmul_source, codegen_matmul_backend, "matmul_rejects_other_ops.c"),
        (bmm_fn, get_bmm_source, codegen_bmm_backend, "bmm_rejects_other_ops.c"),
    ],
)
def test_codegen_matmul_rejects_other_ops(fn, source_fn, backend, reference_name):
    if fn is matmul_fn:
        a, b = _matmul_inputs()
    else:
        a, b = _bmm_inputs()
    _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b))
    add_a = torch.randn(2, 3, dtype=torch.float32)
    add_b = torch.randn(2, 3, dtype=torch.float32)
    compiled = torch.compile(add_fn, backend=backend)
    with pytest.raises(BackendCompilerFailed, match="Unsupported call_function"):
        compiled(add_a, add_b)


@pytest.mark.parametrize(
    ("fn", "source_fn", "backend", "reference_name"),
    [
        (matmul_fn, get_matmul_source, codegen_matmul_backend, "matmul_rejects_multiple_ops.c"),
        (bmm_fn, get_bmm_source, codegen_bmm_backend, "bmm_rejects_multiple_ops.c"),
    ],
)
def test_codegen_matmul_rejects_multiple_ops(fn, source_fn, backend, reference_name):
    def chained(a, b):
        return fn(a, b) + fn(a, b)

    if fn is matmul_fn:
        a, b = _matmul_inputs()
    else:
        a, b = _bmm_inputs()
    _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b))
    compiled = torch.compile(chained, backend=backend)
    with pytest.raises(BackendCompilerFailed, match="Unsupported call_function"):
        compiled(a, b)


@pytest.mark.parametrize(
    ("fn", "backend"),
    [
        (matmul_fn, codegen_matmul_backend),
        (bmm_fn, codegen_bmm_backend),
    ],
)
def test_codegen_matmul_rejects_bad_runtime_shapes(fn, backend):
    if fn is matmul_fn:
        good_a, good_b = _matmul_inputs()
        bad_a = torch.randn(4, 2, dtype=torch.float32)
        bad_b = torch.randn(2, 3, dtype=torch.float32)
    else:
        good_a, good_b = _bmm_inputs()
        bad_a = torch.randn(3, 2, 4, dtype=torch.float32)
        bad_b = torch.randn(3, 4, 5, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(fn)
    compiled = backend(gm, [good_a, good_b])
    with pytest.raises(RefBackendError, match="requires inputs to have shapes"):
        compiled(bad_a, bad_b)
