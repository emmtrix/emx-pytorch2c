import os
from pathlib import Path

import pytest
import torch
from codegen_backend import codegen_generic_backend
from codegen_backend.backend import (
    get_generic_source,
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


def dnn_fn(a, b, c):
    return torch.relu(a @ b + c)


def mixed_ops_fn(a, b, c):
    return torch.relu(a + b) - c


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


def _dnn_inputs():
    return (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
        torch.randn(2, 4, dtype=torch.float32),
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
        ("add_matches_eager.c", add_fn, get_generic_source, codegen_generic_backend),
        ("sub_matches_eager.c", sub_fn, get_generic_source, codegen_generic_backend),
        (
            "matmul_matches_eager.c",
            matmul_fn,
            get_generic_source,
            codegen_generic_backend,
        ),
        ("bmm_matches_eager.c", bmm_fn, get_generic_source, codegen_generic_backend),
        ("dnn_matches_eager.c", dnn_fn, get_generic_source, codegen_generic_backend),
    ],
)
def test_codegen_binary_matches_eager(reference_name, fn, source_fn, backend):
    if fn is matmul_fn:
        a, b = _matmul_inputs()
    elif fn is dnn_fn:
        a, b, c = _dnn_inputs()
    elif fn is bmm_fn:
        a, b = _bmm_inputs()
    else:
        a = torch.randn(2, 3, dtype=torch.float32)
        b = torch.randn(2, 3, dtype=torch.float32)
    if fn is dnn_fn:
        _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b, c))
    else:
        _assert_codegen_source_matches(reference_name, source_fn, fn, (a, b))
    compiled = torch.compile(fn, backend=backend)
    if fn is dnn_fn:
        result = compiled(a, b, c)
        torch.testing.assert_close(result, fn(a, b, c))
    else:
        result = compiled(a, b)
        torch.testing.assert_close(result, fn(a, b))


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        (
            "add_handles_non_contiguous.c",
            add_fn,
            get_generic_source,
            codegen_generic_backend,
        ),
        (
            "sub_handles_non_contiguous.c",
            sub_fn,
            get_generic_source,
            codegen_generic_backend,
        ),
        (
            "matmul_handles_non_contiguous.c",
            matmul_fn,
            get_generic_source,
            codegen_generic_backend,
        ),
        (
            "bmm_handles_non_contiguous.c",
            bmm_fn,
            get_generic_source,
            codegen_generic_backend,
        ),
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
        (
            "add_chain.c",
            add_chain_fn,
            get_generic_source,
            codegen_generic_backend,
        ),
        (
            "sub_chain.c",
            sub_chain_fn,
            get_generic_source,
            codegen_generic_backend,
        ),
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
    ("fn", "backend"),
    [
        (matmul_fn, codegen_generic_backend),
        (bmm_fn, codegen_generic_backend),
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


def test_codegen_generic_handles_mixed_ops():
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    c = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "mixed_ops.c", get_generic_source, mixed_ops_fn, (a, b, c)
    )
    compiled = torch.compile(mixed_ops_fn, backend=codegen_generic_backend)
    result = compiled(a, b, c)
    torch.testing.assert_close(result, mixed_ops_fn(a, b, c))
