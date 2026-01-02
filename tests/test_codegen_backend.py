import os
from pathlib import Path

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from codegen_backend import codegen_add_backend
from codegen_backend.backend import get_add_source


REFERENCE_DIR = Path(__file__).resolve().parent / "codegen_refs"


def _assert_codegen_source_matches(reference_name: str) -> None:
    reference_path = REFERENCE_DIR / reference_name
    source = get_add_source().lstrip()
    if os.getenv("UPDATE_CODEGEN_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = reference_path.read_text(encoding="utf-8")
    assert source == expected


def add_fn(a, b):
    return a + b


def mul_fn(a, b):
    return a * b


def test_codegen_add_matches_eager():
    _assert_codegen_source_matches("add_matches_eager.c")
    compiled = torch.compile(add_fn, backend=codegen_add_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, add_fn(a, b))


def test_codegen_add_handles_non_contiguous():
    _assert_codegen_source_matches("add_handles_non_contiguous.c")
    compiled = torch.compile(add_fn, backend=codegen_add_backend)
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    result = compiled(a, b)
    torch.testing.assert_close(result, add_fn(a, b))


def test_codegen_add_rejects_other_ops():
    _assert_codegen_source_matches("add_rejects_other_ops.c")
    compiled = torch.compile(mul_fn, backend=codegen_add_backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    with pytest.raises(BackendCompilerFailed, match="Unsupported call_function"):
        compiled(a, b)
