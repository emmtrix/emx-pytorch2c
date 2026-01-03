import os
from pathlib import Path

import pytest
import torch
from codegen_backend import codegen_generic_backend
from codegen_backend.backend import (
    get_generic_source,
)
REFERENCE_DIR = Path(__file__).resolve().parent / "codegen_refs"


def _assert_codegen_source_matches(
    reference_name: str, source_fn, fn, example_inputs
) -> None:
    reference_path = REFERENCE_DIR / reference_name
    gm = torch.fx.symbolic_trace(fn)
    source = source_fn(gm, example_inputs).lstrip()
    if os.getenv("UPDATE_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = reference_path.read_text(encoding="utf-8")
    assert source == expected


def add_chain_fn(a, b, c):
    return (a + b) + c


def add_broadcast_fn(a, b):
    return a + b


def add_strided_fn(a, b):
    return a + b


def sub_chain_fn(a, b, c):
    return (a - b) - c


def mul_chain_fn(a, b, c):
    return (a * b) * c


def mixed_ops_fn(a, b, c):
    return torch.relu(a + b) - c


def atan_fn(a):
    return torch.atan(a)


def inplace_fn(a):
    b = torch.atan(a)
    b = torch.ops.aten.add_.Tensor(b, a)
    return torch.mul(b, a)


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
        (
            "mul_chain.c",
            mul_chain_fn,
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


def test_codegen_generic_handles_add_broadcast():
    a = torch.randn(2, 1, 3, dtype=torch.float32)
    b = torch.randn(1, 4, 1, dtype=torch.float32)
    _assert_codegen_source_matches(
        "add_broadcast.c", get_generic_source, add_broadcast_fn, (a, b)
    )
    compiled = torch.compile(add_broadcast_fn, backend=codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, add_broadcast_fn(a, b))


def test_codegen_generic_handles_strided_inputs():
    base = torch.randn(2, 3, dtype=torch.float32)
    a = base.t()
    b = torch.randn(2, 3, dtype=torch.float32).t()
    source = get_generic_source(torch.fx.symbolic_trace(add_strided_fn), (a, b))
    assert "((float*)a)" in source
    assert "((float*)b)" in source
    _assert_codegen_source_matches(
        "add_strided.c", get_generic_source, add_strided_fn, (a, b)
    )
    compiled = torch.compile(add_strided_fn, backend=codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, add_strided_fn(a, b))


def test_codegen_generic_handles_atan():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches("atan.c", get_generic_source, atan_fn, (a,))
    compiled = torch.compile(atan_fn, backend=codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, atan_fn(a))


def test_codegen_generic_supports_inplace_ops():
    a = torch.randn(2, 3, dtype=torch.float32)
    expected = a.clone()
    gm = torch.fx.symbolic_trace(inplace_fn)
    source = get_generic_source(gm, (a,))
    assert "node2_add_f32(tmp_0, input_0, tmp_0);" in source
    _assert_codegen_source_matches(
        "inplace_chain.c", get_generic_source, inplace_fn, (a,)
    )
    compiled = torch.compile(inplace_fn, backend=codegen_generic_backend)
    result = compiled(a)
    expected_result = inplace_fn(expected)
    torch.testing.assert_close(result, expected_result)
    torch.testing.assert_close(a, expected)
