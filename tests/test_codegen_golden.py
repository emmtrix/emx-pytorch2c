import os
from pathlib import Path

import pytest
import torch
from codegen_backend import CodegenBackend
from codegen_backend.indexing import _emit_strided_access, _format_strided_access

REFERENCE_DIR = Path(__file__).resolve().parent / "codegen_refs"
BACKEND = CodegenBackend()


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


def mish_fn(a):
    return torch.ops.aten.mish.default(a)


def inplace_fn(a):
    b = torch.atan(a)
    b = torch.ops.aten.add_.Tensor(b, a)
    return torch.mul(b, a)


def reduction_global_fn(a):
    return a.sum()


def reduction_mean_global_fn(a):
    return a.mean()


def reduction_mean_dim_fn(a):
    return a.mean(dim=1)


def reduction_dim_fn(a):
    return a.sum(dim=1)


def reduction_keepdim_fn(a):
    return a.sum(dim=1, keepdim=True)


def reduction_strided_fn(a):
    return a.sum(dim=0)


def reduction_broadcast_fn(a, b):
    return (a + b).sum(dim=1)


def argmax_dim_fn(a):
    return torch.ops.aten.argmax.default(a, 1, False)


def where_fn(condition, a, b):
    return torch.where(condition, a, b)


def cat_fn(a, b):
    return torch.cat([a, b], dim=1)


def conv2d_fn(a, weight, bias):
    return torch.ops.aten.conv2d.default(
        a, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
    )


def max_pool2d_fn(a):
    return torch.ops.aten.max_pool2d.default(
        a, (2, 2), (1, 1), (0, 0), (1, 1), False
    )


def temp_alloc_fn(a, b, c):
    return (a + b) + c


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        (
            "add_chain.c",
            add_chain_fn,
            BACKEND.get_generic_source,
            BACKEND.codegen_generic_backend,
        ),
        (
            "sub_chain.c",
            sub_chain_fn,
            BACKEND.get_generic_source,
            BACKEND.codegen_generic_backend,
        ),
        (
            "mul_chain.c",
            mul_chain_fn,
            BACKEND.get_generic_source,
            BACKEND.codegen_generic_backend,
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
        "mixed_ops.c", BACKEND.get_generic_source, mixed_ops_fn, (a, b, c)
    )
    compiled = torch.compile(mixed_ops_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b, c)
    torch.testing.assert_close(result, mixed_ops_fn(a, b, c))


def test_codegen_generic_handles_add_broadcast():
    a = torch.randn(2, 1, 3, dtype=torch.float32)
    b = torch.randn(1, 4, 1, dtype=torch.float32)
    _assert_codegen_source_matches(
        "add_broadcast.c", BACKEND.get_generic_source, add_broadcast_fn, (a, b)
    )
    compiled = torch.compile(add_broadcast_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, add_broadcast_fn(a, b))


def test_codegen_generic_handles_strided_inputs():
    base = torch.randn(2, 3, dtype=torch.float32)
    a = base.t()
    b = torch.randn(2, 3, dtype=torch.float32).t()
    source = BACKEND.get_generic_source(torch.fx.symbolic_trace(add_strided_fn), (a, b))
    assert "((float*)a)" in source
    assert "((float*)b)" in source
    _assert_codegen_source_matches(
        "add_strided.c", BACKEND.get_generic_source, add_strided_fn, (a, b)
    )
    compiled = torch.compile(add_strided_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, add_strided_fn(a, b))


def test_codegen_generic_handles_conv2d():
    a = torch.randn(1, 2, 5, 5, dtype=torch.float32)
    weight = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "conv2d.c", BACKEND.get_generic_source, conv2d_fn, (a, weight, bias)
    )
    compiled = torch.compile(conv2d_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, weight, bias)
    torch.testing.assert_close(result, conv2d_fn(a, weight, bias))


def test_codegen_generic_handles_max_pool2d():
    a = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "max_pool2d.c", BACKEND.get_generic_source, max_pool2d_fn, (a,)
    )
    compiled = torch.compile(max_pool2d_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, max_pool2d_fn(a))


def test_codegen_generic_handles_atan():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches("atan.c", BACKEND.get_generic_source, atan_fn, (a,))
    compiled = torch.compile(atan_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, atan_fn(a))


def test_codegen_generic_handles_mish():
    a = torch.randn(2, 3, dtype=torch.float32)
    source = BACKEND.get_generic_source(torch.fx.symbolic_trace(mish_fn), (a,))
    assert "ref_scalar_f32_mish" in source
    compiled = torch.compile(mish_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, mish_fn(a))


def test_codegen_generic_handles_where():
    condition = torch.tensor([[True, False, True]], dtype=torch.bool)
    a = torch.randn(1, 3, dtype=torch.float32)
    b = torch.randn(1, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "where.c", BACKEND.get_generic_source, where_fn, (condition, a, b)
    )
    compiled = torch.compile(where_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(condition, a, b)
    torch.testing.assert_close(result, where_fn(condition, a, b))


def test_codegen_generic_handles_cat():
    a = torch.randn(2, 2, dtype=torch.float32)
    b = torch.randn(2, 1, dtype=torch.float32)
    _assert_codegen_source_matches(
        "cat.c", BACKEND.get_generic_source, cat_fn, (a, b)
    )
    compiled = torch.compile(cat_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b)
    torch.testing.assert_close(result, cat_fn(a, b))


def test_codegen_generic_supports_inplace_ops():
    a = torch.randn(2, 3, dtype=torch.float32)
    expected = a.clone()
    gm = torch.fx.symbolic_trace(inplace_fn)
    source = BACKEND.get_generic_source(gm, (a,))
    assert "node2_add_f32(tmp_0, input_0, tmp_0);" in source
    _assert_codegen_source_matches(
        "inplace_chain.c", BACKEND.get_generic_source, inplace_fn, (a,)
    )
    compiled = torch.compile(inplace_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    expected_result = inplace_fn(expected)
    torch.testing.assert_close(result, expected_result)
    torch.testing.assert_close(a, expected)


def test_i32():
    a = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    b = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    _assert_codegen_source_matches(
        "mul_chain_i32.c", BACKEND.get_generic_source, mul_chain_fn, (a, b, b)
    )
    compiled = torch.compile(mul_chain_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a, b, b)
    torch.testing.assert_close(result, mul_chain_fn(a, b, b))


def test_emit_strided_access_expressions():
    broadcast_expr = _format_strided_access(
        "b", (1, 3), (3, 1), (2, 3)
    )
    helper_expr = _emit_strided_access(
        "b", ("i0", "i1"), (3, 1), contig=False, sizes=(1, 3)
    )
    assert broadcast_expr == helper_expr
    assert broadcast_expr == "((float*)b)[i1 * 1]"
    assert (
        _emit_strided_access("a", ("i", "t"), (5, 1), contig=False, sizes=(2, 3))
        == "((float*)a)[i * 5 + t * 1]"
    )
    assert (
        _emit_strided_access("a", ("i", "t"), (5, 1), contig=True, sizes=(2, 3))
        == "a[i][t]"
    )


def test_codegen_generic_handles_reduction_global():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "sum_global.c", BACKEND.get_generic_source, reduction_global_fn, (a,)
    )
    compiled = torch.compile(reduction_global_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_global_fn(a))


def test_codegen_generic_handles_reduction_mean_global():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "mean_global.c", BACKEND.get_generic_source, reduction_mean_global_fn, (a,)
    )
    compiled = torch.compile(
        reduction_mean_global_fn, backend=BACKEND.codegen_generic_backend
    )
    result = compiled(a)
    torch.testing.assert_close(result, reduction_mean_global_fn(a))


def test_codegen_generic_handles_reduction_mean_dim():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "mean_dim.c", BACKEND.get_generic_source, reduction_mean_dim_fn, (a,)
    )
    compiled = torch.compile(reduction_mean_dim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_mean_dim_fn(a))


def test_codegen_generic_handles_reduction_dim():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "sum_dim.c", BACKEND.get_generic_source, reduction_dim_fn, (a,)
    )
    compiled = torch.compile(reduction_dim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_dim_fn(a))


def test_codegen_generic_handles_reduction_keepdim():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "sum_keepdim.c", BACKEND.get_generic_source, reduction_keepdim_fn, (a,)
    )
    compiled = torch.compile(reduction_keepdim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_keepdim_fn(a))


def test_codegen_generic_handles_reduction_strided():
    base = torch.randn(3, 4, dtype=torch.float32)
    a = base.t()
    _assert_codegen_source_matches(
        "sum_strided.c", BACKEND.get_generic_source, reduction_strided_fn, (a,)
    )
    compiled = torch.compile(reduction_strided_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, reduction_strided_fn(a))


def test_codegen_generic_handles_reduction_broadcast_producer():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    b = torch.randn(1, 4, dtype=torch.float32)
    _assert_codegen_source_matches(
        "sum_broadcast.c", BACKEND.get_generic_source, reduction_broadcast_fn, (a, b)
    )
    compiled = torch.compile(
        reduction_broadcast_fn, backend=BACKEND.codegen_generic_backend
    )
    result = compiled(a, b)
    torch.testing.assert_close(result, reduction_broadcast_fn(a, b))


def test_codegen_generic_handles_argmax_dim():
    a = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "argmax_dim.c", BACKEND.get_generic_source, argmax_dim_fn, (a,)
    )
    compiled = torch.compile(argmax_dim_fn, backend=BACKEND.codegen_generic_backend)
    result = compiled(a)
    torch.testing.assert_close(result, argmax_dim_fn(a))


def test_elementwise_kernel_source_matches_expected():
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    _assert_codegen_source_matches(
        "atan_single.c", BACKEND.get_generic_source, atan_fn, (a,)
    )
    _assert_codegen_source_matches(
        "add_single.c", BACKEND.get_generic_source, add_broadcast_fn, (a, b)
    )


def test_codegen_generic_temp_allocations_exceed_threshold():
    backend = CodegenBackend(temp_allocation_threshold=1)
    a = torch.randn(1, 2, 2, 2, dtype=torch.float32)
    b = torch.randn(1, 2, 2, 2, dtype=torch.float32)
    c = torch.randn(1, 2, 2, 2, dtype=torch.float32)
    _assert_codegen_source_matches(
        "temp_alloc.c", backend.get_generic_source, temp_alloc_fn, (a, b, c)
    )
