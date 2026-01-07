from __future__ import annotations

import torch
import torch.fx

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend import backend
from codegen_backend.c_types import dtype_to_c_type, format_scalar
from codegen_backend.dtypes import _CODEGEN_DTYPES
from codegen_backend.indexing import format_input_access, format_output_access
from codegen_backend.templates import get_template_env


def test_templates_env_cached() -> None:
    env = get_template_env()
    template = env.get_template("elementwise_kernel.c.j2")
    assert template is not None
    assert get_template_env() is env


def test_c_types_dtype_to_c_type() -> None:
    assert dtype_to_c_type(torch.float32) == "float"
    assert dtype_to_c_type(torch.int64) == "int64_t"
    try:
        dtype_to_c_type(torch.float64)
    except RefBackendError:
        pass
    else:
        raise AssertionError("Expected RefBackendError for unsupported dtype")


def test_c_types_format_scalar() -> None:
    dtype = _CODEGEN_DTYPES[torch.float32]
    assert format_scalar(1.5, dtype) == "1.5f"
    assert format_scalar(torch.tensor(2.0), dtype) == "2.0f"


def test_indexing_format_input_access() -> None:
    access = format_input_access(
        "a",
        (1, 3),
        (3, 1),
        (2, 3),
        broadcast_contiguous=True,
        c_type="float",
        input_is_contiguous=True,
    )
    assert access == "a[0][i1]"

    access = format_input_access(
        "a",
        (2, 3),
        (1, 2),
        (2, 3),
        broadcast_contiguous=False,
        c_type="float",
        input_is_contiguous=False,
    )
    assert access == "((float*)a)[i0 * 1 + i1 * 2]"


def test_indexing_format_output_access() -> None:
    access = format_output_access(
        "out",
        (2, 3),
        (3, 1),
        c_type="float",
        output_is_contiguous=True,
    )
    assert access == "out[i0][i1]"

    access = format_output_access(
        "out",
        (2, 3),
        (4, 1),
        c_type="float",
        output_is_contiguous=False,
    )
    assert access == "((float*)out)[i0 * 4 + i1 * 1]"


def test_codegen_backend_integration_add() -> None:
    def add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    gm = torch.fx.symbolic_trace(add_fn)
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    compiled = backend.codegen_generic_backend(gm, [a, b])
    result = compiled(a, b)
    assert torch.allclose(result, a + b)
