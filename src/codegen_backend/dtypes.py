from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _CodegenDType:
    torch_dtype: torch.dtype
    c_type: str
    scalar_header: str
    scalar_prefix: str
    suffix: str


_CODEGEN_DTYPES = {
    torch.float32: _CodegenDType(
        torch_dtype=torch.float32,
        c_type="float",
        scalar_header="ops_scalar_f32.h",
        scalar_prefix="ref_scalar_f32_",
        suffix="f32",
    ),
    torch.int8: _CodegenDType(
        torch_dtype=torch.int8,
        c_type="int8_t",
        scalar_header="ops_scalar_i8.h",
        scalar_prefix="ref_scalar_i8_",
        suffix="i8",
    ),
    torch.int32: _CodegenDType(
        torch_dtype=torch.int32,
        c_type="int32_t",
        scalar_header="ops_scalar_i32.h",
        scalar_prefix="ref_scalar_i32_",
        suffix="i32",
    ),
    torch.bool: _CodegenDType(
        torch_dtype=torch.bool,
        c_type="bool",
        scalar_header="ops_scalar_bool.h",
        scalar_prefix="ref_scalar_bool_",
        suffix="bool",
    ),
}

_INTEGER_CODEGEN_DTYPES = {torch.int8, torch.int32}
_C_TYPE_BY_DTYPE = {
    torch.bool: "uint8_t",
    torch.int8: "int8_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.float32: "float",
}
_EMBEDDING_INDEX_DTYPES = {torch.int32, torch.int64}
