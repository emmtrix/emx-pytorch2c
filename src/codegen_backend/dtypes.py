from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _CodegenDType:
    torch_dtype: torch.dtype
    c_type: str
    scalar_prefix: str
    suffix: str


_CODEGEN_DTYPES = {
    torch.float32: _CodegenDType(
        torch_dtype=torch.float32,
        c_type="float",
        scalar_prefix="ref_scalar_f32_",
        suffix="f32",
    ),
    torch.float64: _CodegenDType(
        torch_dtype=torch.float64,
        c_type="double",
        scalar_prefix="ref_scalar_f64_",
        suffix="f64",
    ),
    torch.int8: _CodegenDType(
        torch_dtype=torch.int8,
        c_type="int8_t",
        scalar_prefix="ref_scalar_i8_",
        suffix="i8",
    ),
    torch.int16: _CodegenDType(
        torch_dtype=torch.int16,
        c_type="int16_t",
        scalar_prefix="ref_scalar_i16_",
        suffix="i16",
    ),
    torch.uint8: _CodegenDType(
        torch_dtype=torch.uint8,
        c_type="uint8_t",
        scalar_prefix="ref_scalar_u8_",
        suffix="u8",
    ),
    torch.uint16: _CodegenDType(
        torch_dtype=torch.uint16,
        c_type="uint16_t",
        scalar_prefix="ref_scalar_u16_",
        suffix="u16",
    ),
    torch.uint32: _CodegenDType(
        torch_dtype=torch.uint32,
        c_type="uint32_t",
        scalar_prefix="ref_scalar_u32_",
        suffix="u32",
    ),
    torch.int32: _CodegenDType(
        torch_dtype=torch.int32,
        c_type="int32_t",
        scalar_prefix="ref_scalar_i32_",
        suffix="i32",
    ),
    torch.int64: _CodegenDType(
        torch_dtype=torch.int64,
        c_type="int64_t",
        scalar_prefix="ref_scalar_i64_",
        suffix="i64",
    ),
    torch.bool: _CodegenDType(
        torch_dtype=torch.bool,
        c_type="bool",
        scalar_prefix="ref_scalar_bool_",
        suffix="bool",
    ),
    torch.uint64: _CodegenDType(
        torch_dtype=torch.uint64,
        c_type="uint64_t",
        scalar_prefix="ref_scalar_u64_",
        suffix="u64",
    ),
}

_INTEGER_CODEGEN_DTYPES = {
    torch.int8,
    torch.int16,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.int32,
    torch.int64,
    torch.uint64,
}
_C_TYPE_BY_DTYPE = {
    torch.bool: "uint8_t",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.uint8: "uint8_t",
    torch.uint16: "uint16_t",
    torch.uint32: "uint32_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint64: "uint64_t",
    torch.float32: "float",
    torch.float64: "double",
}
_EMBEDDING_INDEX_DTYPES = {torch.int32, torch.int64}
