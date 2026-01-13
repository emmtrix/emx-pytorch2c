from __future__ import annotations

from enum import Enum


class ScalarFunctionError(RuntimeError):
    pass


class ScalarType(str, Enum):
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    BOOL = "bool"

    @classmethod
    def from_torch_dtype(cls, dtype: object) -> "ScalarType":
        if isinstance(dtype, ScalarType):
            return dtype
        if isinstance(dtype, str):
            dtype_name = dtype
        else:
            dtype_name = getattr(dtype, "name", None) or str(dtype)
        normalized = dtype_name.lower()
        if normalized.startswith("torch."):
            normalized = normalized[len("torch.") :]
        mapping = {
            "float32": cls.F32,
            "float64": cls.F64,
            "int8": cls.I8,
            "int16": cls.I16,
            "int32": cls.I32,
            "int64": cls.I64,
            "uint8": cls.U8,
            "uint16": cls.U16,
            "uint32": cls.U32,
            "uint64": cls.U64,
            "bool": cls.BOOL,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            raise ScalarFunctionError(
                f"unsupported dtype for scalar functions: {dtype_name}"
            ) from exc
