#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, Set

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _aten_ops_from_schemas() -> Set[str]:
    ops: set[str] = set()
    for schema in torch._C._jit_get_all_schemas():
        if not schema.name.startswith("aten::"):
            continue
        op_name = schema.name.split("::", 1)[1]
        if op_name.endswith("Implicit"):
            continue
        ops.add(op_name)
    return ops


def _ops_from_backend_source(path: Path) -> Set[str]:
    pattern = re.compile(r"torch\.ops\.aten\.([A-Za-z0-9_]+)")
    return set(pattern.findall(path.read_text(encoding="utf-8")))


def _op_names_from_overloads(overloads: Iterable[object]) -> Set[str]:
    names: set[str] = set()
    for overload in overloads:
        schema_name = getattr(getattr(overload, "_schema", None), "name", "")
        if schema_name.startswith("aten::"):
            names.add(schema_name.split("::", 1)[1])
    return names


def _ops_from_codegen_tests(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    spec = importlib.util.spec_from_file_location("test_codegen_ops", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if hasattr(module, "CODEGEN_ATEN_OPS"):
        ops = _op_names_from_overloads(module.CODEGEN_ATEN_OPS)
        if hasattr(module, "INPLACE_ATEN_OPS"):
            ops |= _op_names_from_overloads(module.INPLACE_ATEN_OPS)
        return ops
    return set(getattr(module, "CODEGEN_OP_NAMES", []))


def _format_row(op_name: str, cref_supported: bool, codegen_supported: bool) -> str:
    cref_label = "yes" if cref_supported else "-"
    codegen_label = "yes" if codegen_supported else "-"
    return f"{op_name:40} {cref_label:5} {codegen_label:7}"


def _summarize_ops(
    aten_ops: Iterable[str], cref_ops: Set[str], codegen_ops: Set[str]
) -> tuple[int, int, int, int]:
    total = 0
    cref_only = 0
    codegen_only = 0
    both = 0
    for op_name in aten_ops:
        total += 1
        cref = op_name in cref_ops
        codegen = op_name in codegen_ops
        if cref and codegen:
            both += 1
        elif cref:
            cref_only += 1
        elif codegen:
            codegen_only += 1
    return total, cref_only, codegen_only, both


def main() -> None:
    aten_ops = sorted(_aten_ops_from_schemas())
    cref_ops = _ops_from_backend_source(REPO_ROOT / "src/c_ref_backend/backend.py")
    codegen_ops = _ops_from_codegen_tests(
        REPO_ROOT / "tests" / "test_codegen_ops.py"
    )

    print(f"{'aten op':40} {'cref':5} {'codegen':7}")
    print("-" * 56)
    for op_name in aten_ops:
        print(_format_row(op_name, op_name in cref_ops, op_name in codegen_ops))

    total, cref_only, codegen_only, both = _summarize_ops(
        aten_ops, cref_ops, codegen_ops
    )
    unsupported = total - cref_only - codegen_only - both
    supported_codegen = codegen_only + both
    supported_cref = cref_only + both
    codegen_percent = (supported_codegen / total * 100) if total else 0
    cref_percent = (supported_cref / total * 100) if total else 0
    print("\nSummary")
    print(f"total aten ops: {total}")
    print(f"supported by cref only: {cref_only}")
    print(f"supported by codegen only: {codegen_only}")
    print(f"supported by both: {both}")
    print(f"unsupported by either: {unsupported}")
    print(
        f"supported by codegen: {supported_codegen} / {total} ({codegen_percent:.1f} %)"
    )
    print(f"supported by c-ref: {supported_cref} / {total} ({cref_percent:.1f} %)")


if __name__ == "__main__":
    main()
