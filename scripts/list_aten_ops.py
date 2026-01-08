#!/usr/bin/env python
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, Set

import torch
from torch.testing._internal.common_methods_invocations import op_db

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



def _attribute_chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return None


def _extract_aten_op_name(node: ast.AST, name_aliases: dict[str, str]) -> str | None:
    parts = _attribute_chain(node)
    if not parts:
        return None
    if len(parts) >= 4 and parts[:3] == ["torch", "ops", "aten"]:
        return parts[3]
    if len(parts) >= 2 and parts[0] in name_aliases:
        return name_aliases[parts[0]]
    return None


def _alias_from_assignment(node: ast.Assign) -> tuple[str, str] | None:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    target_name = node.targets[0].id
    value = node.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id == "getattr" and len(value.args) >= 2:
            aten_root = value.args[0]
            name_arg = value.args[1]
            if (
                isinstance(aten_root, ast.Attribute)
                and isinstance(aten_root.value, ast.Attribute)
                and isinstance(aten_root.value.value, ast.Name)
                and aten_root.value.value.id == "torch"
                and aten_root.value.attr == "ops"
                and aten_root.attr == "aten"
                and isinstance(name_arg, ast.Constant)
                and isinstance(name_arg.value, str)
            ):
                return target_name, name_arg.value
    return None


def _ops_from_codegen_tests(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ops: set[str] = set()
    aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            alias = _alias_from_assignment(node)
            if alias:
                aliases[alias[0]] = alias[1]
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "CODEGEN_ATEN_OPS",
                    "INPLACE_ATEN_OPS",
                }:
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            op_name = _extract_aten_op_name(elt, aliases)
                            if op_name:
                                ops.add(op_name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "append" or not node.args:
                continue
            if isinstance(node.func.value, ast.Name) and node.func.value.id in {
                "CODEGEN_ATEN_OPS",
                "INPLACE_ATEN_OPS",
            }:
                op_name = _extract_aten_op_name(node.args[0], aliases)
                if op_name:
                    ops.add(op_name)
    return ops


def _format_support_label(codegen_supported: bool) -> str:
    return "✅" if codegen_supported else "—"


def _summarize_ops(aten_ops: Iterable[str], codegen_ops: Set[str]) -> tuple[int, int]:
    total = 0
    codegen_only = 0
    for op_name in aten_ops:
        total += 1
        if op_name in codegen_ops:
            codegen_only += 1
    return total, codegen_only


def _opinfo_display_name(opinfo) -> str:
    variant = opinfo.variant_test_name
    if variant:
        return f"{opinfo.aten_name}.{variant}"
    return opinfo.aten_name


def _opinfo_dtypes(opinfo) -> tuple[torch.dtype, ...]:
    dtypes = opinfo.dtypesIfCPU or opinfo.dtypes
    if dtypes is None:
        return ()
    dtype_list = list(dtypes)
    return tuple(sorted(dtype_list, key=lambda dtype: str(dtype)))


def _count_opinfo_samples(opinfo) -> int:
    dtypes = _opinfo_dtypes(opinfo)
    if not dtypes:
        return 0
    samples = list(opinfo.sample_inputs("cpu", dtypes[0]))
    return len(samples)


def _opinfo_test_case_counts(op_names: Set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for opinfo in op_db:
        name = _opinfo_display_name(opinfo)
        if name not in op_names:
            continue
        counts[name] = counts.get(name, 0) + _count_opinfo_samples(opinfo)
    return counts


def main() -> None:
    aten_ops = sorted(_aten_ops_from_schemas())
    codegen_ops = _ops_from_codegen_tests(
        REPO_ROOT / "tests" / "test_codegen_ops.py"
    )
    opinfo_counts = _opinfo_test_case_counts(set(aten_ops))

    print("# All ATen ops support (codegen backend)")
    print()
    print(
        "This list shows all ATen operators, whether the codegen backend supports them,"
        " and how many OpInfo sample inputs exist for the first CPU dtype."
    )
    print()
    print("| aten op | codegen support | opinfo test cases |")
    print("| --- | --- | --- |")
    for op_name in aten_ops:
        opinfo_count = opinfo_counts.get(op_name)
        opinfo_label = "—" if opinfo_count is None else str(opinfo_count)
        print(
            f"| `{op_name}` | {_format_support_label(op_name in codegen_ops)} | {opinfo_label} |"
        )

    total, supported_codegen = _summarize_ops(aten_ops, codegen_ops)
    unsupported = total - supported_codegen
    codegen_percent = (supported_codegen / total * 100) if total else 0
    print("\n## Summary")
    print(f"- total aten ops: {total}")
    print(
        f"- supported by codegen: {supported_codegen} / {total} ({codegen_percent:.1f} %)"
    )
    print(f"- unsupported by codegen: {unsupported}")


if __name__ == "__main__":
    main()
