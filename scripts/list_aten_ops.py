#!/usr/bin/env python
from __future__ import annotations

import ast
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


def _format_row(op_name: str, codegen_supported: bool) -> str:
    codegen_label = "yes" if codegen_supported else "-"
    return f"{op_name:40} {codegen_label:7}"


def _summarize_ops(aten_ops: Iterable[str], codegen_ops: Set[str]) -> tuple[int, int]:
    total = 0
    codegen_only = 0
    for op_name in aten_ops:
        total += 1
        if op_name in codegen_ops:
            codegen_only += 1
    return total, codegen_only


def main() -> None:
    aten_ops = sorted(_aten_ops_from_schemas())
    codegen_ops = _ops_from_codegen_tests(
        REPO_ROOT / "tests" / "test_codegen_ops.py"
    )

    print(f"{'aten op':40} {'codegen':7}")
    print("-" * 50)
    for op_name in aten_ops:
        print(_format_row(op_name, op_name in codegen_ops))

    total, supported_codegen = _summarize_ops(aten_ops, codegen_ops)
    unsupported = total - supported_codegen
    codegen_percent = (supported_codegen / total * 100) if total else 0
    print("\nSummary")
    print(f"total aten ops: {total}")
    print(f"supported by codegen: {supported_codegen}")
    print(f"unsupported by codegen: {unsupported}")
    print(
        f"supported by codegen: {supported_codegen} / {total} ({codegen_percent:.1f} %)"
    )


if __name__ == "__main__":
    main()
