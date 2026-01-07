#!/usr/bin/env python
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, Set

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _torch_base_version() -> str:
    return torch.__version__.split("+", 1)[0]


def _native_functions_path() -> Path:
    version = _torch_base_version()
    path = REPO_ROOT / "scripts" / f"native_functions.{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected native functions file at {path} for torch {version}, but it does not exist."
        )
    return path


def _core_ops_from_native_functions(path: Path) -> Set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected native functions YAML to be a list, got {type(data)}")
    ops: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict):
            continue
        func_sig = entry.get("func")
        if not func_sig:
            continue
        tags = entry.get("tags", [])
        if isinstance(tags, str):
            tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        else:
            tags_list = list(tags) if isinstance(tags, list) else []
        if "core" in tags_list:
            ops.add(func_sig.split("(", 1)[0].strip())
    return ops


def _expand_op_names(op_names: Iterable[str]) -> Set[str]:
    expanded: set[str] = set()
    for name in op_names:
        expanded.add(name)
        if "." in name:
            expanded.add(name.split(".", 1)[0])
    return expanded


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
        return ".".join(parts[3:])
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
    return _expand_op_names(ops)


def _format_support_label(codegen_supported: bool) -> str:
    return "✅" if codegen_supported else "—"


def _summarize_ops(aten_ops: Iterable[str], codegen_ops: Set[str]) -> tuple[int, int]:
    total = 0
    supported = 0
    for op_name in aten_ops:
        total += 1
        if op_name in codegen_ops:
            supported += 1
    return total, supported


def main() -> None:
    native_functions = _native_functions_path()
    aten_ops = sorted(_core_ops_from_native_functions(native_functions))
    codegen_ops = _ops_from_codegen_tests(
        REPO_ROOT / "tests" / "test_codegen_ops.py"
    )

    print("# Core ATen ops support (codegen backend)")
    print()
    print("| aten op | codegen support |")
    print("| --- | --- |")
    for op_name in aten_ops:
        print(f"| `{op_name}` | {_format_support_label(op_name in codegen_ops)} |")

    total, supported = _summarize_ops(aten_ops, codegen_ops)
    unsupported = total - supported
    codegen_percent = (supported / total * 100) if total else 0
    print()
    print("## Summary")
    print(f"- total core aten ops: {total}")
    print(
        f"- supported by codegen: {supported} / {total} ({codegen_percent:.1f}%)"
    )
    print(f"- unsupported by codegen: {unsupported}")


if __name__ == "__main__":
    main()
