from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Set

from codegen_backend.dtypes import _CODEGEN_DTYPES
from codegen_backend.errors import CodegenBackendError


_REF_PREFIX_PATTERN = re.compile(r"\b(ref_scalar_[a-z0-9_]+)\b")
_FUNCTION_NAME_PATTERN = re.compile(
    r"static inline\s+[^\(]+\s+(ref_scalar_[a-z0-9_]+)\s*\("
)
_MACRO_INVOCATION_PATTERN = re.compile(r"^(REF_[A-Z0-9_]+)\((\w+)\)")


@dataclass(frozen=True)
class _ScalarHeaderInfo:
    prefix: str
    path: Path
    includes: List[str]
    preamble: List[str]
    functions: Dict[str, List[str]]
    function_deps: Dict[str, Set[str]]
    macro_defs: Dict[str, List[str]]
    macro_undefs: Dict[str, str]
    macro_functions: Dict[str, str]


class ScalarFunctionRegistry:
    def __init__(self, headers_root: Path | None = None) -> None:
        if headers_root is None:
            headers_root = Path(__file__).resolve().parents[2] / "csrc"
        self._headers_root = headers_root
        self._prefix_to_header = {
            dtype.scalar_prefix: dtype.scalar_header
            for dtype in _CODEGEN_DTYPES.values()
        }
        self._header_order = [
            dtype.scalar_prefix for dtype in _CODEGEN_DTYPES.values()
        ]
        self._headers: Dict[str, _ScalarHeaderInfo] = {}
        self._requested: List[str] = []
        self._requested_set: Set[str] = set()

    def register(self, function_name: str) -> None:
        if function_name in self._requested_set:
            return
        prefix = self._resolve_prefix(function_name)
        if prefix is None:
            raise CodegenBackendError(
                f"unknown scalar function requested: {function_name}"
            )
        self._ensure_header(prefix)
        self._requested.append(function_name)
        self._requested_set.add(function_name)

    def include_lines(self) -> List[str]:
        includes: List[str] = []
        seen: Set[str] = set()
        for prefix in self._header_order:
            info = self._headers.get(prefix)
            if info is None:
                continue
            for include in info.includes:
                if include in seen:
                    continue
                seen.add(include)
                includes.append(include)
        return includes

    def render(self) -> List[str]:
        if not self._requested:
            return []
        lines: List[str] = []
        function_lines: List[str] = []
        emitted: Set[str] = set()
        macro_invocations: Dict[str, List[str]] = {}
        needed_macro_defs: Dict[str, _ScalarHeaderInfo] = {}
        macro_order: List[str] = []

        def emit_function(name: str) -> None:
            if name in emitted:
                return
            prefix = self._resolve_prefix(name)
            if prefix is None:
                raise CodegenBackendError(
                    f"unknown scalar function dependency: {name}"
                )
            self._ensure_header(prefix)
            info = self._headers[prefix]
            if name in info.macro_functions:
                macro_name = info.macro_functions[name]
                macro_def = info.macro_defs.get(macro_name)
                if macro_def is None:
                    raise CodegenBackendError(
                        f"missing macro definition {macro_name} in {info.path}"
                    )
                deps = self._macro_dependencies(macro_def, name, info)
                for dep in sorted(deps):
                    emit_function(dep)
                macro_invocations.setdefault(macro_name, []).append(
                    name[len(info.prefix) :]
                )
                needed_macro_defs[macro_name] = info
                if macro_name not in macro_order:
                    macro_order.append(macro_name)
                emitted.add(name)
                return
            definition = info.functions.get(name)
            if definition is None:
                raise CodegenBackendError(
                    f"missing scalar definition for {name} in {info.path}"
                )
            for dep in sorted(info.function_deps.get(name, set())):
                emit_function(dep)
            function_lines.extend(definition)
            function_lines.append("")
            emitted.add(name)

        for name in self._requested:
            emit_function(name)

        for prefix in self._header_order:
            info = self._headers.get(prefix)
            if info is None:
                continue
            lines.extend(info.preamble)
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(function_lines)
        for macro_name in macro_order:
            info = needed_macro_defs[macro_name]
            macro_def = info.macro_defs[macro_name]
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend(macro_def)
            for suffix in macro_invocations.get(macro_name, []):
                lines.append(f"{macro_name}({suffix})")
            undef_line = info.macro_undefs.get(macro_name)
            if undef_line:
                lines.append(undef_line)
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _resolve_prefix(self, function_name: str) -> str | None:
        for prefix in self._header_order:
            if function_name.startswith(prefix):
                return prefix
        return None

    def _ensure_header(self, prefix: str) -> None:
        if prefix in self._headers:
            return
        header_name = self._prefix_to_header[prefix]
        path = self._headers_root / header_name
        self._headers[prefix] = self._parse_header(path, prefix)

    def _parse_header(self, path: Path, prefix: str) -> _ScalarHeaderInfo:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        guard_end_index = None
        for index, line in enumerate(raw_lines):
            if line.strip() == "#endif":
                guard_end_index = index
        excluded_lines: Set[int] = set()
        includes: List[str] = []
        preamble: List[str] = []
        macro_defs: Dict[str, List[str]] = {}
        macro_undefs: Dict[str, str] = {}
        guard_prefix = "REF_BACKEND_OPS_SCALAR_"
        index = 0
        while index < len(raw_lines):
            line = raw_lines[index]
            stripped = line.strip()
            if stripped.startswith("#include"):
                if "ops_scalar_" not in stripped:
                    includes.append(stripped)
                index += 1
                continue
            if stripped.startswith("#ifndef") and guard_prefix in stripped:
                excluded_lines.add(index)
                index += 1
                continue
            if stripped.startswith("#define") and guard_prefix in stripped:
                excluded_lines.add(index)
                index += 1
                continue
            if stripped == "#endif":
                if guard_end_index is not None and index == guard_end_index:
                    excluded_lines.add(index)
                    index += 1
                    continue
                preamble.append(stripped)
                index += 1
                continue
            if stripped.startswith("#define REF_") and "(" in stripped:
                macro_name = stripped.split()[1].split("(")[0]
                macro_lines = [line]
                excluded_lines.add(index)
                index += 1
                while index < len(raw_lines):
                    macro_line = raw_lines[index]
                    macro_lines.append(macro_line)
                    excluded_lines.add(index)
                    index += 1
                    if not macro_line.rstrip().endswith("\\"):
                        break
                macro_defs[macro_name] = macro_lines
                continue
            if stripped.startswith("#undef REF_"):
                macro_name = stripped.split()[1]
                macro_undefs[macro_name] = stripped
                excluded_lines.add(index)
                index += 1
                continue
            if stripped.startswith("#"):
                preamble.append(stripped)
                index += 1
                continue
            index += 1

        macro_functions: Dict[str, str] = {}
        for line in raw_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _MACRO_INVOCATION_PATTERN.match(stripped)
            if match:
                macro_name, suffix = match.groups()
                macro_functions[f"{prefix}{suffix}"] = macro_name

        functions = self._parse_functions(raw_lines, excluded_lines)
        function_deps = {
            name: self._extract_deps(name, definition)
            for name, definition in functions.items()
        }
        return _ScalarHeaderInfo(
            prefix=prefix,
            path=path,
            includes=includes,
            preamble=preamble,
            functions=functions,
            function_deps=function_deps,
            macro_defs=macro_defs,
            macro_undefs=macro_undefs,
            macro_functions=macro_functions,
        )

    def _parse_functions(
        self, raw_lines: List[str], excluded_lines: Set[int]
    ) -> Dict[str, List[str]]:
        functions: Dict[str, List[str]] = {}
        current_lines: List[str] = []
        current_name: str | None = None
        brace_count = 0
        for index, line in enumerate(raw_lines):
            if index in excluded_lines:
                continue
            if current_name is None:
                if line.lstrip().startswith("static inline"):
                    match = _FUNCTION_NAME_PATTERN.search(line)
                    if not match:
                        continue
                    current_name = match.group(1)
                    current_lines = [line]
                    brace_count = line.count("{") - line.count("}")
                    if brace_count == 0:
                        functions[current_name] = current_lines
                        current_name = None
                        current_lines = []
                continue
            current_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0:
                functions[current_name] = current_lines
                current_name = None
                current_lines = []
        return functions

    def _extract_deps(self, name: str, definition: Iterable[str]) -> Set[str]:
        deps = set(_REF_PREFIX_PATTERN.findall("\n".join(definition)))
        deps.discard(name)
        return deps

    def _macro_dependencies(
        self, macro_def: List[str], name: str, info: _ScalarHeaderInfo
    ) -> Set[str]:
        suffix = name[len(info.prefix) :]
        expanded = "\n".join(macro_def).replace("##name", suffix)
        deps = set(_REF_PREFIX_PATTERN.findall(expanded))
        deps.discard(name)
        return deps
