from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

from codegen_backend.specs import OpKind, _OpSpec, _binary_spec, _unary_spec

# Dev note: when introducing a new OpKind with a centralized handler, wire the
# handler/emitter in codegen_backend/emitters/registry.py and, if build-time
# validation/parsing lives in the backend, add a backend OpKindHandler subclass
# in codegen_backend/backend.py and register it in _KIND_HANDLERS.


def _flatten_targets(targets: tuple[object, ...]) -> list[object]:
    if len(targets) == 1 and isinstance(targets[0], (list, tuple, set)):
        return list(targets[0])
    return list(targets)


class _OpBuilder:
    def __init__(
        self,
        registry: "_OpRegistry",
        name: str,
        kind: OpKind,
        symbol: str | None = None,
    ) -> None:
        self._registry = registry
        self._name = name
        self._kind = kind
        self._symbol = symbol
        self._targets: list[object] = []
        self._inplace_targets: list[object] = []
        self._inplace_arg_index: int | None = None

    def targets(self, *targets: object) -> "_OpBuilder":
        self._targets = _flatten_targets(targets)
        return self

    def inplace(self, *targets: object, arg_index: int | None = None) -> "_OpBuilder":
        self._inplace_targets = _flatten_targets(targets)
        self._inplace_arg_index = arg_index
        return self

    def build(self) -> _OpSpec:
        if self._kind == OpKind.BINARY:
            spec = _binary_spec(
                self._name,
                self._targets,
                self._symbol,
                inplace_targets=self._inplace_targets,
                inplace_arg_index=self._inplace_arg_index,
            )
        elif self._kind == OpKind.UNARY:
            spec = _unary_spec(
                self._name,
                self._targets,
                inplace_targets=self._inplace_targets,
                inplace_arg_index=self._inplace_arg_index,
            )
        else:
            spec = _OpSpec(
                name=self._name,
                kind=self._kind,
                symbol=self._symbol,
                supported_targets=set(self._targets),
                inplace_targets=set(self._inplace_targets),
                inplace_arg_index=self._inplace_arg_index,
            )
        self._registry._add(spec)
        return spec


def validate_op_spec(spec: _OpSpec) -> None:
    if not isinstance(spec.kind, OpKind):
        expected = ", ".join(kind.value for kind in OpKind)
        raise ValueError(
            "Invalid op spec for "
            f"'{spec.name}': kind={spec.kind!r}; expected one of {expected}."
        )
    if not spec.supported_targets:
        raise ValueError(
            "Invalid op spec for "
            f"'{spec.name}': supported_targets is empty; "
            "expected at least one target."
        )
    if spec.kind != OpKind.BINARY and spec.symbol is not None:
        raise ValueError(
            "Invalid op spec for "
            f"'{spec.name}': symbol is only valid for binary ops; "
            f"got kind={spec.kind.value}."
        )
    if bool(spec.inplace_targets) != (spec.inplace_arg_index is not None):
        raise ValueError(
            "Invalid op spec for "
            f"'{spec.name}': inplace_targets and inplace_arg_index must "
            "be set together."
        )
    if spec.inplace_arg_index is not None and spec.kind in {
        OpKind.BINARY,
        OpKind.UNARY,
    }:
        if spec.inplace_arg_index != 0:
            raise ValueError(
                "Invalid op spec for "
                f"'{spec.name}': inplace_arg_index must be 0 for "
                f"{spec.kind.value} ops; got {spec.inplace_arg_index}."
            )
    missing_inplace = spec.inplace_targets - spec.supported_targets
    if missing_inplace:
        raise ValueError(
            "Invalid op spec for "
            f"'{spec.name}': inplace_targets must be a subset of "
            "supported_targets; missing "
            f"{sorted(missing_inplace, key=repr)}."
        )


class _OpRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, _OpSpec] = {}
        self._allowed_duplicate_targets: Counter[object] = Counter()

    @contextmanager
    def allow_duplicate_targets(self, *targets: object) -> "_OpRegistry":
        flattened_targets = _flatten_targets(targets)
        self._allowed_duplicate_targets.update(flattened_targets)
        try:
            yield self
        finally:
            for target in flattened_targets:
                self._allowed_duplicate_targets[target] -= 1
                if self._allowed_duplicate_targets[target] <= 0:
                    del self._allowed_duplicate_targets[target]

    def register_unary(self, name: str) -> _OpBuilder:
        return _OpBuilder(self, name, OpKind.UNARY)

    def register_binary(self, name: str, symbol: str | None = None) -> _OpBuilder:
        return _OpBuilder(self, name, OpKind.BINARY, symbol=symbol)

    def register_op(
        self, name: str, kind: OpKind, symbol: str | None = None
    ) -> _OpBuilder:
        return _OpBuilder(self, name, kind, symbol=symbol)

    def _add(self, spec: _OpSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate op spec registered: {spec.name}")
        validate_op_spec(spec)
        self._specs[spec.name] = spec

    def build(self) -> dict[str, _OpSpec]:
        _validate_registry(
            self._specs, allow_duplicate_targets=set(self._allowed_duplicate_targets)
        )
        return dict(self._specs)


def _validate_registry(
    specs: dict[str, _OpSpec],
    allow_duplicate_targets: set[object],
) -> None:
    seen_targets: dict[object, str] = {}
    for spec in specs.values():
        for target in spec.supported_targets:
            if target in allow_duplicate_targets:
                continue
            if target in seen_targets and seen_targets[target] != spec.name:
                raise ValueError(
                    "Duplicate target registered for ops "
                    f"'{seen_targets[target]}' and '{spec.name}'."
                )
            seen_targets[target] = spec.name


__all__ = ["_OpRegistry", "validate_op_spec"]
