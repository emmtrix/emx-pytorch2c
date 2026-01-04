from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class _OpSpec:
    name: str
    kind: str
    symbol: str | None
    supported_targets: set
    inplace_targets: set = field(default_factory=set)
    inplace_arg_index: int | None = None


def _binary_spec(
    name: str,
    targets: Iterable[object],
    symbol: str | None,
    inplace_targets: Iterable[object] = (),
) -> _OpSpec:
    inplace_targets_set = set(inplace_targets)
    return _OpSpec(
        name=name,
        kind="binary",
        symbol=symbol,
        supported_targets=set(targets),
        inplace_targets=inplace_targets_set,
        inplace_arg_index=0 if inplace_targets_set else None,
    )


def _unary_spec(
    name: str, targets: Iterable[object], inplace_targets: Iterable[object] = ()
) -> _OpSpec:
    inplace_targets_set = set(inplace_targets)
    return _OpSpec(
        name=name,
        kind="unary",
        symbol=None,
        supported_targets=set(targets),
        inplace_targets=inplace_targets_set,
        inplace_arg_index=0 if inplace_targets_set else None,
    )
