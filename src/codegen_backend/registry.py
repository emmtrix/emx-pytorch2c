from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class _TargetInfo:
    op_spec: _OpSpec
    inplace_arg_index: int | None


def build_target_registry(
    supported_ops: Mapping[str, _OpSpec],
) -> Dict[object, _TargetInfo]:
    registry: Dict[object, _TargetInfo] = {}
    for spec in supported_ops.values():
        for target in spec.supported_targets:
            inplace_arg_index = (
                spec.inplace_arg_index if target in spec.inplace_targets else None
            )
            registry[target] = _TargetInfo(
                op_spec=spec, inplace_arg_index=inplace_arg_index
            )
    return registry


def build_target_registry_from_groups() -> Dict[object, _TargetInfo]:
    from codegen_backend.groups.registry import get_group_registry

    return get_group_registry().merged_target_registry()


__all__ = ["_TargetInfo", "build_target_registry", "build_target_registry_from_groups"]
