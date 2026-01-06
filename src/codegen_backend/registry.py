from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from codegen_backend.ops_registry import SUPPORTED_OPS
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class _TargetInfo:
    op_spec: _OpSpec
    inplace_arg_index: int | None


def _build_target_registry() -> Dict[object, _TargetInfo]:
    registry: Dict[object, _TargetInfo] = {}
    for spec in SUPPORTED_OPS.values():
        for target in spec.supported_targets:
            inplace_arg_index = (
                spec.inplace_arg_index if target in spec.inplace_targets else None
            )
            registry[target] = _TargetInfo(
                op_spec=spec, inplace_arg_index=inplace_arg_index
            )
    return registry


TARGET_REGISTRY = _build_target_registry()
TARGET_REGISTRY[torch.ops.aten.atan2.out] = _TargetInfo(
    op_spec=SUPPORTED_OPS["atan2"],
    inplace_arg_index=2,
)
