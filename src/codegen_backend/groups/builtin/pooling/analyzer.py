from __future__ import annotations

from typing import Mapping

from codegen_backend.groups.analysis import RegistryGroupAnalyzer
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


class PoolingAnalyzer(RegistryGroupAnalyzer):
    def __init__(
        self,
        supported_ops: Mapping[str, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> None:
        super().__init__(
            name="pooling",
            supported_ops=supported_ops,
            target_registry=target_registry,
        )


__all__ = ["PoolingAnalyzer"]
