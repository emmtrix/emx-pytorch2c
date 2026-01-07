from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from codegen_backend.specs import OpKind


@dataclass(frozen=True)
class KindHandlerRegistration:
    handler_cls: Type["OpKindHandler"]
    emitter_cls: Type["KindEmitter"]


def build_kind_handler_registry() -> Dict[OpKind, KindHandlerRegistration]:
    """Return the default kind handler registry."""
    return {}


__all__ = ["KindHandlerRegistration", "build_kind_handler_registry"]
