from __future__ import annotations

from dataclasses import dataclass

import pytest

import codegen_backend.groups.registry as registry_module
from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.registry import (
    GroupRegistry,
    get_group_registry,
    register_group,
)
from codegen_backend.registry import build_target_registry
from codegen_backend.specs import _binary_spec, _OpSpec


@dataclass(frozen=True)
class DummyGroup(OperatorGroupDefinition):
    name: str = "dummy"

    def build_supported_ops(self):
        return {}

    def build_target_registry(self, supported_ops):
        return {}

    def build_analyzers(self, supported_ops, target_registry):
        return ()


@dataclass(frozen=True)
class SpecGroup(OperatorGroupDefinition):
    name: str
    op_spec: _OpSpec

    def build_supported_ops(self):
        return {self.op_spec.name: self.op_spec}

    def build_target_registry(self, supported_ops):
        return build_target_registry(supported_ops)

    def build_analyzers(self, supported_ops, target_registry):
        return ()


def test_register_group_adds_custom_group(monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTERED_GROUPS", {})
    monkeypatch.setattr(registry_module, "_GROUP_REGISTRY", None)
    monkeypatch.setattr(registry_module, "_DEFAULT_GROUPS_LOADED", True)
    monkeypatch.setattr(registry_module, "_ENTRY_POINTS_LOADED", True)

    register_group(DummyGroup())

    registry = get_group_registry()
    group_names = [group.name for group in registry.groups]

    assert "dummy" in group_names


def test_merged_target_registry_raises_on_colliding_targets():
    target = "aten.test"
    spec_left = _binary_spec("left", [target], symbol="+")
    spec_right = _binary_spec("right", [target], symbol="-")
    registry = GroupRegistry(
        groups=[
            SpecGroup(name="left_group", op_spec=spec_left),
            SpecGroup(name="right_group", op_spec=spec_right),
        ]
    )

    with pytest.raises(ValueError, match="aten.test.*left_group/left.*right_group/right"):
        registry.merged_target_registry()


def test_merged_target_registry_raises_on_duplicate_op_spec_instances():
    target = "aten.same"
    spec_left = _binary_spec("same", [target], symbol="+")
    spec_right = _binary_spec("same", [target], symbol="+")
    registry = GroupRegistry(
        groups=[
            SpecGroup(name="left_group", op_spec=spec_left),
            SpecGroup(name="right_group", op_spec=spec_right),
        ]
    )

    with pytest.raises(ValueError, match="aten.same.*left_group/same.*right_group/same"):
        registry.merged_target_registry()
