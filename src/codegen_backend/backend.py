from typing import Callable, Dict, List, Sequence

import torch
import torch.fx

from codegen_backend.graph import _GenericGraph
from codegen_backend.groups.registry import get_group_registry
from codegen_backend.compiler import Compiler
from codegen_backend.emitter import Emitter
from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.graph_builder import GraphBuilder
from codegen_backend.parser import Parser
from codegen_backend.services import GraphAnalysisService
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.templates import get_template_env


class CodegenBackend:
    def __init__(
        self,
        *,
        group_registry: object | None = None,
        analysis_service: GraphAnalysisService | None = None,
        templates_env: object | None = None,
    ) -> None:
        self.group_registry = (
            group_registry if group_registry is not None else get_group_registry()
        )
        self.templates_env = (
            templates_env if templates_env is not None else get_template_env()
        )
        self._supported_ops: Dict[str, _OpSpec] | None = None
        self._target_registry: Dict[object, "_TargetInfo"] | None = None
        self._kind_handlers: Dict[OpKind, "OpKindHandler"] | None = None
        self._kind_handler_registrations: Dict[
            OpKind, KindHandlerRegistration
        ] | None = None
        self._analysis_service = (
            analysis_service
            if analysis_service is not None
            else GraphAnalysisService(lambda: self.kind_handlers)
        )
        self._parser = Parser(
            kind_handlers=lambda: self.kind_handlers,
            target_registry=lambda: self.target_registry,
        )
        self._graph_builder = GraphBuilder(
            group_registry=lambda: self.group_registry,
            kind_handlers=lambda: self.kind_handlers,
            parser=self._parser,
        )
        self._emitter = Emitter(
            templates_env=lambda: self.templates_env,
            kind_handlers=lambda: self.kind_handlers,
            kind_handler_registrations=lambda: self.kind_handler_registrations,
        )
        self._compiler = Compiler(self._graph_builder, self._emitter)
        self._context_provider = self.group_registry.build_context_provider(self)

    @property
    def supported_ops(self) -> Dict[str, _OpSpec]:
        if self._supported_ops is None:
            self._supported_ops = self.group_registry.merged_supported_ops()
        return self._supported_ops

    @property
    def target_registry(self) -> Dict[object, "_TargetInfo"]:
        if self._target_registry is None:
            self._target_registry = self.group_registry.merged_target_registry()
        return self._target_registry

    @property
    def kind_handlers(self) -> Dict[OpKind, "OpKindHandler"]:
        if self._kind_handlers is None:
            self._kind_handlers = self.group_registry.build_kind_handlers(
                self._context_provider
            )
        return self._kind_handlers

    @property
    def kind_handler_registrations(self) -> Dict[OpKind, KindHandlerRegistration]:
        if self._kind_handler_registrations is None:
            self._kind_handler_registrations = (
                self.group_registry.merged_kind_handler_registrations()
            )
        return self._kind_handler_registrations

    @property
    def analysis_service(self) -> GraphAnalysisService:
        return self._analysis_service

    def get_generic_source(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> str:
        return self._compiler.get_source(gm, example_inputs)

    def codegen_generic_backend(
        self, gm: torch.fx.GraphModule, example_inputs: List[object]
    ) -> Callable[..., torch.Tensor]:
        return self._compiler.compile_graph(gm, example_inputs)

    def _analyze_generic_graph(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> _GenericGraph:
        return self._graph_builder.build(gm, example_inputs)

    def _write_generic_source(self, graph: _GenericGraph) -> str:
        return self._emitter.emit(graph)


_DEFAULT_BACKEND = CodegenBackend()


def get_generic_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> str:
    return _DEFAULT_BACKEND.get_generic_source(gm, example_inputs)


def codegen_generic_backend(
    gm: torch.fx.GraphModule, example_inputs: List[object]
) -> Callable[..., torch.Tensor]:
    return _DEFAULT_BACKEND.codegen_generic_backend(gm, example_inputs)
