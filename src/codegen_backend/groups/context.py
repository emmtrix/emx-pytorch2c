from __future__ import annotations

from typing import List, TYPE_CHECKING

import torch

from codegen_backend.backend_helpers import _kernel_inputs
from codegen_backend.groups.builtin.reductions.args import ReductionsArgParser
from codegen_backend.graph import _OpNode
from codegen_backend.kinds import (
    ConvContext,
    ElementwiseContext,
    EmbeddingContext,
    HandlerContextProvider,
    PoolingContext,
    ReductionContext,
    TensorContext,
)
from codegen_backend.services import GraphAnalysisService

if TYPE_CHECKING:
    from codegen_backend.backend import CodegenBackend


class _BackendKernelContext:
    def __init__(self, backend: "CodegenBackend") -> None:
        self._backend = backend

    @property
    def analysis_service(self) -> GraphAnalysisService:
        return self._backend.analysis_service

    def kernel_inputs(self, op_node: _OpNode) -> List[torch.fx.Node]:
        return _kernel_inputs(op_node)


class _BackendElementwiseContext(_BackendKernelContext):
    pass


class _BackendReductionContext(_BackendKernelContext):
    def __init__(self, backend: "CodegenBackend") -> None:
        super().__init__(backend)
        self._arg_parser = ReductionsArgParser(backend.analysis_service)

    @property
    def arg_parser(self) -> ReductionsArgParser:
        return self._arg_parser


class _BackendPoolingContext(_BackendKernelContext):
    pass


class _BackendConvContext(_BackendKernelContext):
    pass


class _BackendEmbeddingContext(_BackendKernelContext):
    pass


class _BackendTensorContext(_BackendKernelContext):
    pass


class BackendContextProvider(HandlerContextProvider):
    def __init__(self, backend: "CodegenBackend") -> None:
        self._elementwise = _BackendElementwiseContext(backend)
        self._reductions = _BackendReductionContext(backend)
        self._pooling = _BackendPoolingContext(backend)
        self._conv = _BackendConvContext(backend)
        self._embedding = _BackendEmbeddingContext(backend)
        self._tensor = _BackendTensorContext(backend)

    @property
    def elementwise(self) -> ElementwiseContext:
        return self._elementwise

    @property
    def reductions(self) -> ReductionContext:
        return self._reductions

    @property
    def pooling(self) -> PoolingContext:
        return self._pooling

    @property
    def conv(self) -> ConvContext:
        return self._conv

    @property
    def embedding(self) -> EmbeddingContext:
        return self._embedding

    @property
    def tensor(self) -> TensorContext:
        return self._tensor


__all__ = ["BackendContextProvider"]
