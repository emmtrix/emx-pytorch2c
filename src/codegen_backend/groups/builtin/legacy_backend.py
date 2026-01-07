from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from codegen_backend.emitters.addr import AddrEmitter
from codegen_backend.emitters.arange import ArangeEmitter
from codegen_backend.emitters.argreduction import ArgReductionEmitter
from codegen_backend.emitters.concat import ConcatEmitter
from codegen_backend.emitters.conv1d import Conv1dEmitter
from codegen_backend.emitters.conv2d import Conv2dEmitter
from codegen_backend.emitters.elementwise import ElementwiseEmitter
from codegen_backend.emitters.embedding import EmbeddingEmitter
from codegen_backend.emitters.embedding_bag import EmbeddingBagEmitter
from codegen_backend.emitters.empty_strided import EmptyStridedEmitter
from codegen_backend.emitters.matmul import MatmulEmitter
from codegen_backend.emitters.pool1d import Pool1dEmitter
from codegen_backend.emitters.pool2d import Pool2dEmitter
from codegen_backend.emitters.pool2d_backward import Pool2dBackwardEmitter
from codegen_backend.emitters.pool3d import Pool3dEmitter
from codegen_backend.emitters.reduction import ReductionEmitter
from codegen_backend.emitters.softmax import SoftmaxEmitter
from codegen_backend.kinds import HandlerContext, OpKindHandler, build_kind_handlers
from codegen_backend.ops_registry import SUPPORTED_OPS
from codegen_backend.registry import TARGET_REGISTRY, _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class LegacyBackendGroup:
    name: str = "legacy"

    def kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]:
        from codegen_backend import backend as backend_mod

        kind_handlers = build_kind_handlers(context)
        elementwise_emitter = ElementwiseEmitter()
        kind_handlers.update(
            {
                OpKind.BINARY: backend_mod._BackendElementwiseHandler(
                    context, elementwise_emitter, "binary"
                ),
                OpKind.UNARY: backend_mod._BackendElementwiseHandler(
                    context, elementwise_emitter, "unary"
                ),
                OpKind.WHERE: backend_mod._BackendElementwiseHandler(
                    context, elementwise_emitter, "where"
                ),
                OpKind.FILL: backend_mod._BackendElementwiseHandler(
                    context, elementwise_emitter, "fill"
                ),
                OpKind.ARANGE: backend_mod._BackendArangeHandler(
                    context, ArangeEmitter()
                ),
                OpKind.CONCAT: backend_mod._BackendConcatHandler(
                    context, ConcatEmitter()
                ),
                OpKind.EMPTY_STRIDED: backend_mod._BackendEmptyStridedHandler(
                    context, EmptyStridedEmitter()
                ),
                OpKind.REDUCTION: backend_mod._BackendReductionHandler(
                    context, ReductionEmitter()
                ),
                OpKind.ARG_REDUCTION: backend_mod._BackendArgReductionHandler(
                    context, ArgReductionEmitter()
                ),
                OpKind.POOL1D: backend_mod._BackendPool1dHandler(
                    context, Pool1dEmitter()
                ),
                OpKind.POOL2D: backend_mod._BackendPool2dHandler(
                    context, Pool2dEmitter()
                ),
                OpKind.POOL3D: backend_mod._BackendPool3dHandler(
                    context, Pool3dEmitter()
                ),
                OpKind.POOL2D_BACKWARD: backend_mod._BackendPool2dBackwardHandler(
                    context, Pool2dBackwardEmitter()
                ),
                OpKind.SOFTMAX: backend_mod._BackendSoftmaxHandler(
                    context, SoftmaxEmitter()
                ),
                OpKind.EMBEDDING: backend_mod._BackendEmbeddingHandler(
                    context, EmbeddingEmitter()
                ),
                OpKind.EMBEDDING_BAG: backend_mod._BackendEmbeddingBagHandler(
                    context, EmbeddingBagEmitter()
                ),
                OpKind.CONV1D: backend_mod._BackendConv1dHandler(
                    context, Conv1dEmitter()
                ),
                OpKind.CONV2D: backend_mod._BackendConv2dHandler(
                    context, Conv2dEmitter()
                ),
                OpKind.MATMUL: backend_mod._BackendMatmulHandler(
                    context, MatmulEmitter()
                ),
                OpKind.ADDR: backend_mod._BackendAddrHandler(
                    context, AddrEmitter()
                ),
            }
        )
        return kind_handlers

    def supported_ops(self) -> Mapping[object, _OpSpec]:
        return SUPPORTED_OPS

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return TARGET_REGISTRY


OperatorGroup = LegacyBackendGroup


__all__ = ["LegacyBackendGroup", "OperatorGroup"]
