from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from codegen_backend.specs import OpKind


@dataclass(frozen=True)
class KindHandlerRegistration:
    handler_cls: Type["OpKindHandler"]
    emitter_cls: Type["KindEmitter"]


def build_kind_handler_registry() -> Dict[OpKind, KindHandlerRegistration]:
    from codegen_backend.emitters.addbmm import AddbmmEmitter
    from codegen_backend.emitters.addmm import AddmmEmitter
    from codegen_backend.emitters.addmv import AddmvEmitter
    from codegen_backend.emitters.addr import AddrEmitter
    from codegen_backend.emitters.arange import ArangeEmitter
    from codegen_backend.emitters.argreduction import ArgReductionEmitter
    from codegen_backend.emitters.batch_norm import BatchNormEmitter
    from codegen_backend.emitters.cdist import CdistEmitter
    from codegen_backend.emitters.col2im import Col2imEmitter
    from codegen_backend.emitters.concat import ConcatEmitter
    from codegen_backend.emitters.conv1d import Conv1dEmitter
    from codegen_backend.emitters.conv2d import Conv2dEmitter
    from codegen_backend.emitters.cumsum import CumsumEmitter
    from codegen_backend.emitters.diagonal import DiagonalEmitter
    from codegen_backend.emitters.embedding import EmbeddingEmitter
    from codegen_backend.emitters.embedding_bag import EmbeddingBagEmitter
    from codegen_backend.emitters.empty_strided import EmptyStridedEmitter
    from codegen_backend.emitters.elementwise import ElementwiseEmitter
    from codegen_backend.emitters.flip import FlipEmitter
    from codegen_backend.emitters.gather import GatherEmitter
    from codegen_backend.emitters.linear import LinearEmitter
    from codegen_backend.emitters.matmul import MatmulEmitter
    from codegen_backend.emitters.pad import PadEmitter
    from codegen_backend.emitters.pdist import PdistEmitter
    from codegen_backend.emitters.pool1d import Pool1dEmitter
    from codegen_backend.emitters.pool2d import Pool2dEmitter
    from codegen_backend.emitters.pool2d_backward import Pool2dBackwardEmitter
    from codegen_backend.emitters.pool3d import Pool3dEmitter
    from codegen_backend.emitters.reduction import ReductionEmitter
    from codegen_backend.emitters.resize import ResizeEmitter
    from codegen_backend.emitters.softmax import SoftmaxEmitter
    from codegen_backend.emitters.view import ViewEmitter
    from codegen_backend.kinds import (
        AddbmmHandler,
        AddmmHandler,
        AddmvHandler,
        AddrHandler,
        ArangeHandler,
        ArgReductionHandler,
        BatchNormHandler,
        CdistHandler,
        Col2imHandler,
        ConcatHandler,
        Conv1dHandler,
        Conv2dHandler,
        CumsumHandler,
        DiagonalHandler,
        EmbeddingBagHandler,
        EmbeddingHandler,
        EmptyStridedHandler,
        ElementwiseHandler,
        FlipHandler,
        GatherHandler,
        LinearHandler,
        MatmulHandler,
        PadHandler,
        PdistHandler,
        Pool1dHandler,
        Pool2dBackwardHandler,
        Pool2dHandler,
        Pool3dHandler,
        ReductionHandler,
        ResizeHandler,
        SoftmaxHandler,
        ViewHandler,
    )

    return {
        OpKind.ARANGE: KindHandlerRegistration(ArangeHandler, ArangeEmitter),
        OpKind.BINARY: KindHandlerRegistration(ElementwiseHandler, ElementwiseEmitter),
        OpKind.UNARY: KindHandlerRegistration(ElementwiseHandler, ElementwiseEmitter),
        OpKind.WHERE: KindHandlerRegistration(ElementwiseHandler, ElementwiseEmitter),
        OpKind.FILL: KindHandlerRegistration(ElementwiseHandler, ElementwiseEmitter),
        OpKind.FLIP: KindHandlerRegistration(FlipHandler, FlipEmitter),
        OpKind.PAD: KindHandlerRegistration(PadHandler, PadEmitter),
        OpKind.VIEW: KindHandlerRegistration(ViewHandler, ViewEmitter),
        OpKind.RESIZE: KindHandlerRegistration(ResizeHandler, ResizeEmitter),
        OpKind.EMPTY_STRIDED: KindHandlerRegistration(
            EmptyStridedHandler, EmptyStridedEmitter
        ),
        OpKind.DIAGONAL: KindHandlerRegistration(DiagonalHandler, DiagonalEmitter),
        OpKind.REDUCTION: KindHandlerRegistration(ReductionHandler, ReductionEmitter),
        OpKind.ARG_REDUCTION: KindHandlerRegistration(
            ArgReductionHandler, ArgReductionEmitter
        ),
        OpKind.SOFTMAX: KindHandlerRegistration(SoftmaxHandler, SoftmaxEmitter),
        OpKind.CUMSUM: KindHandlerRegistration(CumsumHandler, CumsumEmitter),
        OpKind.EMBEDDING: KindHandlerRegistration(EmbeddingHandler, EmbeddingEmitter),
        OpKind.EMBEDDING_BAG: KindHandlerRegistration(
            EmbeddingBagHandler, EmbeddingBagEmitter
        ),
        OpKind.GATHER: KindHandlerRegistration(GatherHandler, GatherEmitter),
        OpKind.LINEAR: KindHandlerRegistration(LinearHandler, LinearEmitter),
        OpKind.CONCAT: KindHandlerRegistration(ConcatHandler, ConcatEmitter),
        OpKind.POOL2D: KindHandlerRegistration(Pool2dHandler, Pool2dEmitter),
        OpKind.POOL3D: KindHandlerRegistration(Pool3dHandler, Pool3dEmitter),
        OpKind.POOL2D_BACKWARD: KindHandlerRegistration(
            Pool2dBackwardHandler, Pool2dBackwardEmitter
        ),
        OpKind.POOL1D: KindHandlerRegistration(Pool1dHandler, Pool1dEmitter),
        OpKind.COL2IM: KindHandlerRegistration(Col2imHandler, Col2imEmitter),
        OpKind.BATCH_NORM: KindHandlerRegistration(BatchNormHandler, BatchNormEmitter),
        OpKind.PDIST: KindHandlerRegistration(PdistHandler, PdistEmitter),
        OpKind.CDIST: KindHandlerRegistration(CdistHandler, CdistEmitter),
        OpKind.CONV1D: KindHandlerRegistration(Conv1dHandler, Conv1dEmitter),
        OpKind.CONV2D: KindHandlerRegistration(Conv2dHandler, Conv2dEmitter),
        OpKind.ADDMM: KindHandlerRegistration(AddmmHandler, AddmmEmitter),
        OpKind.ADDBMM: KindHandlerRegistration(AddbmmHandler, AddbmmEmitter),
        OpKind.ADDMV: KindHandlerRegistration(AddmvHandler, AddmvEmitter),
        OpKind.ADDR: KindHandlerRegistration(AddrHandler, AddrEmitter),
        OpKind.MATMUL: KindHandlerRegistration(MatmulHandler, MatmulEmitter),
    }


__all__ = ["KindHandlerRegistration", "build_kind_handler_registry"]
