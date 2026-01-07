from __future__ import annotations

from typing import Dict
from codegen_backend.emitters.arange import ArangeEmitter
from codegen_backend.emitters.argreduction import ArgReductionEmitter
from codegen_backend.emitters.base import KindEmitter
from codegen_backend.emitters.elementwise import ElementwiseEmitter
from codegen_backend.emitters.flip import FlipEmitter
from codegen_backend.emitters.legacy import (
    AddbmmEmitter,
    AddmmEmitter,
    AddmvEmitter,
    AddrEmitter,
    BatchNormEmitter,
    CdistEmitter,
    Col2imEmitter,
    ConcatEmitter,
    Conv1dEmitter,
    Conv2dEmitter,
    CumsumEmitter,
    DiagonalEmitter,
    EmbeddingBagEmitter,
    EmbeddingEmitter,
    EmptyStridedEmitter,
    GatherEmitter,
    PdistEmitter,
    Pool1dEmitter,
    Pool2dBackwardEmitter,
    Pool2dEmitter,
    Pool3dEmitter,
    SoftmaxEmitter,
)
from codegen_backend.emitters.matmul import MatmulEmitter
from codegen_backend.emitters.pad import PadEmitter
from codegen_backend.emitters.reduction import ReductionEmitter
from codegen_backend.emitters.resize import ResizeEmitter
from codegen_backend.emitters.view import ViewEmitter
from codegen_backend.specs import OpKind


def build_kind_emitters() -> Dict[OpKind, KindEmitter]:
    elementwise_emitter = ElementwiseEmitter()
    elementwise_kinds = {
        OpKind.BINARY,
        OpKind.FILL,
        OpKind.UNARY,
        OpKind.WHERE,
    }
    kind_emitters: Dict[OpKind, KindEmitter] = {
        OpKind.EMPTY_STRIDED: EmptyStridedEmitter(),
        OpKind.ARANGE: ArangeEmitter(),
        OpKind.ARG_REDUCTION: ArgReductionEmitter(),
        OpKind.DIAGONAL: DiagonalEmitter(),
        OpKind.SOFTMAX: SoftmaxEmitter(),
        OpKind.CUMSUM: CumsumEmitter(),
        OpKind.CONCAT: ConcatEmitter(),
        OpKind.FLIP: FlipEmitter(),
        OpKind.EMBEDDING: EmbeddingEmitter(),
        OpKind.EMBEDDING_BAG: EmbeddingBagEmitter(),
        OpKind.GATHER: GatherEmitter(),
        OpKind.MATMUL: MatmulEmitter(),
        OpKind.ADDMM: AddmmEmitter(),
        OpKind.ADDBMM: AddbmmEmitter(),
        OpKind.ADDMV: AddmvEmitter(),
        OpKind.ADDR: AddrEmitter(),
        OpKind.CONV1D: Conv1dEmitter(),
        OpKind.CONV2D: Conv2dEmitter(),
        OpKind.POOL1D: Pool1dEmitter(),
        OpKind.POOL2D: Pool2dEmitter(),
        OpKind.POOL3D: Pool3dEmitter(),
        OpKind.POOL2D_BACKWARD: Pool2dBackwardEmitter(),
        OpKind.COL2IM: Col2imEmitter(),
        OpKind.BATCH_NORM: BatchNormEmitter(),
        OpKind.PDIST: PdistEmitter(),
        OpKind.CDIST: CdistEmitter(),
        OpKind.PAD: PadEmitter(),
        OpKind.REDUCTION: ReductionEmitter(),
        OpKind.VIEW: ViewEmitter(),
        OpKind.RESIZE: ResizeEmitter(),
    }
    for kind in elementwise_kinds:
        kind_emitters[kind] = elementwise_emitter
    return {kind: kind_emitters[kind] for kind in OpKind}
