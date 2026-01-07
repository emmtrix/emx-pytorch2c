from __future__ import annotations

from typing import Dict
from codegen_backend.emitters.arange import ArangeEmitter
from codegen_backend.emitters.argreduction import ArgReductionEmitter
from codegen_backend.emitters.batch_norm import BatchNormEmitter
from codegen_backend.emitters.base import KindEmitter
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
from codegen_backend.emitters.addbmm import AddbmmEmitter
from codegen_backend.emitters.addmm import AddmmEmitter
from codegen_backend.emitters.addmv import AddmvEmitter
from codegen_backend.emitters.addr import AddrEmitter
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
