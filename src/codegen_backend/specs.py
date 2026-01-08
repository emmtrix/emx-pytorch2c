from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class OpKind(str, Enum):
    BINARY = "binary"
    UNARY = "unary"
    FILL = "fill"
    VIEW = "view"
    WHERE = "where"
    FLIP = "flip"
    ARG_REDUCTION = "arg_reduction"
    REDUCTION = "reduction"
    ARANGE = "arange"
    SOFTMAX = "softmax"
    CUMSUM = "cumsum"
    CONCAT = "concat"
    DIAGONAL = "diagonal"
    ADDMM = "addmm"
    ADDBMM = "addbmm"
    ADDMV = "addmv"
    ADDR = "addr"
    MATMUL = "matmul"
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    POOL1D = "pool1d"
    POOL2D = "pool2d"
    POOL3D = "pool3d"
    POOL2D_BACKWARD = "pool2d_backward"
    EMBEDDING = "embedding"
    EMBEDDING_BAG = "embedding_bag"
    EMBEDDING_DENSE_BACKWARD = "embedding_dense_backward"
    GATHER = "gather"
    INDEX_PUT = "index_put"
    INDEX_SELECT = "index_select"
    BATCH_NORM = "batch_norm"
    PDIST = "pdist"
    CDIST = "cdist"
    PAD = "pad"
    EMPTY_STRIDED = "empty_strided"
    RESIZE = "resize"
    COL2IM = "col2im"
    MASKED_SCATTER = "masked_scatter"
    DROPOUT = "dropout"


@dataclass(frozen=True)
class _OpSpec:
    name: str
    kind: OpKind
    symbol: str | None
    supported_targets: set
    inplace_targets: set = field(default_factory=set)
    inplace_arg_index: int | None = None


def _binary_spec(
    name: str,
    targets: Iterable[object],
    symbol: str | None,
    inplace_targets: Iterable[object] = (),
    inplace_arg_index: int | None = None,
) -> _OpSpec:
    inplace_targets_set = set(inplace_targets)
    if inplace_targets_set and inplace_arg_index is None:
        inplace_arg_index = 0
    return _OpSpec(
        name=name,
        kind=OpKind.BINARY,
        symbol=symbol,
        supported_targets=set(targets),
        inplace_targets=inplace_targets_set,
        inplace_arg_index=inplace_arg_index,
    )


def _unary_spec(
    name: str,
    targets: Iterable[object],
    inplace_targets: Iterable[object] = (),
    inplace_arg_index: int | None = None,
) -> _OpSpec:
    inplace_targets_set = set(inplace_targets)
    if inplace_targets_set and inplace_arg_index is None:
        inplace_arg_index = 0
    return _OpSpec(
        name=name,
        kind=OpKind.UNARY,
        symbol=None,
        supported_targets=set(targets),
        inplace_targets=inplace_targets_set,
        inplace_arg_index=inplace_arg_index,
    )
