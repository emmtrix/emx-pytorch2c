from __future__ import annotations

import numbers
import operator
from typing import Mapping, Sequence, Tuple

import torch
import torch.fx

from codegen_backend.analysis_helpers import (
    error_expected_tensor,
    error_kwarg_specified_once,
)
from codegen_backend.dtypes import _CODEGEN_DTYPES, _CodegenDType
from codegen_backend.errors import CodegenBackendError
from codegen_backend.parsing.common import parse_constant_int


def parse_gather_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object, object]:
    if len(node.args) > 4:
        raise CodegenBackendError("codegen gather expects at most four inputs")
    if not node.args:
        raise CodegenBackendError("codegen gather expects input, dim, and index")
    input_arg = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else None
    index = node.args[2] if len(node.args) > 2 else None
    sparse_grad = node.args[3] if len(node.args) > 3 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise error_kwarg_specified_once("gather", "dim")
            dim = node.kwargs["dim"]
        if "index" in node.kwargs:
            if index is not None:
                raise error_kwarg_specified_once("gather", "index")
            index = node.kwargs["index"]
        if "sparse_grad" in node.kwargs:
            if sparse_grad is not None:
                raise error_kwarg_specified_once("gather", "sparse_grad")
            sparse_grad = node.kwargs["sparse_grad"]
        extra = set(node.kwargs) - {"dim", "index", "sparse_grad"}
        if extra:
            raise CodegenBackendError(
                f"codegen gather got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None or index is None:
        raise CodegenBackendError("codegen gather expects dim and index arguments")
    if sparse_grad is None:
        sparse_grad = False
    return input_arg, dim, index, sparse_grad


def parse_masked_scatter_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object]:
    if len(node.args) > 3:
        raise CodegenBackendError(
            "codegen masked_scatter expects at most three inputs"
        )
    if not node.args:
        raise CodegenBackendError(
            "codegen masked_scatter expects input, mask, and source"
        )
    input_arg = node.args[0]
    mask = node.args[1] if len(node.args) > 1 else None
    source = node.args[2] if len(node.args) > 2 else None
    if node.kwargs:
        if "mask" in node.kwargs:
            if mask is not None:
                raise error_kwarg_specified_once("masked_scatter", "mask")
            mask = node.kwargs["mask"]
        if "source" in node.kwargs:
            if source is not None:
                raise error_kwarg_specified_once("masked_scatter", "source")
            source = node.kwargs["source"]
        extra = set(node.kwargs) - {"mask", "source"}
        if extra:
            raise CodegenBackendError(
                "codegen masked_scatter got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    if mask is None or source is None:
        raise CodegenBackendError(
            "codegen masked_scatter expects mask and source arguments"
        )
    return input_arg, mask, source


def parse_index_put_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object, object]:
    if len(node.args) > 4:
        raise CodegenBackendError(
            "codegen index_put expects at most four inputs"
        )
    if len(node.args) < 3:
        raise CodegenBackendError(
            "codegen index_put expects input, indices, and values"
        )
    input_arg = node.args[0]
    indices = node.args[1] if len(node.args) > 1 else None
    values = node.args[2] if len(node.args) > 2 else None
    accumulate = node.args[3] if len(node.args) > 3 else None
    if node.kwargs:
        if "indices" in node.kwargs:
            if indices is not None:
                raise error_kwarg_specified_once("index_put", "indices")
            indices = node.kwargs["indices"]
        if "values" in node.kwargs:
            if values is not None:
                raise error_kwarg_specified_once("index_put", "values")
            values = node.kwargs["values"]
        if "accumulate" in node.kwargs:
            if accumulate is not None:
                raise error_kwarg_specified_once("index_put", "accumulate")
            accumulate = node.kwargs["accumulate"]
        extra = set(node.kwargs) - {"indices", "values", "accumulate"}
        if extra:
            raise CodegenBackendError(
                f"codegen index_put got unexpected kwargs: {sorted(extra)}"
            )
    if indices is None or values is None:
        raise CodegenBackendError(
            "codegen index_put expects indices and values arguments"
        )
    if accumulate is None:
        accumulate = False
    return input_arg, indices, values, accumulate


def parse_index_select_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object]:
    if len(node.args) > 3:
        raise CodegenBackendError(
            "codegen index_select expects at most three inputs"
        )
    if not node.args:
        raise CodegenBackendError(
            "codegen index_select expects input, dim, and index"
        )
    input_arg = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else None
    index = node.args[2] if len(node.args) > 2 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise error_kwarg_specified_once("index_select", "dim")
            dim = node.kwargs["dim"]
        if "index" in node.kwargs:
            if index is not None:
                raise error_kwarg_specified_once("index_select", "index")
            index = node.kwargs["index"]
        extra = set(node.kwargs) - {"dim", "index"}
        if extra:
            raise CodegenBackendError(
                "codegen index_select got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    if dim is None or index is None:
        raise CodegenBackendError(
            "codegen index_select expects dim and index arguments"
        )
    return input_arg, dim, index


def parse_select_scatter_args(
    node: torch.fx.Node,
) -> Tuple[object, object, object, object]:
    if len(node.args) > 4:
        raise CodegenBackendError(
            "codegen select_scatter expects at most four inputs"
        )
    if not node.args:
        raise CodegenBackendError(
            "codegen select_scatter expects input, src, dim, and index"
        )
    input_arg = node.args[0]
    src = node.args[1] if len(node.args) > 1 else None
    dim = node.args[2] if len(node.args) > 2 else None
    index = node.args[3] if len(node.args) > 3 else None
    if node.kwargs:
        if "src" in node.kwargs:
            if src is not None:
                raise error_kwarg_specified_once("select_scatter", "src")
            src = node.kwargs["src"]
        if "dim" in node.kwargs:
            if dim is not None:
                raise error_kwarg_specified_once("select_scatter", "dim")
            dim = node.kwargs["dim"]
        if "index" in node.kwargs:
            if index is not None:
                raise error_kwarg_specified_once("select_scatter", "index")
            index = node.kwargs["index"]
        extra = set(node.kwargs) - {"src", "dim", "index"}
        if extra:
            raise CodegenBackendError(
                "codegen select_scatter got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    if src is None or dim is None or index is None:
        raise CodegenBackendError(
            "codegen select_scatter expects src, dim, and index arguments"
        )
    return input_arg, src, dim, index


def parse_addmm_like_scalar(
    op_name: str, name: str, value: object | None
) -> float:
    if value is None:
        return 1.0
    if isinstance(value, torch.fx.Node):
        meta_value = value.meta.get("val") or value.meta.get("example_value")
        if meta_value is None:
            raise CodegenBackendError(
                f"codegen {op_name} expects {name} to be a number"
            )
        return parse_addmm_like_scalar(op_name, name, meta_value)
    if isinstance(value, bool):
        raise CodegenBackendError(
            f"codegen {op_name} expects {name} to be a number"
        )
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CodegenBackendError(
                f"codegen {op_name} expects {name} to be a number"
            )
        return float(value.item())
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects {name} to be a number"
            ) from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects {name} to be a number"
        ) from exc


def parse_addmm_like_args(
    op_name: str, node: torch.fx.Node
) -> Tuple[Sequence[torch.fx.Node], float, float]:
    if len(node.args) < 3:
        raise CodegenBackendError(f"codegen {op_name} expects at least three inputs")
    if len(node.args) > 5:
        raise CodegenBackendError(f"codegen {op_name} expects at most five inputs")
    input_node, mat1_node, mat2_node = node.args[:3]
    beta = node.args[3] if len(node.args) > 3 else None
    alpha = node.args[4] if len(node.args) > 4 else None
    if node.kwargs:
        if "beta" in node.kwargs:
            if beta is not None:
                raise error_kwarg_specified_once(op_name, "beta")
            beta = node.kwargs["beta"]
        if "alpha" in node.kwargs:
            if alpha is not None:
                raise error_kwarg_specified_once(op_name, "alpha")
            alpha = node.kwargs["alpha"]
        extra = set(node.kwargs) - {"beta", "alpha"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    return (
        (input_node, mat1_node, mat2_node),
        parse_addmm_like_scalar(op_name, "alpha", alpha),
        parse_addmm_like_scalar(op_name, "beta", beta),
    )


def parse_linear_args(
    op_name: str, node: torch.fx.Node
) -> Tuple[torch.fx.Node, torch.fx.Node, object]:
    if len(node.args) < 2:
        raise CodegenBackendError(f"codegen {op_name} expects at least two inputs")
    if len(node.args) > 3:
        raise CodegenBackendError(f"codegen {op_name} expects at most three inputs")
    input_node, weight_node = node.args[:2]
    bias = node.args[2] if len(node.args) > 2 else None
    if node.kwargs:
        if "bias" in node.kwargs:
            if bias is not None:
                raise error_kwarg_specified_once(op_name, "bias")
            bias = node.kwargs["bias"]
        extra = set(node.kwargs) - {"bias"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    return input_node, weight_node, bias


def parse_concat_args(
    node: torch.fx.Node,
) -> Tuple[Sequence[torch.fx.Node], int]:
    if not node.args:
        raise CodegenBackendError("codegen cat expects a tensor list input")
    if len(node.args) > 2:
        raise CodegenBackendError("codegen cat expects at most two inputs")
    tensors_arg = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise error_kwarg_specified_once("cat", "dim")
            dim = node.kwargs["dim"]
        extra = set(node.kwargs) - {"dim"}
        if extra:
            raise CodegenBackendError(
                f"codegen cat got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None:
        dim_value = 0
    else:
        dim_value = parse_constant_int("cat", "dim", dim)
    if not isinstance(tensors_arg, (list, tuple)) or not tensors_arg:
        raise CodegenBackendError("codegen cat expects a non-empty tensor list input")
    for item in tensors_arg:
        if not isinstance(item, torch.fx.Node):
            raise error_expected_tensor("cat")
    return list(tensors_arg), dim_value


def parse_split_with_sizes_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, Tuple[int, ...], int]:
    if len(node.args) < 2:
        raise CodegenBackendError(
            "codegen split_with_sizes expects input and split sizes"
        )
    if len(node.args) > 3:
        raise CodegenBackendError(
            "codegen split_with_sizes expects at most three inputs"
        )
    input_arg = node.args[0]
    split_sizes = node.args[1]
    dim = node.args[2] if len(node.args) > 2 else None
    if node.kwargs:
        if "split_sizes" in node.kwargs:
            if split_sizes is not None:
                raise error_kwarg_specified_once("split_with_sizes", "split_sizes")
            split_sizes = node.kwargs["split_sizes"]
        if "dim" in node.kwargs:
            if dim is not None:
                raise error_kwarg_specified_once("split_with_sizes", "dim")
            dim = node.kwargs["dim"]
        extra = set(node.kwargs) - {"split_sizes", "dim"}
        if extra:
            raise CodegenBackendError(
                "codegen split_with_sizes got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    if dim is None:
        dim_value = 0
    else:
        dim_value = parse_constant_int("split_with_sizes", "dim", dim)
    if isinstance(split_sizes, torch.Size):
        split_sizes_value = tuple(split_sizes)
    elif isinstance(split_sizes, torch.Tensor):
        if split_sizes.dim() != 1:
            raise CodegenBackendError(
                "codegen split_with_sizes expects split_sizes to be 1D"
            )
        split_sizes_value = tuple(split_sizes.tolist())
    elif isinstance(split_sizes, (tuple, list)):
        split_sizes_value = tuple(split_sizes)
    else:
        raise CodegenBackendError(
            "codegen split_with_sizes expects split_sizes to be a sequence"
        )
    try:
        split_sizes_value = tuple(
            int(operator.index(item)) for item in split_sizes_value
        )
    except TypeError as exc:
        raise CodegenBackendError(
            "codegen split_with_sizes expects split_sizes to be a sequence of ints"
        ) from exc
    if not split_sizes_value:
        raise CodegenBackendError(
            "codegen split_with_sizes expects at least one split size"
        )
    return input_arg, split_sizes_value, dim_value


def validate_split_with_sizes(
    input_arg: object,
    split_sizes: Tuple[int, ...],
    dim: int,
    shapes: Mapping[torch.fx.Node, Tuple[int, ...]],
) -> Tuple[torch.fx.Node, Tuple[int, ...], int]:
    if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
        raise CodegenBackendError(
            "codegen split_with_sizes expects a tensor input"
        )
    input_shape = shapes[input_arg]
    if dim < 0:
        dim += len(input_shape)
    if dim < 0 or dim >= len(input_shape):
        raise CodegenBackendError(
            "codegen split_with_sizes dim is out of range"
        )
    if any(size < 0 for size in split_sizes):
        raise CodegenBackendError(
            "codegen split_with_sizes expects non-negative split sizes"
        )
    if sum(split_sizes) != input_shape[dim]:
        raise CodegenBackendError(
            "codegen split_with_sizes expects split sizes to sum to the input size"
        )
    return input_arg, input_shape, dim


def parse_diagonal_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, int, int]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 4:
        raise CodegenBackendError(f"codegen {op_name} expects at most four inputs")
    offset = node.args[1] if len(node.args) > 1 else 0
    dim1 = node.args[2] if len(node.args) > 2 else 0
    dim2 = node.args[3] if len(node.args) > 3 else 1
    if node.kwargs:
        if "offset" in node.kwargs:
            if len(node.args) > 1:
                raise error_kwarg_specified_once(op_name, "offset")
            offset = node.kwargs["offset"]
        if "dim1" in node.kwargs:
            if len(node.args) > 2:
                raise error_kwarg_specified_once(op_name, "dim1")
            dim1 = node.kwargs["dim1"]
        if "dim2" in node.kwargs:
            if len(node.args) > 3:
                raise error_kwarg_specified_once(op_name, "dim2")
            dim2 = node.kwargs["dim2"]
        extra = set(node.kwargs) - {"offset", "dim1", "dim2"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    offset_value = parse_constant_int(op_name, "offset", offset)
    dim1_value = parse_constant_int(op_name, "dim1", dim1)
    dim2_value = parse_constant_int(op_name, "dim2", dim2)
    rank = len(input_shape)
    if rank < 2:
        raise CodegenBackendError(f"codegen {op_name} expects input rank >= 2")
    if dim1_value < 0:
        dim1_value += rank
    if dim2_value < 0:
        dim2_value += rank
    if dim1_value < 0 or dim1_value >= rank:
        raise CodegenBackendError(f"codegen {op_name} dim1 is out of range")
    if dim2_value < 0 or dim2_value >= rank:
        raise CodegenBackendError(f"codegen {op_name} dim2 is out of range")
    if dim1_value == dim2_value:
        raise CodegenBackendError(f"codegen {op_name} expects dim1 != dim2")
    return offset_value, dim1_value, dim2_value


def parse_cumsum_args(
    op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
) -> Tuple[int, torch.dtype | None]:
    if not node.args:
        raise CodegenBackendError(f"codegen {op_name} expects one input")
    if len(node.args) > 3:
        raise CodegenBackendError(
            f"codegen {op_name} expects at most three inputs (self, dim, dtype)"
        )
    dim = node.args[1] if len(node.args) > 1 else None
    dtype = node.args[2] if len(node.args) > 2 else None
    if node.kwargs:
        if "dim" in node.kwargs:
            if dim is not None:
                raise error_kwarg_specified_once(op_name, "dim")
            dim = node.kwargs["dim"]
        if "dtype" in node.kwargs:
            if dtype is not None:
                raise error_kwarg_specified_once(op_name, "dtype")
            dtype = node.kwargs["dtype"]
        extra = set(node.kwargs) - {"dim", "dtype"}
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    if dim is None:
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    if isinstance(dim, (tuple, list)):
        raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
    dim_value = parse_constant_int(op_name, "dim", dim)
    rank = len(input_shape)
    if rank == 0:
        if dim_value not in (-1, 0):
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        dim_value = 0
    if dim_value < 0:
        dim_value += rank
    if dim_value < 0 or dim_value >= rank:
        raise CodegenBackendError(f"codegen {op_name} dim is out of range")
    if dtype is not None:
        if isinstance(dtype, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.float64, torch.int8, torch.uint8, torch.uint32, torch.int32, or torch.bool"
            )
        if dtype not in (
            torch.float32,
            torch.float64,
            torch.int8,
            torch.uint8,
            torch.uint32,
            torch.int32,
            torch.bool,
        ):
            raise CodegenBackendError(
                f"codegen {op_name} expects dtype to be torch.float32, torch.float64, torch.int8, torch.uint8, torch.uint32, torch.int32, or torch.bool"
            )
    return dim_value, dtype


def parse_arange_dtype(
    op_name: str,
    dtype: torch.dtype | None,
    dtype_info: _CodegenDType | None,
    start: float | int | bool,
    end: float | int | bool,
    step: float | int | bool,
) -> _CodegenDType:
    if dtype is None:
        if any(
            not isinstance(value, numbers.Integral)
            for value in (start, end, step)
        ):
            dtype = torch.get_default_dtype()
        else:
            dtype = torch.int32
    if dtype is torch.bool:
        raise CodegenBackendError(
            f"codegen {op_name} supports only numeric dtypes"
        )
    dtype_spec = _CODEGEN_DTYPES.get(dtype)
    if dtype_spec is None:
        supported = ", ".join(
            f"torch.{supported_dtype.name}"
            for supported_dtype in _CODEGEN_DTYPES
            if supported_dtype is not torch.bool
        )
        raise CodegenBackendError(
            f"codegen {op_name} supports only {supported}"
        )
    if (
        dtype_info is not None
        and dtype_spec.torch_dtype is not dtype_info.torch_dtype
    ):
        raise CodegenBackendError(
            f"codegen {op_name} expects dtype to match the graph dtype"
        )
    return dtype_spec


def parse_resize_size(op_name: str, size_value: object) -> Tuple[int, ...]:
    if isinstance(size_value, torch.fx.Node):
        raise CodegenBackendError(
            f"codegen {op_name} expects size to be a constant"
        )
    if isinstance(size_value, torch.Size):
        size = tuple(size_value)
    elif isinstance(size_value, (list, tuple)):
        size = size_value
    else:
        raise CodegenBackendError(
            f"codegen {op_name} expects size to be a sequence"
        )
    try:
        return tuple(int(operator.index(item)) for item in size)
    except TypeError as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects size to be a sequence of ints"
        ) from exc


def parse_empty_strided_stride(
    op_name: str, stride_value: object, *, size: Sequence[int]
) -> Tuple[int, ...]:
    if isinstance(stride_value, torch.fx.Node):
        raise CodegenBackendError(
            f"codegen {op_name} expects stride to be a constant"
        )
    if isinstance(stride_value, torch.Size):
        stride = tuple(stride_value)
    elif isinstance(stride_value, (list, tuple)):
        stride = stride_value
    else:
        raise CodegenBackendError(
            f"codegen {op_name} expects stride to be a sequence"
        )
    try:
        stride_values = tuple(int(operator.index(item)) for item in stride)
    except TypeError as exc:
        raise CodegenBackendError(
            f"codegen {op_name} expects stride to be a sequence of ints"
        ) from exc
    if len(stride_values) != len(size):
        raise CodegenBackendError(
            f"codegen {op_name} expects stride to match size rank"
        )
    return stride_values


__all__ = [
    "parse_addmm_like_args",
    "parse_addmm_like_scalar",
    "parse_arange_dtype",
    "parse_concat_args",
    "parse_cumsum_args",
    "parse_diagonal_args",
    "parse_empty_strided_stride",
    "parse_gather_args",
    "parse_index_put_args",
    "parse_masked_scatter_args",
    "parse_linear_args",
    "parse_resize_size",
    "parse_split_with_sizes_args",
]
