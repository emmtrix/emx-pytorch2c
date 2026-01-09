from __future__ import annotations

import math
import numbers
import operator
from typing import Dict, List, Sequence, Tuple

import torch
import torch.fx

from codegen_backend import shape_utils
from codegen_backend.analysis_helpers import (
    channels_last_3d_strides,
    channels_last_strides,
    error_expected_tensor,
    error_kwarg_specified_once,
    normalize_as_strided_sequence,
    normalize_flip_dims,
    parse_constant_int,
    resolve_scalar_arg,
)
from codegen_backend.c_types import _normalize_scalar_value
from codegen_backend.dtypes import (
    _CodegenDType,
    _EMBEDDING_INDEX_DTYPES,
    _INTEGER_CODEGEN_DTYPES,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.groups.builtin.tensor.parsing import (
    parse_addmm_like_args,
    parse_arange_dtype,
    parse_concat_args,
    parse_cumsum_args,
    parse_diagonal_args,
    parse_empty_strided_stride,
    parse_gather_args,
    parse_index_put_args,
    parse_index_select_args,
    parse_linear_args,
    parse_masked_scatter_args,
    parse_scatter_src_args,
    parse_scatter_value_args,
    parse_select_scatter_args,
    parse_sort_args,
    parse_resize_size,
)
from codegen_backend.graph import _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.services import GraphAnalysisService
from codegen_backend.specs import _OpSpec


def validate_addmm_like_scalars(
    op_name: str, dtype: torch.dtype, alpha: float, beta: float
) -> None:
    if dtype in _INTEGER_CODEGEN_DTYPES or dtype is torch.bool:
        for name, value in (("alpha", alpha), ("beta", beta)):
            if isinstance(value, float) and not math.isfinite(value):
                raise CodegenBackendError(
                    f"codegen {op_name} expects {name} to be finite for integral tensors"
                )


class TensorOpBuilder:
    def __init__(
        self,
        analysis_service: GraphAnalysisService,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
    ) -> None:
        self._analysis_service = analysis_service
        self._shapes = shapes
        self._strides = strides
        self._dtypes = dtypes
        self._scalar_values = scalar_values

    def _finalize_node(
        self,
        node: torch.fx.Node,
        op_node: _OpNode,
        dtype_info: _CodegenDType,
        input_shapes: Sequence[Tuple[int, ...]],
        *,
        inplace_input: int | None = None,
        output_strides: Tuple[int, ...] | None = None,
    ) -> _OpNode:
        output_shape = self._analysis_service.infer_output_shape(
            op_node, input_shapes
        )
        op_node.output_shape = output_shape
        self._shapes[node] = output_shape
        self._dtypes[node] = dtype_info.torch_dtype
        if output_strides is None:
            if inplace_input is not None:
                output_strides = self._strides[op_node.inputs[inplace_input]]
            else:
                output_strides = _contiguous_strides(output_shape)
        self._strides[node] = output_strides
        return op_node

    def build_flip(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects two inputs"
            )
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen flip expects inputs to share the graph dtype"
            )
        input_shape = self._shapes[input_arg]
        dims = node.args[1] if len(node.args) > 1 else None
        dims_value = normalize_flip_dims(op_spec.name, dims, len(input_shape))
        if node.kwargs:
            extra = set(node.kwargs)
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={"dims": dims_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def _normalize_size_arg(
        self, op_name: str, size: object
    ) -> Tuple[int, ...]:
        if isinstance(size, torch.fx.Node):
            if size in self._scalar_values:
                size = self._scalar_values[size]
            else:
                meta_value = size.meta.get("val")
                if meta_value is None:
                    meta_value = size.meta.get("example_value")
                if meta_value is None:
                    raise CodegenBackendError(
                        f"codegen {op_name} expects size to be a constant"
                    )
                size = meta_value
        if isinstance(size, torch.Size):
            size = tuple(size)
        if isinstance(size, torch.Tensor):
            if size.numel() == 1:
                size = (size.item(),)
            else:
                if size.dim() != 1:
                    raise CodegenBackendError(
                        f"codegen {op_name} expects size to be a 1D tensor or scalar"
                    )
                size = tuple(size.tolist())
        if isinstance(size, (tuple, list)):
            size_tuple = tuple(size)
        elif isinstance(size, numbers.Integral):
            size_tuple = (int(size),)
        else:
            raise CodegenBackendError(
                f"codegen {op_name} expects size to be an int or a tuple of ints"
            )
        try:
            size_tuple = tuple(int(operator.index(item)) for item in size_tuple)
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects size to be an int or a tuple of ints"
            ) from exc
        if any(dim < 0 for dim in size_tuple):
            raise CodegenBackendError(
                f"codegen {op_name} expects non-negative size values"
            )
        return size_tuple

    def build_random(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if not node.args:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects size arguments"
            )
        size_arg = node.args[0] if len(node.args) == 1 else node.args
        size_tuple = self._normalize_size_arg(op_spec.name, size_arg)
        kwargs = dict(node.kwargs)
        extra = set(kwargs) - {
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "requires_grad",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
            )
        if kwargs.get("layout") is not None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects layout to be None"
            )
        device = kwargs.get("device")
        if device is not None and device != "cpu" and device != torch.device("cpu"):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects device to be None or cpu"
            )
        pin_memory = kwargs.get("pin_memory")
        if pin_memory not in (None, False):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects pin_memory to be False"
            )
        requires_grad = kwargs.get("requires_grad")
        if requires_grad not in (None, False):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects requires_grad to be False"
            )
        dtype_value = kwargs.get("dtype")
        if dtype_value is None:
            dtype_value = dtype_info.torch_dtype
        if dtype_value not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64 tensors"
            )
        if dtype_value is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects dtype to match the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[],
            output_shape=(),
            params={"size": size_tuple},
        )
        op_node.output_shape = size_tuple
        self._shapes[node] = size_tuple
        self._dtypes[node] = dtype_value
        self._strides[node] = _contiguous_strides(size_tuple)
        return op_node

    def build_randperm(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if not node.args:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects a size argument"
            )
        size_tuple = self._normalize_size_arg(op_spec.name, node.args[0])
        if len(size_tuple) != 1:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects a single size value"
            )
        kwargs = dict(node.kwargs)
        extra = set(kwargs) - {
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "requires_grad",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
            )
        if kwargs.get("layout") is not None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects layout to be None"
            )
        device = kwargs.get("device")
        if device is not None and device != "cpu" and device != torch.device("cpu"):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects device to be None or cpu"
            )
        pin_memory = kwargs.get("pin_memory")
        if pin_memory not in (None, False):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects pin_memory to be False"
            )
        requires_grad = kwargs.get("requires_grad")
        if requires_grad not in (None, False):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects requires_grad to be False"
            )
        dtype_value = kwargs.get("dtype")
        if dtype_value is None:
            dtype_value = dtype_info.torch_dtype
        supported_dtypes = {
            torch.float32,
            torch.float64,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if dtype_value not in supported_dtypes:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only float32, float64, int8, "
                "uint8, int16, int32, or int64 tensors"
            )
        if dtype_value is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects dtype to match the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[],
            output_shape=(),
            params={"size": size_tuple},
        )
        return self._finalize_node(node, op_node, dtype_info, [])

    def _normalize_normalized_shape(
        self, op_name: str, normalized_shape: object, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        if isinstance(normalized_shape, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects normalized_shape to be a constant"
            )
        if isinstance(normalized_shape, torch.Size):
            normalized_shape = tuple(normalized_shape)
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape_tuple = (int(normalized_shape),)
        elif isinstance(normalized_shape, (tuple, list)):
            try:
                normalized_shape_tuple = tuple(
                    int(operator.index(item)) for item in normalized_shape
                )
            except TypeError as exc:
                raise CodegenBackendError(
                    f"codegen {op_name} expects normalized_shape to be a tuple of ints"
                ) from exc
        else:
            raise CodegenBackendError(
                f"codegen {op_name} expects normalized_shape to be a tuple of ints"
            )
        if not normalized_shape_tuple:
            raise CodegenBackendError(
                f"codegen {op_name} expects normalized_shape to be non-empty"
            )
        if any(dim <= 0 for dim in normalized_shape_tuple):
            raise CodegenBackendError(
                f"codegen {op_name} expects normalized_shape values to be positive"
            )
        if len(normalized_shape_tuple) > len(input_shape):
            raise CodegenBackendError(
                f"codegen {op_name} expects normalized_shape rank <= input rank"
            )
        if tuple(input_shape[-len(normalized_shape_tuple) :]) != normalized_shape_tuple:
            raise CodegenBackendError(
                f"codegen {op_name} expects normalized_shape to match input shape tail"
            )
        return normalized_shape_tuple

    def build_native_layer_norm(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 5:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects five inputs"
            )
        if node.kwargs:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positional args only"
            )
        input_arg, normalized_shape, weight, bias, eps = node.args[:5]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        if self._dtypes[input_arg] not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        input_shape = self._shapes[input_arg]
        normalized_shape_tuple = self._normalize_normalized_shape(
            op_spec.name, normalized_shape, input_shape
        )
        weight_node = None
        bias_node = None
        if weight is not None:
            if not isinstance(weight, torch.fx.Node) or weight not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            weight_node = weight
            if self._shapes[weight_node] != normalized_shape_tuple:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects weight shape to match normalized_shape"
                )
        if bias is not None:
            if not isinstance(bias, torch.fx.Node) or bias not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            bias_node = bias
            if self._shapes[bias_node] != normalized_shape_tuple:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects bias shape to match normalized_shape"
                )
        from codegen_backend.emitters.base import _is_contiguous

        if not _is_contiguous(input_shape, self._strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input"
            )
        for name, node_arg in (("weight", weight_node), ("bias", bias_node)):
            if node_arg is None:
                continue
            if not _is_contiguous(self._shapes[node_arg], self._strides[node_arg]):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} requires contiguous {name}"
                )
            if self._dtypes[node_arg] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to match graph dtype"
                )
        eps_value = float(resolve_scalar_arg(op_spec.name, eps, self._scalar_values))
        inputs = [input_arg]
        if weight_node is not None:
            inputs.append(weight_node)
        if bias_node is not None:
            inputs.append(bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=inputs,
            output_shape=(),
            params={
                "normalized_shape": normalized_shape_tuple,
                "eps": eps_value,
                "has_weight": weight_node is not None,
                "has_bias": bias_node is not None,
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [self._shapes[arg] for arg in inputs]
        )

    def build_native_layer_norm_backward(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 8:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects eight inputs"
            )
        if node.kwargs:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positional args only"
            )
        (
            grad_output,
            input_arg,
            normalized_shape,
            mean,
            rstd,
            weight,
            bias,
            output_mask,
        ) = node.args[:8]
        for arg in (grad_output, input_arg, mean, rstd):
            if not isinstance(arg, torch.fx.Node) or arg not in self._shapes:
                raise error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        for arg in (grad_output, input_arg, mean, rstd):
            if self._dtypes[arg] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects inputs to share the graph dtype"
                )
        input_shape = self._shapes[input_arg]
        if self._shapes[grad_output] != input_shape:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects grad_output to match input shape"
            )
        normalized_shape_tuple = self._normalize_normalized_shape(
            op_spec.name, normalized_shape, input_shape
        )
        expected_stat_shape = input_shape[: -len(normalized_shape_tuple)] + (
            1,
        ) * len(normalized_shape_tuple)
        if self._shapes[mean] != expected_stat_shape or self._shapes[rstd] != expected_stat_shape:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects mean and rstd to match input shape with normalized dims set to 1"
            )
        weight_node = None
        bias_node = None
        if weight is not None:
            if not isinstance(weight, torch.fx.Node) or weight not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            weight_node = weight
            if self._shapes[weight_node] != normalized_shape_tuple:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects weight shape to match normalized_shape"
                )
        if bias is not None:
            if not isinstance(bias, torch.fx.Node) or bias not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            bias_node = bias
            if self._shapes[bias_node] != normalized_shape_tuple:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects bias shape to match normalized_shape"
                )
        if isinstance(output_mask, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects output_mask to be constant"
            )
        if not isinstance(output_mask, (tuple, list)) or len(output_mask) != 3:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects output_mask to be a bool[3]"
            )
        output_mask_tuple = tuple(bool(item) for item in output_mask)
        from codegen_backend.emitters.base import _is_contiguous

        for name, arg in (
            ("grad_output", grad_output),
            ("input", input_arg),
            ("mean", mean),
            ("rstd", rstd),
        ):
            if not _is_contiguous(self._shapes[arg], self._strides[arg]):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} requires contiguous {name}"
                )
        for name, node_arg in (("weight", weight_node), ("bias", bias_node)):
            if node_arg is None:
                continue
            if not _is_contiguous(self._shapes[node_arg], self._strides[node_arg]):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} requires contiguous {name}"
                )
            if self._dtypes[node_arg] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to match graph dtype"
                )
        inputs = [grad_output, input_arg, mean, rstd]
        if weight_node is not None:
            inputs.append(weight_node)
        if bias_node is not None:
            inputs.append(bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=inputs,
            output_shape=(),
            params={
                "normalized_shape": normalized_shape_tuple,
                "has_weight": weight_node is not None,
                "has_bias": bias_node is not None,
                "output_mask": output_mask_tuple,
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [self._shapes[arg] for arg in inputs]
        )

    def build_native_group_norm(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 8:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects eight inputs"
            )
        if node.kwargs:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positional args only"
            )
        (
            input_arg,
            weight,
            bias,
            n_arg,
            c_arg,
            hxw_arg,
            group_arg,
            eps,
        ) = node.args[:8]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        if self._dtypes[input_arg] not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        input_shape = self._shapes[input_arg]
        if len(input_shape) < 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs with rank >= 2"
            )
        n_value = int(resolve_scalar_arg(op_spec.name, n_arg, self._scalar_values))
        c_value = int(resolve_scalar_arg(op_spec.name, c_arg, self._scalar_values))
        hxw_value = int(resolve_scalar_arg(op_spec.name, hxw_arg, self._scalar_values))
        group_value = int(
            resolve_scalar_arg(op_spec.name, group_arg, self._scalar_values)
        )
        if n_value != input_shape[0] or c_value != input_shape[1]:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects N and C to match input shape"
            )
        spatial = 1
        for dim in input_shape[2:]:
            spatial *= dim
        if spatial != hxw_value:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects HxW to match input spatial size"
            )
        if group_value <= 0 or c_value % group_value != 0:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects groups to divide channels"
            )
        weight_node = None
        bias_node = None
        if weight is not None:
            if not isinstance(weight, torch.fx.Node) or weight not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            weight_node = weight
            if self._shapes[weight_node] != (c_value,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects weight shape to match channels"
                )
        if bias is not None:
            if not isinstance(bias, torch.fx.Node) or bias not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            bias_node = bias
            if self._shapes[bias_node] != (c_value,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects bias shape to match channels"
                )
        from codegen_backend.emitters.base import _is_contiguous

        if not _is_contiguous(input_shape, self._strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input"
            )
        for name, node_arg in (("weight", weight_node), ("bias", bias_node)):
            if node_arg is None:
                continue
            if not _is_contiguous(self._shapes[node_arg], self._strides[node_arg]):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} requires contiguous {name}"
                )
            if self._dtypes[node_arg] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to match graph dtype"
                )
        eps_value = float(resolve_scalar_arg(op_spec.name, eps, self._scalar_values))
        inputs = [input_arg]
        if weight_node is not None:
            inputs.append(weight_node)
        if bias_node is not None:
            inputs.append(bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=inputs,
            output_shape=(),
            params={
                "groups": group_value,
                "eps": eps_value,
                "has_weight": weight_node is not None,
                "has_bias": bias_node is not None,
                "N": n_value,
                "C": c_value,
                "HxW": hxw_value,
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [self._shapes[arg] for arg in inputs]
        )

    def build_native_group_norm_backward(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 10:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects ten inputs"
            )
        if node.kwargs:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positional args only"
            )
        (
            grad_output,
            input_arg,
            mean,
            rstd,
            weight,
            n_arg,
            c_arg,
            hxw_arg,
            group_arg,
            output_mask,
        ) = node.args[:10]
        for arg in (grad_output, input_arg, mean, rstd):
            if not isinstance(arg, torch.fx.Node) or arg not in self._shapes:
                raise error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        for arg in (grad_output, input_arg, mean, rstd):
            if self._dtypes[arg] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects inputs to share the graph dtype"
                )
        input_shape = self._shapes[input_arg]
        if self._shapes[grad_output] != input_shape:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects grad_output to match input shape"
            )
        n_value = int(resolve_scalar_arg(op_spec.name, n_arg, self._scalar_values))
        c_value = int(resolve_scalar_arg(op_spec.name, c_arg, self._scalar_values))
        hxw_value = int(resolve_scalar_arg(op_spec.name, hxw_arg, self._scalar_values))
        group_value = int(
            resolve_scalar_arg(op_spec.name, group_arg, self._scalar_values)
        )
        if n_value != input_shape[0] or c_value != input_shape[1]:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects N and C to match input shape"
            )
        spatial = 1
        for dim in input_shape[2:]:
            spatial *= dim
        if spatial != hxw_value:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects HxW to match input spatial size"
            )
        if group_value <= 0 or c_value % group_value != 0:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects groups to divide channels"
            )
        if self._shapes[mean] != (n_value, group_value) or self._shapes[rstd] != (
            n_value,
            group_value,
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects mean and rstd to have shape (N, group)"
            )
        weight_node = None
        if weight is not None:
            if not isinstance(weight, torch.fx.Node) or weight not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            weight_node = weight
            if self._shapes[weight_node] != (c_value,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects weight shape to match channels"
                )
        if isinstance(output_mask, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects output_mask to be constant"
            )
        if not isinstance(output_mask, (tuple, list)) or len(output_mask) != 3:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects output_mask to be a bool[3]"
            )
        output_mask_tuple = tuple(bool(item) for item in output_mask)
        from codegen_backend.emitters.base import _is_contiguous

        for name, arg in (
            ("grad_output", grad_output),
            ("input", input_arg),
            ("mean", mean),
            ("rstd", rstd),
        ):
            if not _is_contiguous(self._shapes[arg], self._strides[arg]):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} requires contiguous {name}"
                )
        if weight_node is not None:
            if not _is_contiguous(
                self._shapes[weight_node], self._strides[weight_node]
            ):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} requires contiguous weight"
                )
            if self._dtypes[weight_node] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects weight to match graph dtype"
                )
        inputs = [grad_output, input_arg, mean, rstd]
        if weight_node is not None:
            inputs.append(weight_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=inputs,
            output_shape=(),
            params={
                "groups": group_value,
                "has_weight": weight_node is not None,
                "N": n_value,
                "C": c_value,
                "HxW": hxw_value,
                "output_mask": output_mask_tuple,
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [self._shapes[arg] for arg in inputs]
        )

    def build_batch_norm(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        has_training_flag = op_spec.name == "_native_batch_norm_legit"
        expected_inputs = 8 if has_training_flag else 7
        if len(node.args) < expected_inputs:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects {expected_inputs} inputs"
            )
        if node.kwargs:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positional args only"
            )
        if has_training_flag:
            (
                input_arg,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            ) = node.args[:8]
        else:
            (
                input_arg,
                weight,
                bias,
                running_mean,
                running_var,
                momentum,
                eps,
            ) = node.args[:7]
            training = False
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if (
            not isinstance(running_mean, torch.fx.Node)
            or running_mean not in self._shapes
        ):
            raise error_expected_tensor(op_spec.name)
        if (
            not isinstance(running_var, torch.fx.Node)
            or running_var not in self._shapes
        ):
            raise error_expected_tensor(op_spec.name)
        weight_node = None
        bias_node = None
        if weight is not None:
            if not isinstance(weight, torch.fx.Node) or weight not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            weight_node = weight
        if bias is not None:
            if not isinstance(bias, torch.fx.Node) or bias not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            bias_node = bias
        if dtype_info.torch_dtype not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        if self._dtypes[input_arg] not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
            )
        input_shape = self._shapes[input_arg]
        if len(input_shape) < 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects at least 2D inputs"
            )
        from codegen_backend.emitters.base import _is_contiguous

        if not _is_contiguous(input_shape, self._strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input"
            )
        channels = input_shape[1]
        for stat_arg, name in (
            (running_mean, "running_mean"),
            (running_var, "running_var"),
        ):
            stat_shape = self._shapes[stat_arg]
            if stat_shape != (channels,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} shape to match channels"
                )
            if not _is_contiguous(stat_shape, self._strides[stat_arg]):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} requires contiguous stats"
                )
            if self._dtypes[stat_arg] not in (torch.float32, torch.float64):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
                )
        if weight_node is not None:
            if self._shapes[weight_node] != (channels,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects weight shape to match channels"
                )
            if self._dtypes[weight_node] not in (torch.float32, torch.float64):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
                )
        if bias_node is not None:
            if self._shapes[bias_node] != (channels,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects bias shape to match channels"
                )
            if self._dtypes[bias_node] not in (torch.float32, torch.float64):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32 or torch.float64"
                )
        try:
            training_value = self._analysis_service.resolve_scalar_arg(
                op_spec.name, training, self._scalar_values
            )
        except (TypeError, ValueError) as exc:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects training to be a boolean"
            ) from exc
        if isinstance(training_value, numbers.Real) and not isinstance(
            training_value, bool
        ):
            if float(training_value) not in (0.0, 1.0):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects training to be a boolean"
                )
            training_value = bool(training_value)
        if not isinstance(training_value, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects training to be a boolean"
            )
        try:
            momentum_value = float(
                self._analysis_service.resolve_scalar_arg(
                    op_spec.name, momentum, self._scalar_values
                )
            )
        except (TypeError, ValueError) as exc:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects momentum to be a float"
            ) from exc
        try:
            eps_value = float(
                self._analysis_service.resolve_scalar_arg(
                    op_spec.name, eps, self._scalar_values
                )
            )
        except (TypeError, ValueError) as exc:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects eps to be a float"
            ) from exc
        inputs = [input_arg, running_mean, running_var]
        if weight_node is not None:
            inputs.append(weight_node)
        if bias_node is not None:
            inputs.append(bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=inputs,
            output_shape=(),
            inplace_input=None,
            params={
                "eps": eps_value,
                "momentum": momentum_value,
                "training": training_value,
                "has_weight": weight_node is not None,
                "has_bias": bias_node is not None,
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_pdist(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 1:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        if len(node.args) > 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects at most two inputs"
            )
        input_arg = node.args[0]
        p = node.args[1] if len(node.args) > 1 else 2.0
        if node.kwargs:
            extra = set(node.kwargs)
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen pdist expects inputs to share the graph dtype"
            )
        if isinstance(p, torch.fx.Node):
            raise CodegenBackendError("codegen pdist expects p to be a number")
        if not isinstance(p, (int, float)):
            raise CodegenBackendError("codegen pdist expects p to be a number")
        p_value = float(p)
        if math.isinf(p_value) or math.isnan(p_value):
            raise CodegenBackendError("codegen pdist expects p to be finite")
        input_shape = self._shapes[input_arg]
        if len(input_shape) != 2:
            raise CodegenBackendError("codegen pdist expects a 2D input")
        if input_shape[0] < 2:
            raise CodegenBackendError("codegen pdist expects input rows >= 2")
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={"p": p_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_cdist(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 2:
            raise CodegenBackendError(f"codegen {op_spec.name} expects two inputs")
        if len(node.args) > 4:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects at most four inputs"
            )
        x1 = node.args[0]
        x2 = node.args[1]
        p = node.args[2] if len(node.args) > 2 else 2.0
        compute_mode = node.args[3] if len(node.args) > 3 else None
        if node.kwargs:
            extra = set(node.kwargs)
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        if not isinstance(x1, torch.fx.Node) or x1 not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(x2, torch.fx.Node) or x2 not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[x1] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen cdist expects inputs to share the graph dtype"
            )
        if self._dtypes[x2] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen cdist expects inputs to share the graph dtype"
            )
        if isinstance(p, torch.fx.Node):
            raise CodegenBackendError("codegen cdist expects p to be a number")
        if not isinstance(p, (int, float)):
            raise CodegenBackendError("codegen cdist expects p to be a number")
        p_value = float(p)
        if math.isnan(p_value):
            raise CodegenBackendError("codegen cdist expects p to be a number")
        if p_value < 0:
            raise CodegenBackendError(
                "codegen cdist expects p to be a non-negative number"
            )
        if isinstance(compute_mode, torch.fx.Node):
            raise CodegenBackendError(
                "codegen cdist expects compute_mode to be a string"
            )
        compute_mode_value = None
        if compute_mode is not None:
            if isinstance(compute_mode, str):
                compute_mode_value = compute_mode
            else:
                try:
                    compute_mode_index = int(operator.index(compute_mode))
                except TypeError as exc:
                    raise CodegenBackendError(
                        "codegen cdist expects compute_mode to be a string"
                    ) from exc
                compute_mode_value = {
                    0: "use_mm_for_euclid_dist_if_necessary",
                    1: "use_mm_for_euclid_dist",
                    2: "donot_use_mm_for_euclid_dist",
                }.get(compute_mode_index)
            if compute_mode_value not in (
                "use_mm_for_euclid_dist_if_necessary",
                "use_mm_for_euclid_dist",
                "donot_use_mm_for_euclid_dist",
            ):
                raise CodegenBackendError(
                    "codegen cdist supports compute_mode='use_mm_for_euclid_dist', "
                    "'use_mm_for_euclid_dist_if_necessary', "
                    "'donot_use_mm_for_euclid_dist', or None"
                )
        x1_shape = self._shapes[x1]
        x2_shape = self._shapes[x2]
        if len(x1_shape) < 2 or len(x2_shape) < 2:
            raise CodegenBackendError("codegen cdist expects inputs with rank >= 2")
        if x1_shape[-1] != x2_shape[-1]:
            raise CodegenBackendError(
                "codegen cdist expects inputs to match in the last dimension"
            )
        _ = shape_utils.broadcast_output_shape(
            op_spec.name, x1_shape[:-2], x2_shape[:-2]
        )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[x1, x2],
            output_shape=(),
            inplace_input=None,
            params={"p": p_value, "compute_mode": compute_mode_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [x1_shape, x2_shape]
        )

    def build_native_dropout(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 3:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects three inputs"
            )
        if node.kwargs:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positional args only"
            )
        input_arg, p_arg, train_arg = node.args[:3]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen native_dropout expects inputs to share the graph dtype"
            )
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen native_dropout supports only float32 tensors"
            )
        p_value = float(resolve_scalar_arg(op_spec.name, p_arg, self._scalar_values))
        train_value = bool(
            resolve_scalar_arg(op_spec.name, train_arg, self._scalar_values)
        )
        if p_value < 0.0 or p_value > 1.0:
            raise CodegenBackendError(
                "codegen native_dropout expects p to be within [0, 1]"
            )
        if train_value and p_value != 0.0:
            raise CodegenBackendError(
                "codegen native_dropout supports train=False or p=0.0"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            params={"p": p_value, "train": train_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [self._shapes[input_arg]]
        )

    def build_addmm_like(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None,
    ) -> _OpNode:
        input_nodes, alpha_value, beta_value = parse_addmm_like_args(
            op_spec.name, node
        )
        for arg in input_nodes:
            if not isinstance(arg, torch.fx.Node) or arg not in self._shapes:
                raise error_expected_tensor(op_spec.name)
        input_shapes = [self._shapes[arg] for arg in input_nodes]
        input_dtypes = [self._dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        validate_addmm_like_scalars(
            op_spec.name, dtype_info.torch_dtype, alpha_value, beta_value
        )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
            params={"alpha": alpha_value, "beta": beta_value},
        )
        return self._finalize_node(
            node,
            op_node,
            dtype_info,
            input_shapes,
            inplace_input=inplace_input,
        )

    def build_linear(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        input_arg, weight_arg, bias = parse_linear_args(op_spec.name, node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(weight_arg, torch.fx.Node) or weight_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        input_nodes = [input_arg, weight_arg]
        input_shapes = [self._shapes[input_arg], self._shapes[weight_arg]]
        input_dtypes = [self._dtypes[arg] for arg in input_nodes]
        input_shape, weight_shape = input_shapes
        if len(input_shape) < 2 or len(weight_shape) != 2:
            raise CodegenBackendError(
                "codegen linear expects input rank >= 2 and 2D weight"
            )
        if bias is not None:
            if not isinstance(bias, torch.fx.Node) or bias not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            input_nodes.append(bias)
            input_shapes.append(self._shapes[bias])
            input_dtypes.append(self._dtypes[bias])
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=None,
            params={"has_bias": bias is not None},
        )
        return self._finalize_node(
            node, op_node, dtype_info, input_shapes
        )

    def build_diagonal(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 1:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen diagonal expects input to match the graph dtype"
            )
        input_shape = self._shapes[input_arg]
        offset, dim1, dim2 = parse_diagonal_args(op_spec.name, node, input_shape)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={"offset": offset, "dim1": dim1, "dim2": dim2},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_cumsum(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 1:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen cumsum expects input to match the graph dtype"
            )
        input_shape = self._shapes[input_arg]
        dim_value, dtype = parse_cumsum_args(op_spec.name, node, input_shape)
        if dtype is not None and dtype is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen cumsum expects dtype to match the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={"dim": dim_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_sort(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        input_arg, dim_value, descending, stable = parse_sort_args(
            op_spec.name, node
        )
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen sort expects input to match the graph dtype"
            )
        input_shape = self._shapes[input_arg]
        rank = len(input_shape)
        if rank == 0:
            if dim_value not in (-1, 0):
                raise CodegenBackendError("codegen sort dim is out of range")
            dim_value = 0
        else:
            if dim_value < 0:
                dim_value += rank
            if dim_value < 0 or dim_value >= rank:
                raise CodegenBackendError("codegen sort dim is out of range")
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "dim": dim_value,
                "descending": bool(descending),
                "stable": bool(stable),
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_constant_pad(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if len(node.args) < 2:
            raise CodegenBackendError(
                "codegen constant_pad_nd expects pad and value arguments"
            )
        input_arg = node.args[0]
        pad = node.args[1]
        value = node.args[2] if len(node.args) > 2 else 0.0
        if node.kwargs:
            extra = set(node.kwargs)
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen constant_pad_nd expects input to match the graph dtype"
            )
        if isinstance(pad, torch.fx.Node):
            raise CodegenBackendError(
                "codegen constant_pad_nd expects pad to be a sequence"
            )
        if not isinstance(pad, (list, tuple)):
            raise CodegenBackendError(
                "codegen constant_pad_nd expects pad to be a sequence"
            )
        pad_values = []
        for item in pad:
            if isinstance(item, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen constant_pad_nd expects pad values to be integers"
                )
            try:
                pad_values.append(int(operator.index(item)))
            except TypeError:
                try:
                    pad_values.append(int(item))
                except (TypeError, ValueError) as exc:
                    raise CodegenBackendError(
                        "codegen constant_pad_nd expects pad values to be integers"
                    ) from exc
        input_shape = self._shapes[input_arg]
        rank = len(input_shape)
        if len(pad_values) > 2 * rank:
            raise CodegenBackendError(
                "codegen constant_pad_nd expects pad to have at most 2 * input rank values"
            )
        pad_before = [0] * rank
        pad_after = [0] * rank
        for idx in range(len(pad_values) // 2):
            dim = rank - 1 - idx
            pad_before[dim] = pad_values[2 * idx]
            pad_after[dim] = pad_values[2 * idx + 1]
        if isinstance(value, torch.fx.Node):
            raise CodegenBackendError(
                "codegen constant_pad_nd expects a constant padding value"
            )
        value = _normalize_scalar_value(op_spec.name, value)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            params={
                "pad_before": tuple(pad_before),
                "pad_after": tuple(pad_after),
                "value": value,
                "mode": "constant",
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_pad(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if op_spec.name == "constant_pad_nd":
            return self.build_constant_pad(node, op_spec, dtype_info)
        if op_spec.name == "reflection_pad1d":
            return self._build_mirror_pad(node, op_spec, dtype_info, 1, "reflection")
        if op_spec.name == "reflection_pad2d":
            return self._build_mirror_pad(node, op_spec, dtype_info, 2, "reflection")
        if op_spec.name == "reflection_pad3d":
            return self._build_mirror_pad(node, op_spec, dtype_info, 3, "reflection")
        if op_spec.name == "replication_pad2d":
            return self._build_mirror_pad(node, op_spec, dtype_info, 2, "replication")
        if op_spec.name == "replication_pad3d":
            return self._build_mirror_pad(node, op_spec, dtype_info, 3, "replication")
        raise CodegenBackendError(
            f"codegen pad does not support op {op_spec.name}"
        )

    def _build_mirror_pad(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        pad_dims: int,
        mode: str,
    ) -> _OpNode:
        if len(node.args) < 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects input and padding arguments"
            )
        input_arg = node.args[0]
        pad = node.args[1]
        if node.kwargs:
            extra = set(node.kwargs)
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects input to match the graph dtype"
            )
        if isinstance(pad, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects padding to be a sequence"
            )
        if not isinstance(pad, (list, tuple)):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects padding to be a sequence"
            )
        pad_values = []
        for item in pad:
            if isinstance(item, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects padding values to be integers"
                )
            try:
                pad_values.append(int(operator.index(item)))
            except TypeError:
                try:
                    pad_values.append(int(item))
                except (TypeError, ValueError) as exc:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects padding values to be integers"
                    ) from exc
        if len(pad_values) != 2 * pad_dims:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects {2 * pad_dims} padding values"
            )
        if any(value < 0 for value in pad_values):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects non-negative padding values"
            )
        input_shape = self._shapes[input_arg]
        rank = len(input_shape)
        if rank < pad_dims:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects input rank to be >= {pad_dims}"
            )
        pad_before = [0] * rank
        pad_after = [0] * rank
        for idx in range(pad_dims):
            dim = rank - 1 - idx
            pad_before[dim] = pad_values[2 * idx]
            pad_after[dim] = pad_values[2 * idx + 1]
        if mode == "reflection":
            for dim, (before, after) in enumerate(zip(pad_before, pad_after)):
                if before == 0 and after == 0:
                    continue
                if input_shape[dim] <= 0:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects non-empty input dimensions"
                    )
                if before >= input_shape[dim] or after >= input_shape[dim]:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects padding < input size"
                    )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            params={
                "pad_before": tuple(pad_before),
                "pad_after": tuple(pad_after),
                "mode": mode,
            },
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_gather(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        input_arg, dim, index, sparse_grad = parse_gather_args(node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(index, torch.fx.Node) or index not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen gather expects input to match the graph dtype"
            )
        if self._dtypes[index] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen gather expects index dtype to be torch.int32 or torch.int64"
            )
        if isinstance(sparse_grad, torch.fx.Node):
            raise CodegenBackendError("codegen gather expects sparse_grad to be False")
        if sparse_grad not in (False, 0, None):
            raise CodegenBackendError("codegen gather supports only sparse_grad=False")
        input_shape = self._shapes[input_arg]
        index_shape = self._shapes[index]
        if not input_shape:
            raise CodegenBackendError(
                "codegen gather expects input to have at least 1 dimension"
            )
        if len(index_shape) != len(input_shape):
            raise CodegenBackendError(
                "codegen gather expects index to have the same rank as input"
            )
        dim_value = parse_constant_int(op_spec.name, "dim", dim)
        if dim_value < 0:
            dim_value += len(input_shape)
        if dim_value < 0 or dim_value >= len(input_shape):
            raise CodegenBackendError("codegen gather dim is out of range")
        for idx, (input_dim, index_dim) in enumerate(
            zip(input_shape, index_shape)
        ):
            if idx == dim_value:
                continue
            if input_dim != index_dim:
                raise CodegenBackendError(
                    "codegen gather expects index shape to match input shape"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg, index],
            output_shape=(),
            inplace_input=None,
            params={"dim": dim_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape, index_shape]
        )

    def build_index_put(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None,
    ) -> _OpNode:
        input_arg, indices, values, accumulate = parse_index_put_args(node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(indices, (list, tuple)) or not indices:
            raise CodegenBackendError(
                "codegen index_put expects indices to be a non-empty sequence"
            )
        if not isinstance(values, torch.fx.Node) or values not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if isinstance(accumulate, torch.fx.Node):
            raise CodegenBackendError(
                "codegen index_put expects accumulate to be a constant"
            )
        if accumulate not in (None, False, True, 0, 1):
            raise CodegenBackendError(
                "codegen index_put expects accumulate to be a bool"
            )
        accumulate_value = accumulate in (True, 1)
        index_nodes = []
        index_shapes = []
        for index in indices:
            if not isinstance(index, torch.fx.Node) or index not in self._shapes:
                raise error_expected_tensor(op_spec.name)
            index_nodes.append(index)
            index_shapes.append(self._shapes[index])
        input_shape = self._shapes[input_arg]
        if len(index_nodes) > len(input_shape):
            raise CodegenBackendError(
                "codegen index_put expects indices length <= input rank"
            )
        if any(shape != index_shapes[0] for shape in index_shapes[1:]):
            raise CodegenBackendError(
                "codegen index_put expects indices to share the same shape"
            )
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen index_put expects input to match the graph dtype"
            )
        if self._dtypes[values] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen index_put expects values to match the graph dtype"
            )
        uses_mask = False
        for index in index_nodes:
            index_dtype = self._dtypes[index]
            if index_dtype is torch.bool:
                uses_mask = True
                continue
            if index_dtype not in _EMBEDDING_INDEX_DTYPES:
                raise CodegenBackendError(
                    "codegen index_put expects indices to be int32, int64, or bool"
                )
        values_shape = self._shapes[values]
        if uses_mask:
            if len(index_nodes) != 1:
                raise CodegenBackendError(
                    "codegen index_put expects a single bool mask index"
                )
            if len(index_shapes[0]) != 1:
                raise CodegenBackendError(
                    "codegen index_put expects a 1D bool mask"
                )
            if input_shape and index_shapes[0][0] != input_shape[0]:
                raise CodegenBackendError(
                    "codegen index_put expects mask length to match input"
                )
            expected_values_shape = tuple(input_shape[1:])
        else:
            expected_values_shape = tuple(index_shapes[0]) + tuple(
                input_shape[len(index_nodes) :]
            )
        if tuple(values_shape) != expected_values_shape:
            if uses_mask:
                raise CodegenBackendError(
                    "codegen index_put expects values to match the tail shape"
                )
            raise CodegenBackendError(
                "codegen index_put expects values to match indices + tail shape"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg, *index_nodes, values],
            output_shape=(),
            inplace_input=inplace_input,
            params={"index_rank": len(index_nodes), "accumulate": accumulate_value},
        )
        input_shapes = [input_shape, *index_shapes, values_shape]
        return self._finalize_node(
            node,
            op_node,
            dtype_info,
            input_shapes,
            inplace_input=inplace_input,
        )

    def build_masked_scatter(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None = None,
    ) -> _OpNode:
        input_arg, mask, source = parse_masked_scatter_args(node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(mask, torch.fx.Node) or mask not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(source, torch.fx.Node) or source not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen masked_scatter expects input to match the graph dtype"
            )
        if self._dtypes[source] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen masked_scatter expects source to match the graph dtype"
            )
        if self._dtypes[mask] is not torch.bool:
            raise CodegenBackendError(
                "codegen masked_scatter expects mask dtype to be torch.bool"
            )
        input_shape = self._shapes[input_arg]
        mask_shape = self._shapes[mask]
        source_shape = self._shapes[source]
        if input_shape != mask_shape:
            raise CodegenBackendError(
                "codegen masked_scatter expects mask shape to match input shape"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg, mask, source],
            output_shape=(),
            inplace_input=inplace_input,
            params={},
        )
        return self._finalize_node(
            node,
            op_node,
            dtype_info,
            [input_shape, mask_shape, source_shape],
            inplace_input=inplace_input,
        )

    def build_index_select(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        input_arg, dim, index = parse_index_select_args(node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(index, torch.fx.Node) or index not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen index_select expects input to match the graph dtype"
            )
        if self._dtypes[index] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen index_select expects index dtype to be torch.int32 or torch.int64"
            )
        input_shape = self._shapes[input_arg]
        if not input_shape:
            raise CodegenBackendError(
                "codegen index_select expects input to have at least 1 dimension"
            )
        index_shape = self._shapes[index]
        if len(index_shape) != 1:
            raise CodegenBackendError(
                "codegen index_select expects index to be a 1D tensor"
            )
        dim_value = parse_constant_int(op_spec.name, "dim", dim)
        if dim_value < 0:
            dim_value += len(input_shape)
        if dim_value < 0 or dim_value >= len(input_shape):
            raise CodegenBackendError("codegen index_select dim is out of range")
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg, index],
            output_shape=(),
            inplace_input=None,
            params={"dim": dim_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape, index_shape]
        )

    def build_select_scatter(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None = None,
    ) -> _OpNode:
        input_arg, src, dim, index = parse_select_scatter_args(node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(src, torch.fx.Node) or src not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen select_scatter expects input to match the graph dtype"
            )
        if self._dtypes[src] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen select_scatter expects src to match the graph dtype"
            )
        input_shape = self._shapes[input_arg]
        if not input_shape:
            raise CodegenBackendError(
                "codegen select_scatter expects input to have at least 1 dimension"
            )
        dim_value = parse_constant_int(op_spec.name, "dim", dim)
        if dim_value < 0:
            dim_value += len(input_shape)
        if dim_value < 0 or dim_value >= len(input_shape):
            raise CodegenBackendError("codegen select_scatter dim is out of range")
        index_value = parse_constant_int(op_spec.name, "index", index)
        if index_value < 0:
            index_value += input_shape[dim_value]
        if index_value < 0 or index_value >= input_shape[dim_value]:
            raise CodegenBackendError(
                "codegen select_scatter index is out of range"
            )
        src_shape = self._shapes[src]
        expected_src_shape = tuple(
            size
            for dim_index, size in enumerate(input_shape)
            if dim_index != dim_value
        )
        if src_shape != expected_src_shape:
            raise CodegenBackendError(
                "codegen select_scatter expects src to match input shape without dim"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg, src],
            output_shape=(),
            inplace_input=inplace_input,
            params={"dim": dim_value, "index": index_value},
        )
        return self._finalize_node(
            node,
            op_node,
            dtype_info,
            [input_shape, src_shape],
            inplace_input=inplace_input,
        )

    def build_scatter(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None,
    ) -> _OpNode:
        if op_spec.name == "scatter_src":
            return self._build_scatter_src(
                node, op_spec, dtype_info, inplace_input
            )
        if op_spec.name == "scatter_value":
            return self._build_scatter_value(
                node, op_spec, dtype_info, inplace_input
            )
        raise CodegenBackendError(
            f"codegen scatter does not support op {op_spec.name}"
        )

    def _build_scatter_src(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None,
    ) -> _OpNode:
        input_arg, dim, index, src = parse_scatter_src_args(node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(index, torch.fx.Node) or index not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(src, torch.fx.Node) or src not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen scatter expects input to match the graph dtype"
            )
        if self._dtypes[src] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen scatter expects src to match the graph dtype"
            )
        if self._dtypes[index] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen scatter expects index dtype to be torch.int32 or torch.int64"
            )
        input_shape = self._shapes[input_arg]
        index_shape = self._shapes[index]
        src_shape = self._shapes[src]
        if len(input_shape) == 0:
            raise CodegenBackendError(
                "codegen scatter expects input to have at least 1 dimension"
            )
        if len(index_shape) != len(input_shape):
            raise CodegenBackendError(
                "codegen scatter expects index to have the same rank as input"
            )
        dim_value = parse_constant_int(op_spec.name, "dim", dim)
        if dim_value < 0:
            dim_value += len(input_shape)
        if dim_value < 0 or dim_value >= len(input_shape):
            raise CodegenBackendError("codegen scatter dim is out of range")
        if tuple(index_shape) != tuple(src_shape):
            raise CodegenBackendError(
                "codegen scatter expects index and src to share the same shape"
            )
        for idx, (input_dim, index_dim) in enumerate(
            zip(input_shape, index_shape)
        ):
            if idx == dim_value:
                continue
            if input_dim != index_dim:
                raise CodegenBackendError(
                    "codegen scatter expects index shape to match input shape "
                    "for non-scatter dimensions"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg, index, src],
            output_shape=(),
            inplace_input=inplace_input,
            params={"dim": dim_value},
        )
        return self._finalize_node(
            node,
            op_node,
            dtype_info,
            [input_shape, index_shape, src_shape],
            inplace_input=inplace_input,
        )

    def _build_scatter_value(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None,
    ) -> _OpNode:
        input_arg, dim, index, value = parse_scatter_value_args(node)
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if not isinstance(index, torch.fx.Node) or index not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen scatter expects input to match the graph dtype"
            )
        if self._dtypes[index] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen scatter expects index dtype to be torch.int32 or torch.int64"
            )
        input_shape = self._shapes[input_arg]
        index_shape = self._shapes[index]
        if len(input_shape) == 0:
            raise CodegenBackendError(
                "codegen scatter expects input to have at least 1 dimension"
            )
        if len(index_shape) != len(input_shape):
            raise CodegenBackendError(
                "codegen scatter expects index to have the same rank as input"
            )
        dim_value = parse_constant_int(op_spec.name, "dim", dim)
        if dim_value < 0:
            dim_value += len(input_shape)
        if dim_value < 0 or dim_value >= len(input_shape):
            raise CodegenBackendError("codegen scatter dim is out of range")
        for idx, (input_dim, index_dim) in enumerate(
            zip(input_shape, index_shape)
        ):
            if idx == dim_value:
                continue
            if input_dim != index_dim:
                raise CodegenBackendError(
                    "codegen scatter expects index shape to match input shape "
                    "for non-scatter dimensions"
                )
        if isinstance(value, torch.fx.Node):
            raise CodegenBackendError(
                "codegen scatter expects value to be a constant"
            )
        value = _normalize_scalar_value(op_spec.name, value)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg, index],
            output_shape=(),
            inplace_input=inplace_input,
            params={"dim": dim_value, "value": value},
        )
        return self._finalize_node(
            node,
            op_node,
            dtype_info,
            [input_shape, index_shape],
            inplace_input=inplace_input,
        )

    def build_repeat(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        repeats_arg = None
        if len(node.args) > 1:
            if len(node.args) == 2:
                repeats_arg = node.args[1]
            else:
                repeats_arg = node.args[1:]
        if node.kwargs:
            if "repeats" in node.kwargs:
                if repeats_arg is not None and len(node.args) > 1:
                    raise error_kwarg_specified_once(op_spec.name, "repeats")
                repeats_arg = node.kwargs["repeats"]
            extra = set(node.kwargs) - {"repeats"}
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        if repeats_arg is None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects repeats arguments"
            )
        repeats = normalize_as_strided_sequence(
            op_spec.name,
            repeats_arg,
            "repeats",
            scalar_values=self._scalar_values,
        )
        input_shape = self._shapes[input_arg]
        if len(repeats) < len(input_shape):
            raise CodegenBackendError(
                "codegen repeat expects repeats to cover the input rank"
            )
        if any(repeat < 0 for repeat in repeats):
            raise CodegenBackendError(
                "codegen repeat expects repeats to be non-negative"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={"repeats": repeats},
        )
        return self._finalize_node(
            node,
            op_node,
            dtype_info,
            [input_shape],
        )

    def build_view(
        self, node: torch.fx.Node, op_spec: _OpSpec, dtype_info: _CodegenDType
    ) -> _OpNode:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        if op_spec.name == "_local_scalar_dense":
            if len(node.args) != 1:
                raise CodegenBackendError(
                    "codegen _local_scalar_dense expects one input"
                )
            if node.kwargs:
                raise CodegenBackendError(
                    "codegen _local_scalar_dense expects no kwargs"
                )
            input_shape = self._shapes[input_arg]
            if input_shape:
                raise CodegenBackendError(
                    "codegen _local_scalar_dense expects a 0-d input"
                )
            op_node = _OpNode(
                node=node,
                spec=op_spec,
                inputs=[input_arg],
                output_shape=(),
                params={
                    "size": (),
                    "view_strides": _contiguous_strides(()),
                    "storage_offset": 0,
                },
            )
            return self._finalize_node(
                node, op_node, dtype_info, [input_shape]
            )
        if op_spec.name == "as_strided":
            if len(node.args) > 4:
                raise CodegenBackendError(
                    "codegen as_strided expects at most four inputs"
                )
            size = node.args[1] if len(node.args) > 1 else None
            stride = node.args[2] if len(node.args) > 2 else None
            storage_offset = node.args[3] if len(node.args) > 3 else None
            if node.kwargs:
                if "size" in node.kwargs:
                    if size is not None:
                        raise error_kwarg_specified_once(op_spec.name, "size")
                    size = node.kwargs["size"]
                if "stride" in node.kwargs:
                    if stride is not None:
                        raise error_kwarg_specified_once(op_spec.name, "stride")
                    stride = node.kwargs["stride"]
                if "storage_offset" in node.kwargs:
                    if storage_offset is not None:
                        raise error_kwarg_specified_once(
                            op_spec.name, "storage_offset"
                        )
                    storage_offset = node.kwargs["storage_offset"]
                extra = set(node.kwargs) - {"size", "stride", "storage_offset"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if size is None or stride is None:
                raise CodegenBackendError(
                    "codegen as_strided expects size and stride"
                )
            if storage_offset is None:
                storage_offset = 0
            if (
                isinstance(storage_offset, torch.fx.Node)
                and storage_offset in self._scalar_values
            ):
                storage_offset = self._scalar_values[storage_offset]
            stride_input = (
                stride
                if isinstance(stride, torch.fx.Node) and stride in self._shapes
                else None
            )
            size_tuple = normalize_as_strided_sequence(
                op_spec.name,
                size,
                "size",
                scalar_values=self._scalar_values,
            )
            stride_tuple = normalize_as_strided_sequence(
                op_spec.name,
                stride,
                "stride",
                scalar_values=self._scalar_values,
            )
            if len(size_tuple) != len(stride_tuple):
                raise CodegenBackendError(
                    "codegen as_strided expects size and stride to match length"
                )
            storage_offset_value = parse_constant_int(
                op_spec.name, "storage_offset", storage_offset
            )
            inputs = [input_arg]
            params = {
                "size": size_tuple,
                "view_strides": stride_tuple,
                "storage_offset": storage_offset_value,
            }
            if stride_input is not None:
                stride_shape = self._shapes[stride_input]
                stride_dtype = self._dtypes[stride_input]
                if stride_dtype not in (torch.int32, torch.int64):
                    raise CodegenBackendError(
                        "codegen as_strided expects stride tensor to have an int dtype"
                    )
                if len(stride_shape) != 1 or stride_shape[0] != len(size_tuple):
                    raise CodegenBackendError(
                        "codegen as_strided expects stride tensor to match size length"
                    )
                inputs.append(stride_input)
                params["view_strides_input_index"] = len(inputs) - 1
            op_node = _OpNode(
                node=node,
                spec=op_spec,
                inputs=inputs,
                output_shape=(),
                params=params,
            )
            return self._finalize_node(
                node, op_node, dtype_info, [self._shapes[input_arg]]
            )
        if op_spec.name == "reshape":
            shape_values: List[object] | None = None
            shape_arg = None
            if len(node.args) > 2:
                shape_values = list(node.args[1:])
            else:
                if len(node.args) > 1:
                    shape_arg = node.args[1]
                if node.kwargs:
                    if "shape" in node.kwargs:
                        if shape_arg is not None:
                            raise error_kwarg_specified_once(
                                op_spec.name, "shape"
                            )
                        shape_arg = node.kwargs["shape"]
                    extra = set(node.kwargs) - {"shape"}
                    if extra:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                        )
                if shape_arg is None:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects a shape argument"
                    )
                if isinstance(shape_arg, torch.Size):
                    shape_values = list(shape_arg)
                elif isinstance(shape_arg, (tuple, list)):
                    shape_values = list(shape_arg)
                elif isinstance(shape_arg, (torch.Tensor, torch.fx.Node)):
                    try:
                        shape_values = list(
                            normalize_as_strided_sequence(
                                op_spec.name,
                                shape_arg,
                                "shape",
                                scalar_values=self._scalar_values,
                            )
                        )
                    except CodegenBackendError:
                        shape_values = [shape_arg]
                else:
                    shape_values = [shape_arg]
            from codegen_backend.emitters.base import _is_contiguous

            if not _is_contiguous(
                self._shapes[input_arg], self._strides[input_arg]
            ):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects contiguous input"
                )
            output_shape: List[int] = []
            unknown_dim = None
            known_product = 1
            for dim in shape_values:
                dim_value = parse_constant_int(op_spec.name, "shape", dim)
                if dim_value == -1:
                    if unknown_dim is not None:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects at most one -1 dim"
                        )
                    unknown_dim = len(output_shape)
                    output_shape.append(-1)
                    continue
                if dim_value < -1:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} expects shape dims >= -1"
                    )
                output_shape.append(dim_value)
                known_product *= dim_value
            input_numel = math.prod(self._shapes[input_arg])
            if unknown_dim is not None:
                if known_product == 0:
                    if input_numel != 0:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects reshape numel to match input numel"
                        )
                    output_shape[unknown_dim] = 0
                else:
                    if input_numel % known_product != 0:
                        raise CodegenBackendError(
                            f"codegen {op_spec.name} expects reshape numel to match input numel"
                        )
                    output_shape[unknown_dim] = input_numel // known_product
            elif known_product != input_numel:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects reshape numel to match input numel"
                )
            output_shape_tuple = tuple(output_shape)
            op_node = _OpNode(
                node=node,
                spec=op_spec,
                inputs=[input_arg],
                output_shape=(),
                params={
                    "size": output_shape_tuple,
                    "view_strides": _contiguous_strides(output_shape_tuple),
                    "storage_offset": 0,
                },
            )
            return self._finalize_node(
                node, op_node, dtype_info, [self._shapes[input_arg]]
            )
        if op_spec.name == "select":
            if len(node.args) > 3:
                raise CodegenBackendError(
                    "codegen select expects input, dim, and index arguments"
                )
            dim = node.args[1] if len(node.args) > 1 else None
            index = node.args[2] if len(node.args) > 2 else None
            if node.kwargs:
                if "dim" in node.kwargs:
                    if dim is not None:
                        raise error_kwarg_specified_once(op_spec.name, "dim")
                    dim = node.kwargs["dim"]
                if "index" in node.kwargs:
                    if index is not None:
                        raise error_kwarg_specified_once(op_spec.name, "index")
                    index = node.kwargs["index"]
                extra = set(node.kwargs) - {"dim", "index"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if dim is None or index is None:
                raise CodegenBackendError(
                    "codegen select expects dim and index arguments"
                )
            input_shape = self._shapes[input_arg]
            if not input_shape:
                raise CodegenBackendError(
                    "codegen select expects input to have at least 1 dimension"
                )
            dim_value = parse_constant_int(op_spec.name, "dim", dim)
            if dim_value < 0:
                dim_value += len(input_shape)
            if dim_value < 0 or dim_value >= len(input_shape):
                raise CodegenBackendError("codegen select dim is out of range")
            index_value = parse_constant_int(op_spec.name, "index", index)
            if index_value < 0:
                index_value += input_shape[dim_value]
            if index_value < 0 or index_value >= input_shape[dim_value]:
                raise CodegenBackendError("codegen select index is out of range")
            output_shape = tuple(
                size
                for dim_index, size in enumerate(input_shape)
                if dim_index != dim_value
            )
            input_strides = self._strides[input_arg]
            view_strides = tuple(
                stride
                for dim_index, stride in enumerate(input_strides)
                if dim_index != dim_value
            )
            storage_offset = index_value * input_strides[dim_value]
            op_node = _OpNode(
                node=node,
                spec=op_spec,
                inputs=[input_arg],
                output_shape=(),
                params={
                    "size": output_shape,
                    "view_strides": view_strides,
                    "storage_offset": storage_offset,
                },
            )
            return self._finalize_node(
                node, op_node, dtype_info, [self._shapes[input_arg]]
            )
        if op_spec.name == "flatten":
            if len(node.args) > 3:
                raise CodegenBackendError(
                    "codegen flatten expects input, start_dim, and end_dim arguments"
                )
            start_dim = 0
            end_dim = -1
            if len(node.args) > 1:
                start_dim = node.args[1]
            if len(node.args) > 2:
                end_dim = node.args[2]
            if node.kwargs:
                if "start_dim" in node.kwargs:
                    if len(node.args) > 1:
                        raise error_kwarg_specified_once(
                            op_spec.name, "start_dim"
                        )
                    start_dim = node.kwargs["start_dim"]
                if "end_dim" in node.kwargs:
                    if len(node.args) > 2:
                        raise error_kwarg_specified_once(
                            op_spec.name, "end_dim"
                        )
                    end_dim = node.kwargs["end_dim"]
                extra = set(node.kwargs) - {"start_dim", "end_dim"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                    )
            if isinstance(start_dim, torch.fx.Node) or isinstance(
                end_dim, torch.fx.Node
            ):
                raise CodegenBackendError(
                    "codegen flatten expects start_dim/end_dim to be constants"
                )
            start_dim_value = parse_constant_int(
                op_spec.name, "start_dim", start_dim
            )
            end_dim_value = parse_constant_int(
                op_spec.name, "end_dim", end_dim
            )
            input_shape = self._shapes[input_arg]
            rank = len(input_shape)
            if rank == 0:
                raise CodegenBackendError(
                    "codegen flatten expects input to have at least one dimension"
                )
            if start_dim_value < 0:
                start_dim_value += rank
            if end_dim_value < 0:
                end_dim_value += rank
            if start_dim_value < 0 or start_dim_value >= rank:
                raise CodegenBackendError(
                    "codegen flatten start_dim is out of range"
                )
            if end_dim_value < 0 or end_dim_value >= rank:
                raise CodegenBackendError(
                    "codegen flatten end_dim is out of range"
                )
            if start_dim_value > end_dim_value:
                raise CodegenBackendError(
                    "codegen flatten expects start_dim <= end_dim"
                )
            from codegen_backend.emitters.base import _is_contiguous

            if not _is_contiguous(
                input_shape, self._strides[input_arg]
            ):
                raise CodegenBackendError(
                    "codegen flatten expects contiguous input"
                )
            flattened_dim = math.prod(
                input_shape[start_dim_value : end_dim_value + 1]
            )
            output_shape = (
                list(input_shape[:start_dim_value])
                + [flattened_dim]
                + list(input_shape[end_dim_value + 1 :])
            )
            output_shape_tuple = tuple(output_shape)
            op_node = _OpNode(
                node=node,
                spec=op_spec,
                inputs=[input_arg],
                output_shape=(),
                params={
                    "size": output_shape_tuple,
                    "view_strides": _contiguous_strides(output_shape_tuple),
                    "storage_offset": 0,
                },
            )
            return self._finalize_node(
                node, op_node, dtype_info, [input_shape]
            )
        if len(node.args) < 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects input and size arguments"
            )
        input_arg, size_arg = node.args[:2]
        if node.kwargs:
            extra = set(node.kwargs)
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
                )
        size = normalize_as_strided_sequence(
            op_spec.name, size_arg, "size", scalar_values=self._scalar_values
        )
        input_shape = self._shapes[input_arg]
        output_numel = math.prod(size) if size else 1
        input_numel = math.prod(input_shape) if input_shape else 1
        if output_numel != input_numel:
            raise CodegenBackendError(
                "codegen view expects size to match input numel"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={"size": size},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [input_shape]
        )

    def build_resize(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType,
        inplace_input: int | None,
    ) -> _OpNode:
        memory_format = None
        if node.kwargs:
            if set(node.kwargs) != {"memory_format"}:
                raise CodegenBackendError(
                    "codegen resize_ expects only memory_format as a keyword argument"
                )
            memory_format = node.kwargs.get("memory_format")
        if len(node.args) != 2:
            raise CodegenBackendError(
                "codegen resize_ expects input and size arguments"
            )
        input_arg, size_arg = node.args
        if not isinstance(input_arg, torch.fx.Node):
            raise error_expected_tensor(op_spec.name)
        if input_arg not in self._shapes:
            raise error_expected_tensor(op_spec.name)
        if self._dtypes[input_arg] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen resize_ expects inputs to share the graph dtype"
            )
        size = parse_resize_size(op_spec.name, size_arg)
        if any(dim < 0 for dim in size):
            raise CodegenBackendError(
                "codegen resize_ expects size values to be non-negative"
            )
        input_shape = self._shapes[input_arg]
        if math.prod(size) != math.prod(input_shape):
            raise CodegenBackendError(
                "codegen resize_ expects size to match the input numel"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={"size": size},
        )
        output_shape = self._analysis_service.infer_output_shape(
            op_node, [input_shape]
        )
        op_node.output_shape = output_shape
        self._shapes[node] = output_shape
        self._dtypes[node] = dtype_info.torch_dtype
        if memory_format is None or memory_format is torch.contiguous_format:
            output_strides = _contiguous_strides(output_shape)
        elif memory_format is torch.channels_last:
            output_strides = channels_last_strides(output_shape)
        elif memory_format is torch.channels_last_3d:
            output_strides = channels_last_3d_strides(output_shape)
        else:
            raise CodegenBackendError("Unsupported memory formatPreserve")
        self._strides[node] = output_strides
        return op_node


__all__ = [
    "TensorOpBuilder",
    "parse_addmm_like_args",
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
]
