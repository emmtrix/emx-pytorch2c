from __future__ import annotations

import math
import operator
from typing import Dict, List, Sequence, Tuple

import torch
import torch.fx

from codegen_backend.analysis_helpers import (
    channels_last_3d_strides,
    channels_last_strides,
    error_expected_tensor,
    error_kwarg_specified_once,
    normalize_as_strided_sequence,
    normalize_flip_dims,
    parse_constant_int,
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
    parse_linear_args,
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
            if not float(value).is_integer():
                raise CodegenBackendError(
                    f"codegen {op_name} expects {name} to be an integer for integral tensors"
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
            if isinstance(training, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects constant training flag"
                )
            if training not in (False, 0):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only training=False"
                )
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
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
            )
        if self._dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32"
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
            if self._dtypes[stat_arg] is not torch.float32:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32"
                )
        if weight_node is not None:
            if self._shapes[weight_node] != (channels,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects weight shape to match channels"
                )
            if self._dtypes[weight_node] is not torch.float32:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32"
                )
        if bias_node is not None:
            if self._shapes[bias_node] != (channels,):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects bias shape to match channels"
                )
            if self._dtypes[bias_node] is not torch.float32:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} supports only torch.float32"
                )
        try:
            _ = float(
                self._analysis_service.resolve_scalar_arg(
                    op_spec.name, momentum, self._scalar_values
                )
            )
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
        if math.isinf(p_value) or math.isnan(p_value):
            raise CodegenBackendError("codegen cdist expects p to be finite")
        if isinstance(compute_mode, torch.fx.Node):
            raise CodegenBackendError(
                "codegen cdist expects compute_mode to be a string"
            )
        if compute_mode is not None and compute_mode not in (
            "use_mm_for_euclid_dist",
        ):
            raise CodegenBackendError(
                "codegen cdist supports compute_mode='use_mm_for_euclid_dist' or None"
            )
        x1_shape = self._shapes[x1]
        x2_shape = self._shapes[x2]
        if len(x1_shape) != 2 or len(x2_shape) != 2:
            raise CodegenBackendError("codegen cdist expects 2D inputs")
        if x1_shape[1] != x2_shape[1]:
            raise CodegenBackendError(
                "codegen cdist expects inputs to match in dimension 1"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[x1, x2],
            output_shape=(),
            inplace_input=None,
            params={"p": p_value},
        )
        return self._finalize_node(
            node, op_node, dtype_info, [x1_shape, x2_shape]
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
            if isinstance(size, torch.fx.Node) or isinstance(stride, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen as_strided expects size/stride to be constants"
                )
            if storage_offset is None:
                storage_offset = 0
            if isinstance(storage_offset, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen as_strided expects storage_offset to be an int"
                )
            size_tuple = normalize_as_strided_sequence(op_spec.name, size, "size")
            stride_tuple = normalize_as_strided_sequence(
                op_spec.name, stride, "stride"
            )
            if len(size_tuple) != len(stride_tuple):
                raise CodegenBackendError(
                    "codegen as_strided expects size and stride to match length"
                )
            storage_offset_value = int(operator.index(storage_offset))
            if storage_offset_value < 0:
                raise CodegenBackendError(
                    "codegen as_strided expects storage_offset to be non-negative"
                )
            op_node = _OpNode(
                node=node,
                spec=op_spec,
                inputs=[input_arg],
                output_shape=(),
                params={
                    "size": size_tuple,
                    "view_strides": stride_tuple,
                    "storage_offset": storage_offset_value,
                },
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
                else:
                    shape_values = [shape_arg]
            if any(isinstance(value, torch.fx.Node) for value in shape_values):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects shape to be constant"
                )
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
        size = normalize_as_strided_sequence(op_spec.name, size_arg, "size")
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
    "parse_linear_args",
    "parse_resize_size",
]
