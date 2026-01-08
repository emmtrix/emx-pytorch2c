from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import torch
import torch.fx

from codegen_backend.analysis_helpers import (
    channels_last_3d_strides,
    channels_last_strides,
    error_expected_tensor,
    error_kwarg_specified_once,
    is_out_overload,
    normalize_as_strided_sequence,
    normalize_flip_dims,
    normalize_reduction_dims,
    parse_bitwise_scalar,
    parse_constant_bool,
    parse_constant_float,
    parse_constant_int,
    resolve_scalar_arg,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode
from codegen_backend.kinds import OpKindHandler
from codegen_backend.specs import OpKind


class GraphAnalysisService:
    def __init__(
        self,
        kind_handlers: Callable[[], Dict[OpKind, OpKindHandler]],
    ) -> None:
        self._kind_handlers = kind_handlers

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        handlers = self._kind_handlers()
        handler = handlers.get(op_node.spec.kind)
        if handler is None:
            raise CodegenBackendError(
                f"codegen backend does not support kind '{op_node.spec.kind.value}'"
            )
        return handler.infer_shapes(op_node, input_shapes)

    def error_expected_tensor(self, op_name: str) -> CodegenBackendError:
        return error_expected_tensor(op_name)

    def error_kwarg_specified_once(
        self, op_name: str, kwarg: str
    ) -> CodegenBackendError:
        return error_kwarg_specified_once(op_name, kwarg)

    def is_out_overload(self, target: object) -> bool:
        return is_out_overload(target)

    def normalize_as_strided_sequence(
        self, op_name: str, value: object, arg_name: str
    ) -> Tuple[int, ...]:
        return normalize_as_strided_sequence(op_name, value, arg_name)

    def normalize_flip_dims(
        self, op_name: str, dims: object, rank: int
    ) -> Tuple[int, ...]:
        return normalize_flip_dims(op_name, dims, rank)

    def normalize_reduction_dims(
        self, op_name: str, dim: object | None, rank: int
    ) -> Tuple[int, ...]:
        return normalize_reduction_dims(op_name, dim, rank)

    def parse_constant_bool(self, op_name: str, name: str, value: object) -> bool:
        return parse_constant_bool(op_name, name, value)

    def parse_constant_float(self, op_name: str, name: str, value: object) -> float:
        return parse_constant_float(op_name, name, value)

    def parse_constant_int(self, op_name: str, name: str, value: object) -> int:
        return parse_constant_int(op_name, name, value)

    def parse_bitwise_scalar(
        self, op_name: str, value: object, dtype: torch.dtype
    ) -> object:
        return parse_bitwise_scalar(op_name, value, dtype)

    def resolve_scalar_arg(
        self,
        op_name: str,
        value: object,
        scalar_values: Dict[torch.fx.Node, object],
    ) -> float | int | bool:
        return resolve_scalar_arg(op_name, value, scalar_values)

    def channels_last_strides(self, shape: Sequence[int]) -> Tuple[int, ...]:
        return channels_last_strides(shape)

    def channels_last_3d_strides(self, shape: Sequence[int]) -> Tuple[int, ...]:
        return channels_last_3d_strides(shape)
