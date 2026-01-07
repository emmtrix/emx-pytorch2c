from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import torch.fx

from codegen_backend.analysis_helpers import (
    error_expected_tensor,
    error_kwarg_specified_once,
    is_out_overload,
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

    def resolve_scalar_arg(
        self,
        op_name: str,
        value: object,
        scalar_values: Dict[torch.fx.Node, object],
    ) -> float | int | bool:
        return resolve_scalar_arg(op_name, value, scalar_values)
