from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence, Tuple

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from codegen_backend.dtypes import _CODEGEN_DTYPES, _CodegenDType, _EMBEDDING_INDEX_DTYPES
from codegen_backend.errors import CodegenBackendError
from codegen_backend.kinds import OpKind, OpKindHandler


class Parser:
    def __init__(
        self,
        *,
        kind_handlers: Callable[[], Dict[OpKind, OpKindHandler]],
        target_registry: Callable[[], Dict[object, "_TargetInfo"]],
    ) -> None:
        self._kind_handlers = kind_handlers
        self._target_registry = target_registry

    def resolve_dtype_info(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[object],
    ) -> _CodegenDType | None:
        tensor_examples = list(self._iter_example_tensors(example_inputs))
        if tensor_examples:
            return self._validate_example_inputs(example_inputs)
        return self._infer_empty_strided_dtype(gm)

    def unwrap_output_node(
        self, output_node: torch.fx.Node
    ) -> Tuple[torch.fx.Node, object]:
        output_value = output_node.args[0]
        output_structure = output_value
        if isinstance(output_value, (tuple, list, immutable_list)):
            if not output_value:
                raise CodegenBackendError(
                    "codegen backend expects a non-empty output list"
                )
            if not all(isinstance(item, torch.fx.Node) for item in output_value):
                raise CodegenBackendError("codegen backend expects output nodes only")
            output_value = output_value[0]
        if not isinstance(output_value, torch.fx.Node):
            raise CodegenBackendError("codegen backend expects a single output node")
        return output_value, output_structure

    def _iter_example_tensors(
        self, example_inputs: Sequence[object]
    ) -> Iterable[torch.Tensor]:
        for example in example_inputs:
            if isinstance(example, torch.Tensor):
                yield example
                continue
            if isinstance(example, (list, tuple)):
                for item in example:
                    if isinstance(item, torch.Tensor):
                        yield item
                    elif isinstance(item, (list, tuple)):
                        yield from self._iter_example_tensors(item)

    def _validate_example_inputs(
        self, example_inputs: Sequence[object]
    ) -> _CodegenDType | None:
        all_tensor_examples = list(self._iter_example_tensors(example_inputs))
        non_scalar_examples = [
            example
            for example in all_tensor_examples
            if example.numel() != 1 or example.dim() != 0
        ]
        candidate_examples = non_scalar_examples or all_tensor_examples
        tensor_examples = [
            example
            for example in candidate_examples
            if example.dtype in _CODEGEN_DTYPES
        ]
        if not tensor_examples:
            if all_tensor_examples:
                raise CodegenBackendError(
                    "codegen backend supports only torch.float32, torch.float64, torch.int8, torch.uint8, torch.uint32, torch.int32, or torch.bool tensors"
                )
            return None
        for example in self._iter_example_tensors(example_inputs):
            if example.device.type != "cpu":
                raise CodegenBackendError("codegen backend supports only CPU tensors")
        non_bool_examples = [
            example for example in tensor_examples if example.dtype is not torch.bool
        ]
        if non_bool_examples:
            non_bool_dtypes = {example.dtype for example in non_bool_examples}
            non_index_dtypes = {
                dtype
                for dtype in non_bool_dtypes
                if dtype not in _EMBEDDING_INDEX_DTYPES
            }
            if len(non_index_dtypes) > 1:
                raise CodegenBackendError(
                    "codegen backend expects all tensors to share a dtype"
                )
            if non_index_dtypes:
                first_dtype = next(iter(non_index_dtypes))
            else:
                first_dtype = next(iter(non_bool_dtypes))
        else:
            first_dtype = torch.bool
        dtype_info = _CODEGEN_DTYPES.get(first_dtype)
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen backend supports only torch.float32, torch.float64, torch.int8, torch.uint8, torch.uint32, torch.int32, or torch.bool tensors"
            )
        examples_to_check = non_scalar_examples or tensor_examples
        for example in examples_to_check:
            if example.dtype is torch.bool:
                continue
            if (
                example.dtype is not first_dtype
                and example.dtype not in _EMBEDDING_INDEX_DTYPES
            ):
                raise CodegenBackendError(
                    "codegen backend expects all tensors to share a dtype"
                )
        return dtype_info

    def _infer_empty_strided_dtype(
        self, gm: torch.fx.GraphModule
    ) -> _CodegenDType | None:
        dtype_value = None
        target_registry = self._target_registry()
        kind_handlers = self._kind_handlers()
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target_info = target_registry.get(node.target)
            if target_info is None:
                continue
            handler = kind_handlers.get(target_info.op_spec.kind)
            if handler is None:
                continue
            node_dtype = handler.infer_graph_dtype(node, target_info.op_spec)
            if node_dtype is None:
                continue
            if dtype_value is None:
                dtype_value = node_dtype
            elif dtype_value is not node_dtype:
                raise CodegenBackendError(
                    "codegen empty_strided requires a consistent dtype"
                )
        if dtype_value is None:
            return None
        dtype_info = _CODEGEN_DTYPES.get(dtype_value)
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen empty_strided supports only torch.float32, torch.float64, torch.int8, torch.uint8, torch.uint32, torch.int32, or torch.bool tensors"
            )
        return dtype_info
