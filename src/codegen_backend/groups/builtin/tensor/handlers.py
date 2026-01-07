from __future__ import annotations

from typing import Dict, List

import numbers
import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.addbmm import AddbmmEmitter
from codegen_backend.emitters.addmm import AddmmEmitter
from codegen_backend.emitters.addmv import AddmvEmitter
from codegen_backend.emitters.addr import AddrEmitter
from codegen_backend.emitters.arange import ArangeEmitter
from codegen_backend.emitters.batch_norm import BatchNormEmitter
from codegen_backend.emitters.cdist import CdistEmitter
from codegen_backend.emitters.col2im import Col2imEmitter
from codegen_backend.emitters.concat import ConcatEmitter
from codegen_backend.emitters.cumsum import CumsumEmitter
from codegen_backend.emitters.diagonal import DiagonalEmitter
from codegen_backend.emitters.empty_strided import EmptyStridedEmitter
from codegen_backend.emitters.flip import FlipEmitter
from codegen_backend.emitters.gather import GatherEmitter
from codegen_backend.emitters.linear import LinearEmitter
from codegen_backend.emitters.matmul import MatmulEmitter
from codegen_backend.emitters.pad import PadEmitter
from codegen_backend.emitters.pdist import PdistEmitter
from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.emitters.resize import ResizeEmitter
from codegen_backend.emitters.view import ViewEmitter
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    AddbmmHandler,
    AddmmHandler,
    AddmvHandler,
    AddrHandler,
    ArangeHandler,
    BatchNormHandler,
    CdistHandler,
    Col2imHandler,
    ConcatHandler,
    CumsumHandler,
    DiagonalHandler,
    EmptyStridedHandler,
    FlipHandler,
    GatherHandler,
    HandlerContextProvider,
    LinearHandler,
    MatmulHandler,
    OpKindHandler,
    OpKindHandlerFactory,
    OpNodeBuildResult,
    PadHandler,
    PdistHandler,
    ResizeHandler,
    TensorContext,
    ViewHandler,
)
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.groups.builtin.tensor.analysis import (
    handle_addmm_like_node,
    handle_batch_norm_node,
    handle_cdist_node,
    handle_constant_pad_node,
    handle_cumsum_node,
    handle_diagonal_node,
    handle_flip_node,
    handle_gather_node,
    handle_linear_node,
    handle_pdist_node,
    handle_resize_node,
    handle_view_node,
    parse_arange_dtype,
    parse_concat_args,
    parse_empty_strided_stride,
    parse_resize_size,
)


class _BackendMatmulHandler(MatmulHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        allowed_kwargs = set()
        is_out_overload = self._ctx.analysis_service.is_out_overload(node.target)
        if is_out_overload:
            allowed_kwargs.add("out")
        if node.kwargs and set(node.kwargs) - allowed_kwargs:
            raise CodegenBackendError(
                "codegen backend expects positional args only"
            )
        expected_arity = 2
        out_arg: torch.fx.Node | None = None
        if is_out_overload:
            if inplace_input is None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects out to be provided"
                )
            if "out" in node.kwargs:
                if len(node.args) > expected_arity:
                    raise self._ctx.analysis_service.error_kwarg_specified_once(
                        op_spec.name, "out"
                    )
                out_arg = node.kwargs["out"]
            elif len(node.args) == expected_arity + 1:
                out_arg = node.args[inplace_input]
            elif len(node.args) != expected_arity:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects exactly two inputs"
                )
            if out_arg is None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects out to be provided"
                )
        elif len(node.args) != expected_arity:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects exactly two inputs"
            )
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[tuple[int, ...]] = []
        for arg in node.args[:expected_arity]:
            if not isinstance(arg, torch.fx.Node):
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            if arg not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            input_nodes.append(arg)
            input_shapes.append(shapes[arg])
        if out_arg is not None and out_arg not in input_nodes:
            if not isinstance(out_arg, torch.fx.Node):
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            if out_arg not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            input_nodes.append(out_arg)
            input_shapes.append(shapes[out_arg])
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
        )
        self.validate(op_node, input_shapes, input_dtypes, dtype_info)
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, input_shapes
        )
        op_node.output_shape = output_shape
        if out_arg is not None and shapes[out_arg] != output_shape:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects out to match output shape"
            )
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        if inplace_input is not None:
            strides[node] = strides[input_nodes[inplace_input]]
        else:
            strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendAddrHandler(AddrHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        op_node = handle_addmm_like_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            inplace_input,
            infer_output_shape=self._ctx.analysis_service.infer_output_shape,
        )
        if inplace_input is not None:
            input_shape = shapes[op_node.inputs[inplace_input]]
            output_shape = op_node.output_shape
            if len(input_shape) != 2:
                raise CodegenBackendError(
                    "codegen addr expects 2D input and 1D vectors"
                )
            if tuple(output_shape) != tuple(input_shape):
                raise CodegenBackendError(
                    "codegen addr expects input shape to match outer product output"
                )
        return OpNodeBuildResult(op_node)


class _BackendArangeHandler(ArangeHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        allowed_kwargs = {
            "start",
            "end",
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "step",
        }
        extra = set(node.kwargs) - allowed_kwargs
        if extra:
            raise CodegenBackendError(
                f"codegen {op_spec.name} got unexpected kwargs: {sorted(extra)}"
            )
        if node.kwargs.get("layout") is not None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects layout to be None"
            )
        device = node.kwargs.get("device")
        if device is not None and device != "cpu" and device != torch.device("cpu"):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects device to be None or cpu"
            )
        pin_memory = node.kwargs.get("pin_memory")
        if pin_memory not in (None, False):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects pin_memory to be False"
            )
        start_arg = None
        end_arg = None
        step_arg = None
        if node.args:
            if len(node.args) > 3:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects start and end arguments"
                )
            start_arg = node.args[0]
            if len(node.args) > 1:
                end_arg = node.args[1]
            if len(node.args) > 2:
                step_arg = node.args[2]
        if "start" in node.kwargs:
            if start_arg is not None:
                raise self._ctx.analysis_service.error_kwarg_specified_once(
                    op_spec.name, "start"
                )
            start_arg = node.kwargs["start"]
        if "end" in node.kwargs:
            if end_arg is not None:
                raise self._ctx.analysis_service.error_kwarg_specified_once(
                    op_spec.name, "end"
                )
            end_arg = node.kwargs["end"]
        if "step" in node.kwargs:
            if step_arg is not None:
                raise self._ctx.analysis_service.error_kwarg_specified_once(
                    op_spec.name, "step"
                )
            step_arg = node.kwargs["step"]
        if start_arg is None or end_arg is None:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects start and end arguments"
            )
        if step_arg is None:
            step_arg = 1
        start = self._ctx.analysis_service.resolve_scalar_arg(
            op_spec.name, start_arg, scalar_values
        )
        end = self._ctx.analysis_service.resolve_scalar_arg(
            op_spec.name, end_arg, scalar_values
        )
        step = self._ctx.analysis_service.resolve_scalar_arg(
            op_spec.name, step_arg, scalar_values
        )
        for name, value in (("start", start), ("end", end), ("step", step)):
            if not isinstance(value, numbers.Real):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to be an int or float"
                )
        dtype_spec = parse_arange_dtype(
            op_spec.name, node.kwargs.get("dtype"), dtype_info, start, end, step
        )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[],
            output_shape=(),
            params={"start": start, "end": end, "step": step},
        )
        output_shape = self._ctx.analysis_service.infer_output_shape(op_node, [])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_spec.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node, dtype_spec)


class _BackendConcatHandler(ConcatHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        input_nodes, concat_dim = parse_concat_args(node)
        input_shapes: List[tuple[int, ...]] = []
        for arg in input_nodes:
            if arg not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor("cat")
            input_shapes.append(shapes[arg])
        if not input_shapes:
            raise CodegenBackendError(
                "codegen cat expects a non-empty tensor list input"
            )
        rank = len(input_shapes[0])
        if rank == 0:
            raise CodegenBackendError("codegen cat expects inputs with rank >= 1")
        if concat_dim < 0:
            concat_dim += rank
        if concat_dim < 0 or concat_dim >= rank:
            raise CodegenBackendError("codegen cat dim is out of range")
        for shape in input_shapes:
            if len(shape) != rank:
                raise CodegenBackendError(
                    "codegen cat expects inputs with the same rank"
                )
            for dim, size in enumerate(shape):
                if dim == concat_dim:
                    continue
                if size != input_shapes[0][dim]:
                    raise CodegenBackendError(
                        "codegen cat expects input shapes to match except in the concat dimension"
                    )
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                "codegen cat expects inputs to share the graph dtype"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=None,
            params={"dim": concat_dim},
        )
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, input_shapes
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendEmptyStridedHandler(EmptyStridedHandler):
    def infer_graph_dtype(
        self, node: torch.fx.Node, op_spec: _OpSpec
    ) -> torch.dtype | None:
        node_dtype = None
        if len(node.args) > 2:
            node_dtype = node.args[2]
        if "dtype" in node.kwargs:
            if node_dtype is not None:
                raise self._ctx.analysis_service.error_kwarg_specified_once(
                    op_spec.name, "dtype"
                )
            node_dtype = node.kwargs["dtype"]
        if isinstance(node_dtype, torch.fx.Node):
            raise CodegenBackendError(
                "codegen empty_strided expects dtype to be a constant"
            )
        if node_dtype is None:
            raise CodegenBackendError(
                "codegen empty_strided requires dtype when no tensor inputs are provided"
            )
        return node_dtype

    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        if len(node.args) < 2:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects size and stride arguments"
            )
        if len(node.args) > 7:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects at most seven arguments"
            )
        size_arg, stride_arg = node.args[:2]
        if isinstance(size_arg, torch.fx.Node) or isinstance(
            stride_arg, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects size and stride to be constants"
            )
        kwargs = dict(node.kwargs)
        positional_names = [
            "dtype",
            "layout",
            "device",
            "pin_memory",
            "requires_grad",
        ]
        for index, name in enumerate(positional_names, start=2):
            if len(node.args) > index:
                if name in kwargs:
                    raise self._ctx.analysis_service.error_kwarg_specified_once(
                        op_spec.name, name
                    )
                kwargs[name] = node.args[index]
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
        dtype_value = kwargs.get("dtype")
        if dtype_value is not None and dtype_value is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects dtype to match the graph dtype"
            )
        for name in ("layout", "device"):
            if kwargs.get(name) is not None:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to be None"
                )
        for name in ("pin_memory", "requires_grad"):
            if kwargs.get(name) not in (None, False):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects {name} to be False"
                )
        output_shape = parse_resize_size(op_spec.name, size_arg)
        output_strides = parse_empty_strided_stride(op_spec.name, stride_arg)
        if len(output_shape) != len(output_strides):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects size and stride to match length"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[],
            output_shape=(),
            params={"size": output_shape},
        )
        output_shape = self._ctx.analysis_service.infer_output_shape(op_node, [])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = output_strides
        return OpNodeBuildResult(op_node)


def build_handlers(context: TensorContext) -> Dict[OpKind, OpKindHandler]:
    return {
        OpKind.ARANGE: _BackendArangeHandler(context, ArangeEmitter()),
        OpKind.FLIP: FlipHandler(
            context,
            FlipEmitter(),
            builder=_build_with_dtype(context, handle_flip_node),
        ),
        OpKind.PAD: PadHandler(
            context,
            PadEmitter(),
            builder=_build_with_dtype(context, handle_constant_pad_node),
        ),
        OpKind.VIEW: ViewHandler(
            context,
            ViewEmitter(),
            builder=_build_with_dtype(context, handle_view_node),
        ),
        OpKind.RESIZE: ResizeHandler(
            context,
            ResizeEmitter(),
            builder=_build_with_inplace(context, handle_resize_node),
        ),
        OpKind.MATMUL: _BackendMatmulHandler(context, MatmulEmitter()),
        OpKind.ADDR: _BackendAddrHandler(context, AddrEmitter()),
        OpKind.CONCAT: _BackendConcatHandler(context, ConcatEmitter()),
        OpKind.EMPTY_STRIDED: _BackendEmptyStridedHandler(
            context, EmptyStridedEmitter()
        ),
        OpKind.DIAGONAL: DiagonalHandler(
            context,
            DiagonalEmitter(),
            builder=_build_with_dtype(context, handle_diagonal_node),
        ),
        OpKind.CUMSUM: CumsumHandler(
            context,
            CumsumEmitter(),
            builder=_build_with_dtype(context, handle_cumsum_node),
        ),
        OpKind.GATHER: GatherHandler(
            context,
            GatherEmitter(),
            builder=_build_with_dtype(context, handle_gather_node),
        ),
        OpKind.COL2IM: Col2imHandler(
            context,
            Col2imEmitter(),
            builder=_build_col2im_unavailable(),
        ),
        OpKind.BATCH_NORM: BatchNormHandler(
            context,
            BatchNormEmitter(),
            builder=_build_with_scalar(context, handle_batch_norm_node),
        ),
        OpKind.PDIST: PdistHandler(
            context,
            PdistEmitter(),
            builder=_build_with_dtype(context, handle_pdist_node),
        ),
        OpKind.CDIST: CdistHandler(
            context,
            CdistEmitter(),
            builder=_build_with_dtype(context, handle_cdist_node),
        ),
        OpKind.ADDMM: AddmmHandler(
            context,
            AddmmEmitter(),
            builder=_build_with_inplace(context, handle_addmm_like_node),
        ),
        OpKind.LINEAR: LinearHandler(
            context,
            LinearEmitter(),
            builder=_build_with_dtype(context, handle_linear_node),
        ),
        OpKind.ADDBMM: AddbmmHandler(
            context,
            AddbmmEmitter(),
            builder=_build_with_inplace(context, handle_addmm_like_node),
        ),
        OpKind.ADDMV: AddmvHandler(
            context,
            AddmvEmitter(),
            builder=_build_with_inplace(context, handle_addmm_like_node),
        ),
    }

def _build_with_dtype(context, func):
    def builder(
        node,
        op_spec,
        dtype_info,
        shapes,
        strides,
        dtypes,
        scalar_values,
        inplace_input,
    ):
        if dtype_info is None:
            return None
        op_node = func(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            infer_output_shape=context.analysis_service.infer_output_shape,
        )
        return OpNodeBuildResult(op_node)

    return builder


def _build_with_scalar(context, func):
    def builder(
        node,
        op_spec,
        dtype_info,
        shapes,
        strides,
        dtypes,
        scalar_values,
        inplace_input,
    ):
        if dtype_info is None:
            return None
        op_node = func(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            scalar_values,
            infer_output_shape=context.analysis_service.infer_output_shape,
        )
        return OpNodeBuildResult(op_node)

    return builder


def _build_with_inplace(context, func):
    def builder(
        node,
        op_spec,
        dtype_info,
        shapes,
        strides,
        dtypes,
        scalar_values,
        inplace_input,
    ):
        if dtype_info is None:
            return None
        op_node = func(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            inplace_input,
            infer_output_shape=context.analysis_service.infer_output_shape,
        )
        return OpNodeBuildResult(op_node)

    return builder


def _build_col2im_unavailable():
    def builder(
        node,
        op_spec,
        dtype_info,
        shapes,
        strides,
        dtypes,
        scalar_values,
        inplace_input,
    ):
        raise CodegenBackendError("col2im handler is not available")

    return builder


class TensorKindHandlerFactory:
    def build_handlers(
        self, context_provider: HandlerContextProvider
    ) -> Dict[OpKind, OpKindHandler]:
        return build_handlers(context_provider.tensor)


def build_kind_handler_registrations() -> Dict[OpKind, KindHandlerRegistration]:
    return {
        OpKind.MATMUL: KindHandlerRegistration(
            _BackendMatmulHandler, MatmulEmitter
        ),
        OpKind.ADDR: KindHandlerRegistration(_BackendAddrHandler, AddrEmitter),
        OpKind.ARANGE: KindHandlerRegistration(
            _BackendArangeHandler, ArangeEmitter
        ),
        OpKind.CONCAT: KindHandlerRegistration(
            _BackendConcatHandler, ConcatEmitter
        ),
        OpKind.EMPTY_STRIDED: KindHandlerRegistration(
            _BackendEmptyStridedHandler, EmptyStridedEmitter
        ),
    }


__all__ = [
    "TensorKindHandlerFactory",
    "build_handlers",
    "build_kind_handler_registrations",
]
