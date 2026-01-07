from __future__ import annotations

from types import SimpleNamespace

import torch.fx

from codegen_backend.graph import _OpNode
from codegen_backend.kinds import build_kind_handlers
from codegen_backend.ops_registry import SUPPORTED_OPS


def _make_op_node(
    op_name: str,
    input_count: int,
    *,
    params: dict[str, object] | None = None,
    reduction_dims: tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> _OpNode:
    graph = torch.fx.Graph()
    inputs = [graph.placeholder(f"arg{idx}") for idx in range(input_count)]
    output_node = graph.placeholder("out")
    return _OpNode(
        node=output_node,
        spec=SUPPORTED_OPS[op_name],
        inputs=inputs,
        output_shape=(),
        reduction_dims=reduction_dims,
        keepdim=keepdim,
        params=params or {},
    )


def test_infer_output_shape_by_handler() -> None:
    handlers = build_kind_handlers(SimpleNamespace())
    cases = [
        ("arange", [], {"start": 0, "end": 10, "step": 2}, None, False, (5,)),
        ("add", [(2, 3), (1, 3)], {}, None, False, (2, 3)),
        ("where", [(2, 3), (2, 3), (1, 3)], {}, None, False, (2, 3)),
        ("full_like", [(2, 3)], {"value": 1.0}, None, False, (2, 3)),
        ("flip", [(2, 3, 4)], {"dims": (1,)}, None, False, (2, 3, 4)),
        (
            "constant_pad_nd",
            [(2, 3)],
            {"pad_before": (1, 2), "pad_after": (1, 1)},
            None,
            False,
            (4, 6),
        ),
        ("as_strided", [(2, 3)], {"size": (2, 3)}, None, False, (2, 3)),
        ("squeeze", [(1, 2, 1)], {"squeeze_dims": (0, 2)}, None, False, (2,)),
        ("resize_", [(2, 3)], {"size": (3, 2)}, None, False, (3, 2)),
        ("empty_strided", [], {"size": (2, 2)}, None, False, (2, 2)),
        (
            "diagonal",
            [(3, 4)],
            {"offset": 0, "dim1": 0, "dim2": 1},
            None,
            False,
            (3,),
        ),
        (
            "sum",
            [(2, 3)],
            {"reduce_all": False},
            (1,),
            False,
            (2,),
        ),
        (
            "argmax",
            [(2, 3)],
            {"reduce_all": False},
            (1,),
            True,
            (2, 1),
        ),
        ("_softmax", [(2, 3)], {"dim": 1}, None, False, (2, 3)),
        ("cumsum", [(2, 3)], {"dim": 1}, None, False, (2, 3)),
        ("embedding", [(10, 4), (2, 3)], {"padding_idx": -1}, None, False, (2, 3, 4)),
        (
            "_embedding_bag",
            [(10, 4), (6,), (3,)],
            {"include_last_offset": False},
            None,
            False,
            (3, 4),
        ),
        ("gather", [(2, 3), (2, 3)], {"dim": 1}, None, False, (2, 3)),
        ("cat", [(2, 3), (2, 5)], {"dim": 1}, None, False, (2, 8)),
        (
            "max_pool2d",
            [(1, 3, 5, 5)],
            {
                "kernel_size": (2, 2),
                "stride": (2, 2),
                "padding": (0, 0),
                "dilation": (1, 1),
                "ceil_mode": False,
            },
            None,
            False,
            (1, 3, 2, 2),
        ),
        (
            "adaptive_avg_pool3d",
            [(1, 3, 6, 6, 6)],
            {
                "kernel_size": (2, 2, 2),
                "stride": (2, 2, 2),
                "padding": (0, 0, 0),
                "dilation": (1, 1, 1),
            },
            None,
            False,
            (1, 3, 3, 3, 3),
        ),
        (
            "_adaptive_avg_pool2d_backward",
            [(1, 3, 4, 4), (1, 3, 8, 8)],
            {"kernel_size": (2, 2), "stride": (2, 2)},
            None,
            False,
            (1, 3, 8, 8),
        ),
        (
            "max_pool1d",
            [(1, 3, 10)],
            {
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "ceil_mode": False,
            },
            None,
            False,
            (1, 3, 5),
        ),
        (
            "col2im",
            [(1, 8, 4)],
            {
                "output_size": (3, 3),
                "kernel_size": (2, 2),
                "dilation": (1, 1),
                "padding": (0, 0),
                "stride": (1, 1),
            },
            None,
            False,
            (1, 2, 3, 3),
        ),
        ("_native_batch_norm_legit", [(2, 3, 4, 4)], {}, None, False, (2, 3, 4, 4)),
        ("_pdist_forward", [(5, 4)], {}, None, False, (10,)),
        (
            "_cdist_forward",
            [(5, 3), (4, 3)],
            {},
            None,
            False,
            (5, 4),
        ),
        (
            "conv1d",
            [(2, 4, 10), (6, 4, 3)],
            {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
            None,
            False,
            (2, 6, 8),
        ),
        (
            "conv2d",
            [(1, 3, 8, 8), (6, 3, 3, 3)],
            {
                "stride": (1, 1),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
                "transposed": False,
                "output_padding": (0, 0),
            },
            None,
            False,
            (1, 6, 6, 6),
        ),
        ("addmm", [(), (2, 3), (3, 4)], {}, None, False, (2, 4)),
        ("addbmm", [(), (2, 2, 3), (2, 3, 4)], {}, None, False, (2, 4)),
        ("addmv", [(), (2, 3), (3,)], {}, None, False, (2,)),
        ("addr", [(), (2,), (3,)], {}, None, False, (2, 3)),
        ("matmul", [(2, 3), (3, 4)], {}, None, False, (2, 4)),
    ]

    for (
        op_name,
        input_shapes,
        params,
        reduction_dims,
        keepdim,
        expected_shape,
    ) in cases:
        op_node = _make_op_node(
            op_name,
            len(input_shapes),
            params=params,
            reduction_dims=reduction_dims,
            keepdim=keepdim,
        )
        handler = handlers[op_node.spec.kind]
        output_shape = handler.infer_output_shape(op_node, input_shapes)
        assert output_shape == expected_shape
