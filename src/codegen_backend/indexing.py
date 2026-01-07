from __future__ import annotations

from typing import Optional, Sequence, Tuple


def _broadcast_index_expr(
    input_shape: Sequence[int], output_shape: Sequence[int]
) -> str:
    output_rank = len(output_shape)
    input_rank = len(input_shape)
    if input_rank == 0:
        return "[0]"
    index_expr = []
    offset = output_rank - input_rank
    for input_dim in range(input_rank):
        output_dim = input_dim + offset
        if input_shape[input_dim] == 1:
            index_expr.append("[0]")
        else:
            index_expr.append(f"[i{output_dim}]")
    return "".join(index_expr)


def _contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    if not shape:
        return ()
    strides = [0] * len(shape)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = stride
        stride *= max(shape[dim], 1)
    return tuple(strides)


def _emit_strided_access(
    name: str,
    indices: Sequence[str],
    strides: Sequence[int],
    contig: bool,
    sizes: Optional[Sequence[int]] = None,
    *,
    c_type: str = "float",
) -> str:
    if contig:
        return f"{name}{''.join(f'[{idx}]' for idx in indices)}"
    terms = []
    for idx_name, stride, size in zip(
        indices, strides, sizes or [None] * len(indices)
    ):
        if size == 1:
            continue
        terms.append(f"{idx_name} * {stride}")
    index_expr = " + ".join(terms) if terms else "0"
    return f"(({c_type}*){name})[{index_expr}]"


def _format_strided_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    c_type: str = "float",
) -> str:
    output_rank = len(output_shape)
    input_rank = len(input_shape)
    if input_rank == 0:
        return f"(({c_type}*){name})[0]"
    offset = output_rank - input_rank
    indices = [f"i{input_dim + offset}" for input_dim in range(input_rank)]
    return _emit_strided_access(
        name,
        indices,
        input_strides,
        contig=False,
        sizes=input_shape,
        c_type=c_type,
    )


def _format_output_access(
    name: str,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    *,
    c_type: str = "float",
) -> str:
    if not output_shape:
        return f"(({c_type}*){name})[0]"
    terms = []
    for dim, stride in enumerate(output_strides):
        if output_shape[dim] == 1:
            continue
        terms.append(f"i{dim} * {stride}")
    index_expr = " + ".join(terms) if terms else "0"
    return f"(({c_type}*){name})[{index_expr}]"


def format_input_access(
    name: str,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    *,
    broadcast_contiguous: bool,
    c_type: str,
    input_is_contiguous: bool,
) -> str:
    if input_is_contiguous:
        if broadcast_contiguous:
            return f"{name}{_broadcast_index_expr(input_shape, output_shape)}"
        return (
            f"{name}{''.join(f'[i{dim}]' for dim in range(len(output_shape))) or '[0]'}"
        )
    return _format_strided_access(
        name, input_shape, input_strides, output_shape, c_type=c_type
    )


def format_output_access(
    name: str,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    *,
    c_type: str,
    output_is_contiguous: bool,
) -> str:
    if output_is_contiguous:
        output_access = (
            "".join(f"[i{dim}]" for dim in range(len(output_shape))) or "[0]"
        )
        return f"{name}{output_access}"
    return _format_output_access(name, output_shape, output_strides, c_type=c_type)
