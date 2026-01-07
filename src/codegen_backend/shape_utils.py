from __future__ import annotations

from typing import Sequence, Tuple

from codegen_backend.errors import CodegenBackendError


def broadcast_output_shape(
    op_name: str, *input_shapes: Sequence[int]
) -> Tuple[int, ...]:
    if not input_shapes:
        return ()
    max_len = max(len(shape) for shape in input_shapes)
    output_shape = []
    for dim in range(1, max_len + 1):
        sizes = [
            shape[-dim] if dim <= len(shape) else 1 for shape in input_shapes
        ]
        max_size = max(sizes)
        if any(size not in (1, max_size) for size in sizes):
            raise CodegenBackendError(
                f"codegen {op_name} requires inputs to be broadcastable"
            )
        output_shape.append(max_size)
    return tuple(reversed(output_shape))
