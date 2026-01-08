from __future__ import annotations

from typing import List, Sequence

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_split_with_sizes_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    split_dim: int,
    split_offset: int,
    dtype: _CodegenDType,
) -> List[str]:
    split_template = get_template_env().get_template(
        "split_with_sizes_kernel.c.j2"
    )
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    output_indices = [f"i{dim}" for dim in range(len(output_shape))]
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        _is_contiguous(output_shape, output_strides),
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    input_indices = [
        f"i{dim} + {split_offset}" if dim == split_dim else f"i{dim}"
        for dim in range(len(output_shape))
    ]
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    rendered = split_template.render(
        signature=signature,
        output_shape=output_shape,
        split_dim=split_dim,
        split_offset=split_offset,
        input_access=input_access,
        output_access=output_access,
    )
    return rendered.splitlines()


class SplitWithSizesEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError(
                "split_with_sizes requires op spec and dtype"
            )
        split_dim = int(req.params.get("dim", 0))
        split_offset = int(req.params.get("offset", 0))
        return _write_split_with_sizes_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_shape,
            req.output_strides or (),
            split_dim,
            split_offset,
            dtype,
        )
