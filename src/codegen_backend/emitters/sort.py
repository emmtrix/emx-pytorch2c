from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.c_types import _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix, _is_contiguous
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_sort_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    input_dtype: torch.dtype,
    sort_dim: int,
    descending: bool,
    dtype: _CodegenDType,
) -> List[str]:
    sort_template = get_template_env().get_template("sort_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    out_suffix = _format_array_suffix(output_shape)
    input_c_type = _input_c_type(input_dtype, dtype)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {input_c_type} input{input_suffix}, "
        f"{dtype.c_type} out{out_suffix}) {{"
    )
    input_is_contiguous = _is_contiguous(input_shape, input_strides)
    output_is_contiguous = _is_contiguous(output_shape, output_strides)
    if not output_shape:
        input_access = (
            "input[0]" if input_is_contiguous else f"(({input_c_type}*)input)[0]"
        )
        output_access = (
            "out[0]" if output_is_contiguous else f"(({dtype.c_type}*)out)[0]"
        )
        rendered = sort_template.render(
            signature=signature,
            input_access=input_access,
            output_access=output_access,
            output_shape_len=0,
        )
        return rendered.splitlines()

    sort_size = output_shape[sort_dim]
    outer_dims = [
        (dim, size) for dim, size in enumerate(output_shape) if dim != sort_dim
    ]

    def _indices_for(expr: str) -> List[str]:
        return [
            expr if dim == sort_dim else f"i{dim}"
            for dim in range(len(output_shape))
        ]

    input_indices = _indices_for("k")
    output_indices = _indices_for("k")
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        input_is_contiguous,
        sizes=input_shape,
        c_type=input_c_type,
    )
    output_access = _emit_strided_access(
        "out",
        output_indices,
        output_strides,
        output_is_contiguous,
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    output_access_a = _emit_strided_access(
        "out",
        _indices_for("j"),
        output_strides,
        output_is_contiguous,
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    output_access_b = _emit_strided_access(
        "out",
        _indices_for("j + 1"),
        output_strides,
        output_is_contiguous,
        sizes=output_shape,
        c_type=dtype.c_type,
    )
    compare = f"a < b" if descending else "a > b"
    rendered = sort_template.render(
        signature=signature,
        compare=compare,
        dtype_c_type=dtype.c_type,
        input_access=input_access,
        output_access=output_access,
        output_access_a=output_access_a,
        output_access_b=output_access_b,
        output_shape_len=len(output_shape),
        outer_dims=outer_dims,
        sort_size=sort_size,
    )
    return rendered.splitlines()


class SortEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("sort requires op spec and dtype")
        return _write_sort_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[0],
            int(req.params["dim"]),
            bool(req.params.get("descending", False)),
            dtype,
        )
