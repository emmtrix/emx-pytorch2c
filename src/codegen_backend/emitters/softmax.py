from __future__ import annotations

from typing import List, Sequence

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.indexing import _emit_strided_access, _format_output_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_softmax_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_strides: Sequence[int],
    softmax_dim: int | None,
    dtype: _CodegenDType,
) -> List[str]:
    if softmax_dim is None:
        raise RefBackendError("codegen softmax expects a reduction dimension")
    softmax_template = get_template_env().get_template("softmax_kernel.c.j2")
    rank = len(input_shape)
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(input_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    output_dims = [
        {"dim": dim, "size": size} for dim, size in enumerate(input_shape)
    ]
    input_contig = _is_contiguous(input_shape, input_strides)
    current_indices = [f"i{dim}" for dim in range(rank)]
    r_indices = current_indices.copy()
    r_indices[softmax_dim] = f"r{softmax_dim}"
    zero_indices = current_indices.copy()
    zero_indices[softmax_dim] = "0"
    input_access_r = _emit_strided_access(
        "input",
        r_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    input_access_zero = _emit_strided_access(
        "input",
        zero_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    input_access_current = _emit_strided_access(
        "input",
        current_indices,
        input_strides,
        input_contig,
        sizes=input_shape,
        c_type=dtype.c_type,
    )
    output_access = _format_output_access(
        "out", input_shape, output_strides, c_type=dtype.c_type
    )
    rendered = softmax_template.render(
        signature=signature,
        output_dims=output_dims,
        softmax_dim=softmax_dim,
        softmax_size=input_shape[softmax_dim],
        c_type=dtype.c_type,
        input_access_zero=input_access_zero,
        input_access_r=input_access_r,
        input_access_current=input_access_current,
        output_access=output_access,
        is_log=op_spec.name in {"log_softmax", "_log_softmax"},
    )
    return rendered.strip().splitlines()


class SoftmaxEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("softmax requires op spec and dtype")
        return _write_softmax_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_strides or (),
            int(req.params["dim"]) if req.params.get("dim") is not None else None,
            dtype,
        )
