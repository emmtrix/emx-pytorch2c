from __future__ import annotations

import math
from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.c_types import _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.indexing import _contiguous_strides, _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _format_strided_access(
    name: str,
    shape: Sequence[int],
    strides: Sequence[int],
    indices: Sequence[str],
    dtype: _CodegenDType,
    *,
    contiguous: bool,
) -> str:
    if contiguous:
        return f"{name}{''.join(f'[{idx}]' for idx in indices)}"
    return _emit_strided_access(
        name,
        indices,
        strides,
        contig=False,
        sizes=shape,
        c_type=dtype.c_type,
    )


def _write_cdist_kernel(
    node_index: int,
    op_spec: _OpSpec,
    x1_shape: Sequence[int],
    x2_shape: Sequence[int],
    x1_strides: Sequence[int],
    x2_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    dtype: _CodegenDType,
    *,
    p_value: float,
    compute_mode: str | None,
) -> List[str]:
    cdist_template = get_template_env().get_template("cdist_kernel.c.j2")
    n, m = x1_shape[-2:]
    r = x2_shape[-2]
    x1_suffix = _format_array_suffix(x1_shape)
    x2_suffix = _format_array_suffix(x2_shape)
    output_suffix = _format_array_suffix(output_shape)
    batch_rank = len(output_shape) - 2
    batch_indices = [f"b{dim}" for dim in range(batch_rank)]
    x1_batch_rank = len(x1_shape) - 2
    x2_batch_rank = len(x2_shape) - 2
    x1_offset = batch_rank - x1_batch_rank
    x2_offset = batch_rank - x2_batch_rank
    x1_indices = batch_indices[x1_offset:] + ["i", "k"]
    x2_indices = batch_indices[x2_offset:] + ["j", "k"]
    output_indices = batch_indices + ["i", "j"]
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} x1{x1_suffix}, "
        f"const {dtype.c_type} x2{x2_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    p_zero = math.isclose(p_value, 0.0)
    p_two = math.isclose(p_value, 2.0)
    p_inf = math.isinf(p_value)
    use_mm = p_two and compute_mode in (
        "use_mm_for_euclid_dist",
        "use_mm_for_euclid_dist_if_necessary",
    )
    rendered = cdist_template.render(
        signature=signature,
        batch_dims=[
            {"index": idx, "size": size}
            for idx, size in zip(batch_indices, output_shape[:-2])
        ],
        n=n,
        m=m,
        r=r,
        c_type=dtype.c_type,
        sqrt_fn=f"{dtype.scalar_prefix}sqrt",
        abs_fn=f"{dtype.scalar_prefix}abs",
        max_fn=f"{dtype.scalar_prefix}fmax",
        pow_fn=f"{dtype.scalar_prefix}pow",
        x1_access=_format_strided_access(
            "x1",
            x1_shape,
            x1_strides,
            x1_indices,
            dtype,
            contiguous=False,
        ),
        x2_access=_format_strided_access(
            "x2",
            x2_shape,
            x2_strides,
            x2_indices,
            dtype,
            contiguous=False,
        ),
        output_access=_format_strided_access(
            "out",
            output_shape,
            output_strides,
            output_indices,
            dtype,
            contiguous=_is_contiguous(output_shape, output_strides),
        ),
        p_zero=p_zero,
        p_two=p_two,
        p_inf=p_inf,
        use_mm=use_mm,
        p_literal=_format_scalar_literal(p_value, dtype),
        inv_p_literal=(
            _format_scalar_literal(1.0 / p_value, dtype)
            if not (p_zero or p_inf)
            else None
        ),
        zero_literal=_format_scalar_literal(0.0, dtype),
        one_literal=_format_scalar_literal(1.0, dtype),
    )
    return rendered.strip().splitlines()


class CdistEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("cdist requires op spec and dtype")
        p_value = float(req.params.get("p", 2.0))
        compute_mode = req.params.get("compute_mode")
        output_strides = req.output_strides
        if output_strides is None:
            output_strides = _contiguous_strides(req.output_shape)
        return _write_cdist_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.input_strides[0],
            req.input_strides[1],
            req.output_shape,
            output_strides,
            dtype,
            p_value=p_value,
            compute_mode=compute_mode,
        )
