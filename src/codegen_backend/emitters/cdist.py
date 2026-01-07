from __future__ import annotations

from typing import List, Sequence

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_cdist_kernel(
    node_index: int,
    op_spec: _OpSpec,
    x1_shape: Sequence[int],
    x2_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    cdist_template = get_template_env().get_template("cdist_kernel.c.j2")
    n, m = x1_shape
    r, _ = x2_shape
    x1_suffix = _format_array_suffix(x1_shape)
    x2_suffix = _format_array_suffix(x2_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} x1{x1_suffix}, "
        f"const {dtype.c_type} x2{x2_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = cdist_template.render(
        signature=signature,
        n=n,
        m=m,
        r=r,
        c_type=dtype.c_type,
    )
    return rendered.strip().splitlines()


class CdistEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("cdist requires op spec and dtype")
        return _write_cdist_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.output_shape,
            dtype,
        )
