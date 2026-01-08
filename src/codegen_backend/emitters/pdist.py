from __future__ import annotations

from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_pdist_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    pdist_template = get_template_env().get_template("pdist_kernel.c.j2")
    n, m = input_shape
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pdist_template.render(
        signature=signature,
        n=n,
        m=m,
        c_type=dtype.c_type,
        sqrt_fn=f"{dtype.scalar_prefix}sqrt",
    )
    return rendered.strip().splitlines()


class PdistEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("pdist requires op spec and dtype")
        return _write_pdist_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            dtype,
        )
