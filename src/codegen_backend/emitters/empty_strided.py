from __future__ import annotations

from typing import List, Sequence

from codegen_backend.errors import CodegenBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec


def _write_empty_strided_kernel(
    node_index: int,
    op_spec: _OpSpec,
    output_shape: Sequence[int],
    dtype: _CodegenDType,
) -> List[str]:
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    return [signature, "    (void)out;", "}"]


class EmptyStridedEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("empty_strided requires op spec and dtype")
        return _write_empty_strided_kernel(
            req.node_index, op_spec, req.output_shape, dtype
        )
