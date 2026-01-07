from __future__ import annotations

from typing import List

from codegen_backend.dtypes import _INTEGER_CODEGEN_DTYPES
from codegen_backend.emitters.base import _format_array_suffix, _is_contiguous, KindEmitterBase
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.templates import get_template_env


class MatmulEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        matmul_template = get_template_env().get_template("matmul_kernel.c.j2")
        a_shape, b_shape = req.input_shapes
        a_strides, b_strides = req.input_strides
        dtype = req.dtype
        a_is_contiguous = _is_contiguous(a_shape, a_strides)
        b_is_contiguous = _is_contiguous(b_shape, b_strides)
        acc_type = dtype.c_type
        acc_init = "0" if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES else "0.0f"

        if req.op_spec.name == "matmul":
            if len(a_shape) == 1:
                k = a_shape[0]
                a_suffix = _format_array_suffix((k,))
                b_suffix = _format_array_suffix((k,))
                out_suffix = _format_array_suffix(())
                rendered = matmul_template.render(
                    signature=(
                        f"void node{req.node_index}_{req.op_spec.name}_{dtype.suffix}("
                        f"const {dtype.c_type} a{a_suffix}, "
                        f"const {dtype.c_type} b{b_suffix}, "
                        f"{dtype.c_type} out{out_suffix}) {{"
                    ),
                    batch=None,
                    m=1,
                    n=1,
                    k=k,
                    acc_type=acc_type,
                    acc_init=acc_init,
                    a_access=_emit_strided_access(
                        "a",
                        ("t",),
                        a_strides,
                        a_is_contiguous,
                        sizes=a_shape,
                        c_type=dtype.c_type,
                    ),
                    b_access=_emit_strided_access(
                        "b",
                        ("t",),
                        b_strides,
                        b_is_contiguous,
                        sizes=b_shape,
                        c_type=dtype.c_type,
                    ),
                    out_access="out[0]",
                )
                return rendered.strip().splitlines()
            m, k = a_shape
            _, n = b_shape
            a_suffix = _format_array_suffix((m, k))
            b_suffix = _format_array_suffix((k, n))
            out_suffix = _format_array_suffix((m, n))
            rendered = matmul_template.render(
                signature=(
                    f"void node{req.node_index}_{req.op_spec.name}_{dtype.suffix}("
                    f"const {dtype.c_type} a{a_suffix}, "
                    f"const {dtype.c_type} b{b_suffix}, "
                    f"{dtype.c_type} out{out_suffix}) {{"
                ),
                batch=None,
                m=m,
                n=n,
                k=k,
                acc_type=acc_type,
                acc_init=acc_init,
                a_access=_emit_strided_access(
                    "a",
                    ("i", "t"),
                    a_strides,
                    a_is_contiguous,
                    sizes=a_shape,
                    c_type=dtype.c_type,
                ),
                b_access=_emit_strided_access(
                    "b",
                    ("t", "j"),
                    b_strides,
                    b_is_contiguous,
                    sizes=b_shape,
                    c_type=dtype.c_type,
                ),
                out_access="out[i][j]",
            )
            return rendered.strip().splitlines()
        batch, m, k = a_shape
        _, _, n = b_shape
        a_suffix = _format_array_suffix((batch, m, k))
        b_suffix = _format_array_suffix((batch, k, n))
        out_suffix = _format_array_suffix((batch, m, n))
        rendered = matmul_template.render(
            signature=(
                f"void node{req.node_index}_{req.op_spec.name}_{dtype.suffix}("
                f"const {dtype.c_type} a{a_suffix}, "
                f"const {dtype.c_type} b{b_suffix}, "
                f"{dtype.c_type} out{out_suffix}) {{"
            ),
            batch=batch,
            m=m,
            n=n,
            k=k,
            acc_type=acc_type,
            acc_init=acc_init,
            a_access=_emit_strided_access(
                "a",
                ("b_idx", "i", "t"),
                a_strides,
                a_is_contiguous,
                sizes=a_shape,
                c_type=dtype.c_type,
            ),
            b_access=_emit_strided_access(
                "b",
                ("b_idx", "t", "j"),
                b_strides,
                b_is_contiguous,
                sizes=b_shape,
                c_type=dtype.c_type,
            ),
            out_access="out[b_idx][i][j]",
        )
        return rendered.strip().splitlines()
