from __future__ import annotations

from typing import Dict, List

from codegen_backend.c_types import _input_c_type
from codegen_backend.emitters.base import (
    _format_array_suffix,
    _is_contiguous,
    KindEmitterBase,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.templates import get_template_env


class ArgReductionEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        template = get_template_env().get_template("argreduction_kernel.c.j2")
        input_shape = req.input_shapes[0]
        input_strides = req.input_strides[0]
        output_shape = req.output_shape
        output_strides = req.output_strides
        reduction_dims = req.reduction_dims or ()
        keepdim = bool(req.keepdim)
        reduce_all = bool(req.params.get("reduce_all", False))
        input_c_type = _input_c_type(req.dtype.torch_dtype, req.dtype)
        signature = (
            f"void node{req.node_index}_{req.op_spec.name}_{req.dtype.suffix}("
            f"const {input_c_type} a{_format_array_suffix(input_shape)}, "
            f"int64_t out{_format_array_suffix(output_shape)}) {{"
        )
        output_dims = [
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ]
        output_access = KindEmitterBase.emit_output_access(
            output_shape, output_strides, c_type="int64_t"
        )
        if not input_shape:
            rendered = template.render(
                signature=signature,
                input_rank=0,
                reduce_all=False,
                output_dims=output_dims,
                output_access=output_access,
                input_c_type=input_c_type,
                compare_op=">",
                reduction_dims=[],
                reduction_dim=0,
                reduction_size=0,
                init_access="0",
                loop_access="0",
                input_access="0",
                linear_index="0",
            )
            return rendered.strip().splitlines()
        a_is_contiguous = _is_contiguous(input_shape, input_strides)
        compare_op = ">" if req.op_spec.name == "argmax" else "<"

        def linear_index_expr() -> str:
            expr = "r0" if input_shape else "0"
            for dim in range(1, len(input_shape)):
                expr = f"({expr} * {input_shape[dim]} + r{dim})"
            return expr

        if reduce_all:
            reduction_dims_for_template = [
                {"dim": dim, "size": size} for dim, size in enumerate(input_shape)
            ]
            input_access = _emit_strided_access(
                "a",
                [f"r{dim}" for dim in range(len(input_shape))],
                input_strides,
                contig=a_is_contiguous,
                sizes=input_shape,
                c_type=input_c_type,
            )
            rendered = template.render(
                signature=signature,
                input_rank=len(input_shape),
                reduce_all=True,
                output_dims=output_dims,
                output_access=output_access,
                input_c_type=input_c_type,
                compare_op=compare_op,
                reduction_dims=reduction_dims_for_template,
                reduction_dim=0,
                reduction_size=0,
                init_access="0",
                loop_access="0",
                input_access=input_access,
                linear_index=linear_index_expr(),
            )
            return rendered.strip().splitlines()

        reduction_dim = reduction_dims[0]
        dim_to_output: Dict[int, int] = {}
        if not keepdim:
            dim_to_output = KindEmitterBase.map_reduction_dims(
                len(input_shape), (reduction_dim,)
            )
        init_indices = []
        loop_indices = []
        for dim in range(len(input_shape)):
            if dim == reduction_dim:
                init_indices.append("0")
                loop_indices.append(f"r{dim}")
            else:
                idx = f"i{dim}" if keepdim else f"i{dim_to_output[dim]}"
                init_indices.append(idx)
                loop_indices.append(idx)
        init_access = _emit_strided_access(
            "a",
            init_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=input_c_type,
        )
        loop_access = _emit_strided_access(
            "a",
            loop_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=input_c_type,
        )
        rendered = template.render(
            signature=signature,
            input_rank=len(input_shape),
            reduce_all=False,
            output_dims=output_dims,
            output_access=output_access,
            input_c_type=input_c_type,
            compare_op=compare_op,
            reduction_dims=[],
            reduction_dim=reduction_dim,
            reduction_size=input_shape[reduction_dim],
            init_access=init_access,
            loop_access=loop_access,
            input_access="0",
            linear_index="0",
        )
        return rendered.strip().splitlines()
