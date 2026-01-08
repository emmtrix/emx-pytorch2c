from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch

from codegen_backend.dtypes import _INTEGER_CODEGEN_DTYPES, _CodegenDType
from codegen_backend.emitters.base import (
    _format_array_suffix,
    _is_contiguous,
    KindEmitterBase,
)
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env

_REDUCTION_CONFIG = {
    "sum": {
        "init_value": 0,
        "reduce_op": "+=",
        "post_op": None,
    },
    "mean": {
        "init_value": 0,
        "reduce_op": "+=",
        "post_op": "mean",
    },
    "prod": {
        "init_value": 1,
        "reduce_op": "*=",
        "post_op": None,
    },
    "any": {
        "init_value": 0,
        "reduce_op": "|=",
        "post_op": None,
        "bool_reduction": True,
    },
    "all": {
        "init_value": 1,
        "reduce_op": "&=",
        "post_op": None,
        "bool_reduction": True,
    },
    "amax": {
        "init_value": 0,
        "reduce_op": None,
        "post_op": None,
    },
    "max": {
        "init_value": 0,
        "reduce_op": None,
        "post_op": None,
    },
    "amin": {
        "init_value": 0,
        "reduce_op": None,
        "post_op": None,
    },
    "min": {
        "init_value": 0,
        "reduce_op": None,
        "post_op": None,
    },
}

_MINMAX_INIT_VALUES = {
    torch.float32: {
        "amax": "-INFINITY",
        "max": "-INFINITY",
        "amin": "INFINITY",
        "min": "INFINITY",
    },
    torch.int8: {
        "amax": "INT8_MIN",
        "max": "INT8_MIN",
        "amin": "INT8_MAX",
        "min": "INT8_MAX",
    },
    torch.int32: {
        "amax": "INT32_MIN",
        "max": "INT32_MIN",
        "amin": "INT32_MAX",
        "min": "INT32_MAX",
    },
}


def _build_reduction_input_indices(
    input_rank: int, reduction_dims: Tuple[int, ...], keepdim: bool
) -> List[str]:
    reduction_set = set(reduction_dims)
    if keepdim:
        return [
            f"r{dim}" if dim in reduction_set else f"i{dim}"
            for dim in range(input_rank)
        ]
    dim_to_output = KindEmitterBase.map_reduction_dims(
        input_rank, reduction_dims
    )
    return [
        f"r{dim}" if dim in reduction_set else f"i{dim_to_output[dim]}"
        for dim in range(input_rank)
    ]


def _write_std_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
    *,
    unbiased: bool,
) -> List[str]:
    std_template = get_template_env().get_template("std_kernel.c.j2")
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = KindEmitterBase.emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        input_indices = _build_reduction_input_indices(
            input_rank, reduction_dims, keepdim
        )
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    reduction_count = 1
    for dim in reduction_dims:
        reduction_count *= input_shape[dim]
    acc_type = dtype.c_type
    sqrt_fn = f"{dtype.scalar_prefix}sqrt"
    if dtype.torch_dtype is torch.bool or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        acc_type = "float"
        sqrt_fn = "ref_scalar_f32_sqrt"
    rendered = std_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        output_access=output_access,
        acc_type=acc_type,
        reduction_count=reduction_count,
        unbiased=int(unbiased),
        sqrt_fn=sqrt_fn,
    )
    return rendered.strip().splitlines()


def _write_var_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
    *,
    unbiased: bool,
) -> List[str]:
    var_template = get_template_env().get_template("var_kernel.c.j2")
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = KindEmitterBase.emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        input_indices = _build_reduction_input_indices(
            input_rank, reduction_dims, keepdim
        )
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    reduction_count = 1
    for dim in reduction_dims:
        reduction_count *= input_shape[dim]
    acc_type = dtype.c_type
    if dtype.torch_dtype is torch.bool or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        acc_type = "float"
    rendered = var_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        output_access=output_access,
        acc_type=acc_type,
        reduction_count=reduction_count,
        unbiased=int(unbiased),
    )
    return rendered.strip().splitlines()


def _write_norm_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
    *,
    p_value: float,
) -> List[str]:
    norm_template = get_template_env().get_template("norm_kernel.c.j2")
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = KindEmitterBase.emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        input_indices = _build_reduction_input_indices(
            input_rank, reduction_dims, keepdim
        )
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    acc_type = dtype.c_type
    abs_fn = f"{dtype.scalar_prefix}abs"
    pow_fn = f"{dtype.scalar_prefix}pow"
    is_zero_p = math.isclose(p_value, 0.0)
    if dtype.torch_dtype is torch.bool or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        acc_type = "float"
        abs_fn = "ref_scalar_f32_abs"
        pow_fn = "ref_scalar_f32_pow"
        input_access = f"(float){input_access}"
    p_literal = f"{float(p_value)}f"
    inv_p_literal = f"{(1.0 / p_value)}f" if not is_zero_p else "0.0f"
    rendered = norm_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        output_access=output_access,
        acc_type=acc_type,
        abs_fn=abs_fn,
        pow_fn=pow_fn,
        p_value=p_literal,
        inv_p_value=inv_p_literal,
        is_zero_p=is_zero_p,
    )
    return rendered.strip().splitlines()


def _write_reduction_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    input_strides: Sequence[int],
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    dtype: _CodegenDType,
) -> List[str]:
    reduction_template = get_template_env().get_template("sum_kernel.c.j2")
    config = _REDUCTION_CONFIG[op_spec.name]
    if dtype.torch_dtype is torch.bool:
        if op_spec.name in {"sum", "mean"}:
            config = {
                "init_value": 0,
                "reduce_op": "|=",
                "post_op": None,
                "bool_reduction": True,
            }
        elif op_spec.name in {"amax", "max"}:
            config = {
                "init_value": 0,
                "reduce_op": "|=",
                "post_op": None,
                "bool_reduction": True,
            }
        elif op_spec.name in {"amin", "min"}:
            config = {
                "init_value": 1,
                "reduce_op": "&=",
                "post_op": None,
                "bool_reduction": True,
            }
        elif op_spec.name == "prod":
            config = {
                "init_value": 1,
                "reduce_op": "&=",
                "post_op": None,
                "bool_reduction": True,
            }
    a_is_contiguous = _is_contiguous(input_shape, input_strides)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} a{_format_array_suffix(input_shape)}, "
        f"{dtype.c_type} out{_format_array_suffix(output_shape)}) {{"
    )
    output_access = KindEmitterBase.emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if input_rank == 0:
        input_access = "a[0]"
    else:
        input_indices = _build_reduction_input_indices(
            input_rank, reduction_dims, keepdim
        )
        input_access = _emit_strided_access(
            "a",
            input_indices,
            input_strides,
            contig=a_is_contiguous,
            sizes=input_shape,
            c_type=dtype.c_type,
        )
    compare_op = None
    isnan_fn = None
    if op_spec.name == "max":
        minmax_name = "amax"
    elif op_spec.name == "min":
        minmax_name = "amin"
    else:
        minmax_name = op_spec.name
    if minmax_name in {"amax", "amin"} and dtype.torch_dtype is not torch.bool:
        compare_op = ">" if minmax_name == "amax" else "<"
        init_value_config = _MINMAX_INIT_VALUES[dtype.torch_dtype][minmax_name]
        config = {
            "init_value": init_value_config,
            "post_op": None,
        }
        isnan_fn = (
            f"{dtype.scalar_prefix}isnan"
            if dtype.torch_dtype is torch.float32
            else None
        )
    reduction_count = 1
    for dim in reduction_dims:
        reduction_count *= input_shape[dim]
    bool_reduction = config.get("bool_reduction", False)
    acc_type = "int32_t" if bool_reduction else dtype.c_type
    init_value_config = config["init_value"]
    if isinstance(init_value_config, str):
        init_value = init_value_config
    elif bool_reduction or dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
        init_value = str(init_value_config)
    else:
        init_value = f"{init_value_config}.0f"
    post_op = None
    if config["post_op"] == "mean":
        if dtype.torch_dtype in _INTEGER_CODEGEN_DTYPES:
            post_op = f"acc /= {reduction_count};"
        else:
            post_op = f"acc /= (float){reduction_count};"
    rendered = reduction_template.render(
        signature=signature,
        input_rank=input_rank,
        output_rank=output_rank,
        output_dims=[
            {"dim": dim, "size": size} for dim, size in enumerate(output_shape)
        ],
        reduction_dims=[
            {"dim": dim, "size": input_shape[dim]} for dim in reduction_dims
        ],
        input_access=input_access,
        bool_reduction=bool_reduction,
        reduce_op=config.get("reduce_op"),
        is_minmax=minmax_name in {"amax", "amin"}
        and dtype.torch_dtype is not torch.bool,
        compare_op=compare_op if minmax_name in {"amax", "amin"} else None,
        is_float=dtype.torch_dtype is torch.float32,
        isnan_fn=isnan_fn,
        output_access=output_access,
        acc_type=acc_type,
        init_value=init_value,
        post_op=post_op,
    )
    return rendered.strip().splitlines()


class ReductionEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        reduction_dims = req.reduction_dims or ()
        keepdim = bool(req.keepdim)
        if req.op_spec.name == "std":
            return _write_std_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_strides[0],
                req.output_shape,
                req.output_strides,
                reduction_dims,
                keepdim,
                req.dtype,
                unbiased=bool(req.params.get("unbiased", True)),
            )
        if req.op_spec.name == "var":
            return _write_var_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_strides[0],
                req.output_shape,
                req.output_strides,
                reduction_dims,
                keepdim,
                req.dtype,
                unbiased=bool(req.params.get("unbiased", True)),
            )
        if req.op_spec.name == "norm":
            return _write_norm_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_strides[0],
                req.output_shape,
                req.output_strides,
                reduction_dims,
                keepdim,
                req.dtype,
                p_value=float(req.params.get("p_value", 2.0)),
            )
        return _write_reduction_kernel(
            req.node_index,
            req.op_spec,
            req.input_shapes[0],
            req.input_strides[0],
            req.output_shape,
            req.output_strides,
            reduction_dims,
            keepdim,
            req.dtype,
        )
