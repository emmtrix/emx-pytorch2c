from __future__ import annotations

import torch
import torch.fx

from codegen_backend.analysis_helpers import error_kwarg_specified_once
from codegen_backend.errors import CodegenBackendError
from codegen_backend.parsing.common import (
    parse_constant_bool,
    parse_constant_int,
)


def parse_embedding_args(
    node: torch.fx.Node,
) -> tuple[object, object, int, bool, bool]:
    op_name = "embedding"
    if len(node.args) < 2:
        raise CodegenBackendError(f"codegen {op_name} expects weight and indices")
    if len(node.args) > 5:
        raise CodegenBackendError(
            f"codegen {op_name} expects at most five arguments"
        )
    weight = node.args[0]
    indices = node.args[1]
    padding_idx = node.args[2] if len(node.args) > 2 else -1
    scale_grad_by_freq = node.args[3] if len(node.args) > 3 else False
    sparse = node.args[4] if len(node.args) > 4 else False
    if node.kwargs:
        if "padding_idx" in node.kwargs:
            if len(node.args) > 2:
                raise error_kwarg_specified_once(op_name, "padding_idx")
            padding_idx = node.kwargs["padding_idx"]
        if "scale_grad_by_freq" in node.kwargs:
            if len(node.args) > 3:
                raise error_kwarg_specified_once(op_name, "scale_grad_by_freq")
            scale_grad_by_freq = node.kwargs["scale_grad_by_freq"]
        if "sparse" in node.kwargs:
            if len(node.args) > 4:
                raise error_kwarg_specified_once(op_name, "sparse")
            sparse = node.kwargs["sparse"]
        extra = set(node.kwargs) - {
            "padding_idx",
            "scale_grad_by_freq",
            "sparse",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    padding_idx_value = parse_constant_int(op_name, "padding_idx", padding_idx)
    scale_grad_value = parse_constant_bool(
        op_name, "scale_grad_by_freq", scale_grad_by_freq
    )
    sparse_value = parse_constant_bool(op_name, "sparse", sparse)
    return weight, indices, padding_idx_value, scale_grad_value, sparse_value


def parse_embedding_bag_args(
    node: torch.fx.Node,
) -> tuple[object, object, object, bool, int, bool, object, bool, int]:
    op_name = "_embedding_bag"
    if len(node.args) < 3:
        raise CodegenBackendError(
            f"codegen {op_name} expects weight, indices, and offsets"
        )
    if len(node.args) > 9:
        raise CodegenBackendError(
            f"codegen {op_name} expects at most nine arguments"
        )
    weight = node.args[0]
    indices = node.args[1]
    offsets = node.args[2]
    scale_grad_by_freq = node.args[3] if len(node.args) > 3 else False
    mode = node.args[4] if len(node.args) > 4 else 0
    sparse = node.args[5] if len(node.args) > 5 else False
    per_sample_weights = node.args[6] if len(node.args) > 6 else None
    include_last_offset = node.args[7] if len(node.args) > 7 else False
    padding_idx = node.args[8] if len(node.args) > 8 else -1
    if node.kwargs:
        if "scale_grad_by_freq" in node.kwargs:
            if len(node.args) > 3:
                raise error_kwarg_specified_once(op_name, "scale_grad_by_freq")
            scale_grad_by_freq = node.kwargs["scale_grad_by_freq"]
        if "mode" in node.kwargs:
            if len(node.args) > 4:
                raise error_kwarg_specified_once(op_name, "mode")
            mode = node.kwargs["mode"]
        if "sparse" in node.kwargs:
            if len(node.args) > 5:
                raise error_kwarg_specified_once(op_name, "sparse")
            sparse = node.kwargs["sparse"]
        if "per_sample_weights" in node.kwargs:
            if len(node.args) > 6:
                raise error_kwarg_specified_once(op_name, "per_sample_weights")
            per_sample_weights = node.kwargs["per_sample_weights"]
        if "include_last_offset" in node.kwargs:
            if len(node.args) > 7:
                raise error_kwarg_specified_once(op_name, "include_last_offset")
            include_last_offset = node.kwargs["include_last_offset"]
        if "padding_idx" in node.kwargs:
            if len(node.args) > 8:
                raise error_kwarg_specified_once(op_name, "padding_idx")
            padding_idx = node.kwargs["padding_idx"]
        extra = set(node.kwargs) - {
            "scale_grad_by_freq",
            "mode",
            "sparse",
            "per_sample_weights",
            "include_last_offset",
            "padding_idx",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
            )
    scale_grad_value = parse_constant_bool(
        op_name, "scale_grad_by_freq", scale_grad_by_freq
    )
    mode_value = parse_constant_int(op_name, "mode", mode)
    sparse_value = parse_constant_bool(op_name, "sparse", sparse)
    include_last_offset_value = parse_constant_bool(
        op_name, "include_last_offset", include_last_offset
    )
    padding_idx_value = parse_constant_int(op_name, "padding_idx", padding_idx)
    return (
        weight,
        indices,
        offsets,
        scale_grad_value,
        mode_value,
        sparse_value,
        per_sample_weights,
        include_last_offset_value,
        padding_idx_value,
    )


__all__ = ["parse_embedding_args", "parse_embedding_bag_args"]
