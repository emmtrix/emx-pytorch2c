from __future__ import annotations

import math
import numbers
import operator
from typing import Sequence, Tuple

import torch
import torch.fx

from codegen_backend.analysis_helpers import (
    error_kwarg_specified_once,
    normalize_reduction_dims,
    parse_constant_int,
)
from codegen_backend.errors import CodegenBackendError


class ReductionsArgParser:
    def normalize_reduction_dims(
        self, op_name: str, dim: object | None, rank: int
    ) -> Tuple[int, ...]:
        return normalize_reduction_dims(op_name, dim, rank)

    def parse_reduction_args(
        self, op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
    ) -> Tuple[Tuple[int, ...], bool, bool, bool | None]:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_name} expects one input")
        if len(node.args) > 4:
            raise CodegenBackendError(f"codegen {op_name} expects at most four inputs")
        if op_name == "std":
            unbiased = True
            if len(node.args) > 2:
                raise CodegenBackendError(
                    "codegen std expects at most two inputs (self, unbiased)"
                )
            if len(node.args) > 1:
                unbiased = node.args[1]
            if node.kwargs:
                if "unbiased" in node.kwargs:
                    if len(node.args) > 1:
                        raise error_kwarg_specified_once(op_name, "unbiased")
                    unbiased = node.kwargs["unbiased"]
                extra = set(node.kwargs) - {"unbiased"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen std got unexpected kwargs: {sorted(extra)}"
                    )
            if isinstance(unbiased, torch.fx.Node):
                raise CodegenBackendError("codegen std expects unbiased to be a bool")
            if not isinstance(unbiased, bool):
                raise CodegenBackendError("codegen std expects unbiased to be a bool")
            reduction_dims = tuple(range(len(input_shape)))
            keepdim = False
            reduce_all = True
            return reduction_dims, keepdim, reduce_all, unbiased
        if op_name == "var":
            if len(node.args) > 4:
                raise CodegenBackendError(
                    "codegen var expects at most four inputs (self, dim, unbiased, keepdim)"
                )
            dim = node.args[1] if len(node.args) > 1 else None
            unbiased = node.args[2] if len(node.args) > 2 else True
            keepdim = node.args[3] if len(node.args) > 3 else False
            correction = None
            if node.kwargs:
                if "dim" in node.kwargs:
                    if dim is not None:
                        raise error_kwarg_specified_once(op_name, "dim")
                    dim = node.kwargs["dim"]
                if "unbiased" in node.kwargs:
                    if len(node.args) > 2:
                        raise error_kwarg_specified_once(op_name, "unbiased")
                    unbiased = node.kwargs["unbiased"]
                if "keepdim" in node.kwargs:
                    if len(node.args) > 3:
                        raise error_kwarg_specified_once(op_name, "keepdim")
                    keepdim = node.kwargs["keepdim"]
                if "correction" in node.kwargs:
                    correction = node.kwargs["correction"]
                extra = set(node.kwargs) - {"dim", "unbiased", "keepdim", "correction"}
                if extra:
                    raise CodegenBackendError(
                        f"codegen var got unexpected kwargs: {sorted(extra)}"
                    )
            if isinstance(unbiased, torch.fx.Node):
                raise CodegenBackendError("codegen var expects unbiased to be a bool")
            if not isinstance(unbiased, bool):
                raise CodegenBackendError("codegen var expects unbiased to be a bool")
            if isinstance(keepdim, torch.fx.Node):
                raise CodegenBackendError("codegen var expects keepdim to be a bool")
            if not isinstance(keepdim, bool):
                raise CodegenBackendError("codegen var expects keepdim to be a bool")
            if correction is not None:
                if isinstance(correction, torch.fx.Node):
                    raise CodegenBackendError(
                        "codegen var expects correction to be 0 or 1"
                    )
                if not isinstance(correction, numbers.Number):
                    raise CodegenBackendError(
                        "codegen var expects correction to be 0 or 1"
                    )
                correction_value = float(correction)
                if correction_value not in (0.0, 1.0):
                    raise CodegenBackendError(
                        "codegen var expects correction to be 0 or 1"
                    )
                unbiased = bool(int(correction_value))
            reduction_dims = self.normalize_reduction_dims(
                op_name, dim, len(input_shape)
            )
            reduce_all = dim is None
            return reduction_dims, keepdim, reduce_all, unbiased
        dim = node.args[1] if len(node.args) > 1 else None
        keepdim = node.args[2] if len(node.args) > 2 else False
        dtype = node.args[3] if len(node.args) > 3 else None
        if node.kwargs:
            if "dim" in node.kwargs:
                if dim is not None:
                    raise error_kwarg_specified_once(op_name, "dim")
                dim = node.kwargs["dim"]
            if "keepdim" in node.kwargs:
                if len(node.args) > 2:
                    raise error_kwarg_specified_once(op_name, "keepdim")
                keepdim = node.kwargs["keepdim"]
            if "dtype" in node.kwargs:
                if dtype is not None:
                    raise error_kwarg_specified_once(op_name, "dtype")
                dtype = node.kwargs["dtype"]
            extra = set(node.kwargs) - {"dim", "keepdim", "dtype"}
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
                )
        if isinstance(keepdim, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects keepdim to be a bool"
            )
        if not isinstance(keepdim, bool):
            raise CodegenBackendError(
                f"codegen {op_name} expects keepdim to be a bool"
            )
        if dtype is not None:
            if isinstance(dtype, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_name} expects dtype to be torch.float32, torch.float64, torch.int8, torch.uint8, torch.uint32, torch.int32, or torch.bool"
                )
            if dtype not in (
                torch.float32,
                torch.float64,
                torch.int8,
                torch.uint8,
                torch.uint32,
                torch.int32,
                torch.bool,
            ):
                raise CodegenBackendError(
                    f"codegen {op_name} expects dtype to be torch.float32, torch.float64, torch.int8, torch.uint8, torch.uint32, torch.int32, or torch.bool"
                )
        reduction_dims = self.normalize_reduction_dims(op_name, dim, len(input_shape))
        reduce_all = dim is None
        return reduction_dims, keepdim, reduce_all, None

    def parse_norm_args(
        self, op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
    ) -> Tuple[Tuple[int, ...], bool, bool, float]:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_name} expects one input")
        if len(node.args) > 4:
            raise CodegenBackendError(
                f"codegen {op_name} expects at most four inputs (self, p, dim, keepdim)"
            )
        p = node.args[1] if len(node.args) > 1 else None
        dim = node.args[2] if len(node.args) > 2 else None
        keepdim = node.args[3] if len(node.args) > 3 else False
        if node.kwargs:
            if "p" in node.kwargs:
                if len(node.args) > 1:
                    raise error_kwarg_specified_once(op_name, "p")
                p = node.kwargs["p"]
            if "dim" in node.kwargs:
                if dim is not None:
                    raise error_kwarg_specified_once(op_name, "dim")
                dim = node.kwargs["dim"]
            if "keepdim" in node.kwargs:
                if len(node.args) > 3:
                    raise error_kwarg_specified_once(op_name, "keepdim")
                keepdim = node.kwargs["keepdim"]
            extra = set(node.kwargs) - {"p", "dim", "keepdim"}
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
                )
        if isinstance(keepdim, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects keepdim to be a bool"
            )
        if not isinstance(keepdim, bool):
            raise CodegenBackendError(
                f"codegen {op_name} expects keepdim to be a bool"
            )
        if isinstance(p, torch.fx.Node):
            raise CodegenBackendError(f"codegen {op_name} expects p to be a number")
        if p is None:
            p_value = 2.0
        elif isinstance(p, bool):
            p_value = float(p)
        elif isinstance(p, (int, float)):
            p_value = float(p)
        else:
            raise CodegenBackendError(f"codegen {op_name} expects p to be a number")
        if math.isinf(p_value) or math.isnan(p_value):
            raise CodegenBackendError(f"codegen {op_name} expects p to be finite")
        reduction_dims = self.normalize_reduction_dims(op_name, dim, len(input_shape))
        reduce_all = dim is None
        return reduction_dims, keepdim, reduce_all, p_value

    def parse_argminmax_args(
        self, op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
    ) -> Tuple[Tuple[int, ...], bool, bool]:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_name} expects one input")
        if len(node.args) > 3:
            raise CodegenBackendError(f"codegen {op_name} expects at most three inputs")
        dim = node.args[1] if len(node.args) > 1 else None
        keepdim = node.args[2] if len(node.args) > 2 else False
        if node.kwargs:
            if "dim" in node.kwargs:
                if dim is not None:
                    raise error_kwarg_specified_once(op_name, "dim")
                dim = node.kwargs["dim"]
            if "keepdim" in node.kwargs:
                if len(node.args) > 2:
                    raise error_kwarg_specified_once(op_name, "keepdim")
                keepdim = node.kwargs["keepdim"]
            extra = set(node.kwargs) - {"dim", "keepdim"}
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
                )
        if isinstance(keepdim, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_name} expects keepdim to be a bool"
            )
        if not isinstance(keepdim, bool):
            raise CodegenBackendError(
                f"codegen {op_name} expects keepdim to be a bool"
            )
        if dim is None:
            reduction_dims = tuple(range(len(input_shape)))
            reduce_all = True
            return reduction_dims, keepdim, reduce_all
        if isinstance(dim, (tuple, list)):
            raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
        dim_value = parse_constant_int(op_name, "dim", dim)
        rank = len(input_shape)
        if rank == 0:
            if dim_value not in (-1, 0):
                raise CodegenBackendError(f"codegen {op_name} dim is out of range")
            return (), keepdim, True
        if dim_value < 0:
            dim_value += rank
        if dim_value < 0 or dim_value >= rank:
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        return (dim_value,), keepdim, False

    def parse_softmax_args(
        self, op_name: str, node: torch.fx.Node, input_shape: Sequence[int]
    ) -> Tuple[int, object | None]:
        if not node.args:
            raise CodegenBackendError(f"codegen {op_name} expects one input")
        is_internal = op_name in {"_softmax", "_log_softmax"}
        if len(node.args) > 3:
            raise CodegenBackendError(f"codegen {op_name} expects at most three inputs")
        dim = node.args[1] if len(node.args) > 1 else None
        dtype = None
        half_to_float = None
        if len(node.args) > 2:
            if is_internal:
                half_to_float = node.args[2]
            else:
                dtype = node.args[2]
        if node.kwargs:
            if "dim" in node.kwargs:
                if dim is not None:
                    raise error_kwarg_specified_once(op_name, "dim")
                dim = node.kwargs["dim"]
            if is_internal:
                if "half_to_float" in node.kwargs:
                    if half_to_float is not None:
                        raise error_kwarg_specified_once(op_name, "half_to_float")
                    half_to_float = node.kwargs["half_to_float"]
                extra = set(node.kwargs) - {"dim", "half_to_float"}
            else:
                if "dtype" in node.kwargs:
                    if dtype is not None:
                        raise error_kwarg_specified_once(op_name, "dtype")
                    dtype = node.kwargs["dtype"]
                extra = set(node.kwargs) - {"dim", "dtype"}
            if extra:
                raise CodegenBackendError(
                    f"codegen {op_name} got unexpected kwargs: {sorted(extra)}"
                )
        if dim is None:
            raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
        if isinstance(dim, torch.fx.Node):
            raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
        if isinstance(dim, (tuple, list)):
            raise CodegenBackendError(f"codegen {op_name} expects dim to be an int")
        try:
            dim_value = operator.index(dim)
        except TypeError as exc:
            raise CodegenBackendError(
                f"codegen {op_name} expects dim to be an int"
            ) from exc
        rank = len(input_shape)
        if dim_value < 0:
            dim_value += rank
        if dim_value < 0 or dim_value >= rank:
            raise CodegenBackendError(f"codegen {op_name} dim is out of range")
        if is_internal:
            if half_to_float is None:
                half_to_float = False
            if isinstance(half_to_float, torch.fx.Node):
                raise CodegenBackendError(
                    f"codegen {op_name} expects half_to_float to be a bool"
                )
            if half_to_float not in (False, 0):
                raise CodegenBackendError(
                    f"codegen {op_name} expects half_to_float to be False"
                )
        else:
            if dtype is not None:
                if isinstance(dtype, torch.fx.Node):
                    raise CodegenBackendError(
                        f"codegen {op_name} expects dtype to be torch.float32, torch.float64, or None"
                    )
                if dtype not in (torch.float32, torch.float64):
                    raise CodegenBackendError(
                        f"codegen {op_name} expects dtype to be torch.float32, torch.float64, or None"
                    )
        return dim_value, dtype


__all__ = ["ReductionsArgParser"]
