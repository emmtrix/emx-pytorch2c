import ctypes
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


class RefBackendError(RuntimeError):
    pass


class RefDType:
    REF_F32 = 0


class RefOpKind:
    REF_OP_ADD = 0
    REF_OP_SUB = 1
    REF_OP_MUL = 2
    REF_OP_MATMUL = 3
    REF_OP_BMM = 4
    REF_OP_BROADCAST_IN_DIM = 5
    REF_OP_DIV = 6
    REF_OP_MAXIMUM = 7
    REF_OP_MINIMUM = 8
    REF_OP_NEG = 9
    REF_OP_EXP = 10
    REF_OP_ABS = 11
    REF_OP_SQRT = 12
    REF_OP_LOG = 13
    REF_OP_SIN = 14
    REF_OP_COS = 15
    REF_OP_TANH = 16
    REF_OP_FLOOR = 17
    REF_OP_CEIL = 18
    REF_OP_RECIPROCAL = 19
    REF_OP_RELU = 20
    REF_OP_ACOS = 21
    REF_OP_ACOSH = 22
    REF_OP_ASIN = 23
    REF_OP_ASINH = 24
    REF_OP_ATAN = 25
    REF_OP_ATANH = 26
    REF_OP_COSH = 27
    REF_OP_SINH = 28
    REF_OP_TAN = 29
    REF_OP_ERF = 30
    REF_OP_ERFC = 31
    REF_OP_EXPM1 = 32
    REF_OP_LOG1P = 33
    REF_OP_LOG2 = 34
    REF_OP_LOG10 = 35
    REF_OP_RSQRT = 36
    REF_OP_SIGMOID = 37
    REF_OP_SIGN = 38
    REF_OP_ROUND = 39
    REF_OP_TRUNC = 40
    REF_OP_CONV2D = 41
    REF_OP_ANGLE = 42
    REF_OP_CONJ = 43
    REF_OP_CONJ_PHYSICAL = 44
    REF_OP_DEG2RAD = 45
    REF_OP_DIGAMMA = 46
    REF_OP_ERFINV = 47
    REF_OP_EXP2 = 48
    REF_OP_FRAC = 49
    REF_OP_I0 = 50
    REF_OP_LGAMMA = 51
    REF_OP_LOGIT = 52
    REF_OP_NAN_TO_NUM = 53
    REF_OP_POSITIVE = 54
    REF_OP_RAD2DEG = 55
    REF_OP_REAL = 56
    REF_OP_SGN = 57
    REF_OP_SINC = 58
    REF_OP_SQUARE = 59
    REF_OP_ATAN2 = 60
    REF_OP_POW = 61
    REF_OP_REMAINDER = 62
    REF_OP_FMOD = 63
    REF_OP_FLOOR_DIVIDE = 64
    REF_OP_FMAX = 65
    REF_OP_FMIN = 66
    REF_OP_COPYSIGN = 67
    REF_OP_HYPOT = 68
    REF_OP_LOGADDEXP = 69
    REF_OP_NEXTAFTER = 70
    REF_OP_XLOGY = 71
    REF_OP_HEAVISIDE = 72
    REF_OP_LDEXP = 73
    REF_OP_CLAMP_MIN = 74
    REF_OP_CLAMP_MAX = 75
    REF_OP_SILU = 76
    REF_OP_CBRT = 77
    REF_OP_LSTM = 78
    REF_OP_AMAX = 79
    REF_OP_AMIN = 80


class RefTensorView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int32),
        ("sizes", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("dtype", ctypes.c_int32),
    ]


class RefOpCall(ctypes.Structure):
    _fields_ = [
        ("inputs", ctypes.POINTER(RefTensorView)),
        ("n_inputs", ctypes.c_int32),
        ("outputs", ctypes.POINTER(RefTensorView)),
        ("n_outputs", ctypes.c_int32),
        ("params", ctypes.c_void_p),
    ]


class RefBroadcastInDimParams(ctypes.Structure):
    _fields_ = [
        ("n_dims", ctypes.c_int32),
        ("broadcast_dimensions", ctypes.POINTER(ctypes.c_int32)),
    ]


class RefConv2dParams(ctypes.Structure):
    _fields_ = [
        ("stride_h", ctypes.c_int64),
        ("stride_w", ctypes.c_int64),
        ("padding_h", ctypes.c_int64),
        ("padding_w", ctypes.c_int64),
        ("dilation_h", ctypes.c_int64),
        ("dilation_w", ctypes.c_int64),
        ("groups", ctypes.c_int64),
    ]


class RefLstmParams(ctypes.Structure):
    _fields_ = [
        ("has_biases", ctypes.c_int32),
        ("num_layers", ctypes.c_int32),
        ("train", ctypes.c_int32),
        ("bidirectional", ctypes.c_int32),
        ("batch_first", ctypes.c_int32),
        ("dropout", ctypes.c_float),
    ]


@dataclass
class TensorViewBuffers:
    sizes: ctypes.Array
    strides: ctypes.Array


class RefBackendLibrary:
    def __init__(self) -> None:
        module = importlib.import_module("c_ref_backend._c_ref_backend")
        self.lib = ctypes.CDLL(module.__file__)
        self.lib.ref_run_op.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(RefOpCall),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.ref_run_op.restype = ctypes.c_int32

    def run_op(self, op_kind: int, call: RefOpCall) -> None:
        err_cap = 512
        err_buf = ctypes.create_string_buffer(err_cap)
        rc = self.lib.ref_run_op(op_kind, ctypes.byref(call), err_buf, err_cap)
        if rc != 0:
            msg = err_buf.value.decode("utf-8")
            raise RefBackendError(f"ref_run_op failed ({rc}): {msg}")


_lib_instance: Optional[RefBackendLibrary] = None


def _get_library() -> RefBackendLibrary:
    global _lib_instance
    if _lib_instance is None:
        _lib_instance = RefBackendLibrary()
    return _lib_instance


def _dtype_to_ref(dtype: torch.dtype) -> int:
    if dtype is torch.float32:
        return RefDType.REF_F32
    raise RefBackendError(f"Unsupported dtype: {dtype}")


def _tensor_to_view(tensor: torch.Tensor) -> Tuple[RefTensorView, TensorViewBuffers]:
    sizes = (ctypes.c_int64 * tensor.ndim)(*tensor.size())
    strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
    view = RefTensorView(
        data=ctypes.c_void_p(tensor.data_ptr()),
        ndim=ctypes.c_int32(tensor.ndim),
        sizes=sizes,
        strides=strides,
        dtype=_dtype_to_ref(tensor.dtype),
    )
    return view, TensorViewBuffers(sizes=sizes, strides=strides)


def _build_call(
    inputs: Tuple[torch.Tensor, ...],
    outputs: Tuple[torch.Tensor, ...],
    params: Optional[ctypes.c_void_p] = None,
) -> Tuple[RefOpCall, Tuple[object, ...]]:
    input_views = []
    output_views = []
    buffers = []
    for tensor in inputs:
        view, buf = _tensor_to_view(tensor)
        input_views.append(view)
        buffers.append(buf)
    for tensor in outputs:
        view, buf = _tensor_to_view(tensor)
        output_views.append(view)
        buffers.append(buf)
    input_array = (RefTensorView * len(inputs))(*input_views)
    output_array = (RefTensorView * len(outputs))(*output_views)
    call = RefOpCall(
        inputs=input_array,
        n_inputs=ctypes.c_int32(len(inputs)),
        outputs=output_array,
        n_outputs=ctypes.c_int32(len(outputs)),
        params=params,
    )
    buffers.extend([input_array, output_array, params])
    return call, tuple(buffers)


def _validate_float32(op_name: str, *tensors: torch.Tensor) -> None:
    if any(tensor.dtype is not torch.float32 for tensor in tensors):
        raise RefBackendError(f"{op_name} supports only torch.float32 tensors")


def _validate_max_dims(op_name: str, *tensors: torch.Tensor, max_dims: int = 8) -> None:
    if any(tensor.ndim > max_dims for tensor in tensors):
        raise RefBackendError(f"{op_name} supports at most {max_dims} dimensions")


def _validate_scalar_output(op_name: str, out: torch.Tensor) -> None:
    if out.ndim != 0:
        raise RefBackendError(f"{op_name} requires output to be a scalar tensor")


def _normalize_conv2d_param(name: str, value: object) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return (value[0], value[1])
    raise RefBackendError(f"conv2d expects {name} to be an int or a pair of ints")


def _conv2d_output_shape(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> Tuple[int, int, int, int]:
    batch, in_channels, in_h, in_w = input_tensor.shape
    out_channels, weight_in_channels, kernel_h, kernel_w = weight.shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "conv2d requires output channels to be divisible by groups"
        )
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    numerator_h = in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1
    numerator_w = in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1
    if numerator_h < 0 or numerator_w < 0:
        raise RefBackendError("conv2d requires output shape (N, C_out, H_out, W_out)")
    out_h = numerator_h // stride_h + 1
    out_w = numerator_w // stride_w + 1
    return batch, out_channels, out_h, out_w


def _run_binary_elementwise(
    op_name: str, op_kind: int, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    _validate_float32(op_name, a, b, out)
    if a.shape != b.shape or a.shape != out.shape:
        raise RefBackendError(
            f"{op_name} requires inputs and output to have identical shapes"
        )
    _validate_max_dims(op_name, a, b, out)
    call, buffers = _build_call((a, b), (out,))
    _ = buffers
    _get_library().run_op(op_kind, call)


def _run_unary_elementwise(
    op_name: str, op_kind: int, a: torch.Tensor, out: torch.Tensor
) -> None:
    _validate_float32(op_name, a, out)
    if a.shape != out.shape:
        raise RefBackendError(
            f"{op_name} requires input and output to have identical shapes"
        )
    _validate_max_dims(op_name, a, out)
    call, buffers = _build_call((a,), (out,))
    _ = buffers
    _get_library().run_op(op_kind, call)


def run_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("add", RefOpKind.REF_OP_ADD, a, b, out)


def run_sub(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("sub", RefOpKind.REF_OP_SUB, a, b, out)


def run_mul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("mul", RefOpKind.REF_OP_MUL, a, b, out)


def run_div(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("div", RefOpKind.REF_OP_DIV, a, b, out)


def run_maximum(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("maximum", RefOpKind.REF_OP_MAXIMUM, a, b, out)


def run_minimum(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("minimum", RefOpKind.REF_OP_MINIMUM, a, b, out)


def run_amax(a: torch.Tensor, out: torch.Tensor) -> None:
    _validate_float32("amax", a, out)
    _validate_max_dims("amax", a)
    if a.numel() == 0:
        raise RefBackendError(
            "amax requires input.numel() > 0 when dim is not specified"
        )
    _validate_scalar_output("amax", out)
    call, buffers = _build_call((a,), (out,))
    _ = buffers
    _get_library().run_op(RefOpKind.REF_OP_AMAX, call)


def run_amin(a: torch.Tensor, out: torch.Tensor) -> None:
    _validate_float32("amin", a, out)
    _validate_max_dims("amin", a)
    if a.numel() == 0:
        raise RefBackendError(
            "amin requires input.numel() > 0 when dim is not specified"
        )
    _validate_scalar_output("amin", out)
    call, buffers = _build_call((a,), (out,))
    _ = buffers
    _get_library().run_op(RefOpKind.REF_OP_AMIN, call)


def run_atan2(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("atan2", RefOpKind.REF_OP_ATAN2, a, b, out)


def run_pow(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("pow", RefOpKind.REF_OP_POW, a, b, out)


def run_remainder(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("remainder", RefOpKind.REF_OP_REMAINDER, a, b, out)


def run_fmod(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("fmod", RefOpKind.REF_OP_FMOD, a, b, out)


def run_floor_divide(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise(
        "floor_divide", RefOpKind.REF_OP_FLOOR_DIVIDE, a, b, out
    )


def run_fmax(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("fmax", RefOpKind.REF_OP_FMAX, a, b, out)


def run_fmin(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("fmin", RefOpKind.REF_OP_FMIN, a, b, out)


def run_copysign(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("copysign", RefOpKind.REF_OP_COPYSIGN, a, b, out)


def run_hypot(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("hypot", RefOpKind.REF_OP_HYPOT, a, b, out)


def run_logaddexp(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("logaddexp", RefOpKind.REF_OP_LOGADDEXP, a, b, out)


def run_nextafter(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("nextafter", RefOpKind.REF_OP_NEXTAFTER, a, b, out)


def run_xlogy(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("xlogy", RefOpKind.REF_OP_XLOGY, a, b, out)


def run_heaviside(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("heaviside", RefOpKind.REF_OP_HEAVISIDE, a, b, out)


def run_ldexp(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("ldexp", RefOpKind.REF_OP_LDEXP, a, b, out)


def run_clamp_min(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("clamp_min", RefOpKind.REF_OP_CLAMP_MIN, a, b, out)


def run_clamp_max(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("clamp_max", RefOpKind.REF_OP_CLAMP_MAX, a, b, out)


def run_neg(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("neg", RefOpKind.REF_OP_NEG, a, out)


def run_exp(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("exp", RefOpKind.REF_OP_EXP, a, out)


def run_abs(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("abs", RefOpKind.REF_OP_ABS, a, out)


def run_sqrt(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sqrt", RefOpKind.REF_OP_SQRT, a, out)


def run_cbrt(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("cbrt", RefOpKind.REF_OP_CBRT, a, out)


def run_log(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("log", RefOpKind.REF_OP_LOG, a, out)


def run_sin(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sin", RefOpKind.REF_OP_SIN, a, out)


def run_cos(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("cos", RefOpKind.REF_OP_COS, a, out)


def run_acos(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("acos", RefOpKind.REF_OP_ACOS, a, out)


def run_acosh(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("acosh", RefOpKind.REF_OP_ACOSH, a, out)


def run_asin(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("asin", RefOpKind.REF_OP_ASIN, a, out)


def run_asinh(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("asinh", RefOpKind.REF_OP_ASINH, a, out)


def run_atan(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("atan", RefOpKind.REF_OP_ATAN, a, out)


def run_atanh(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("atanh", RefOpKind.REF_OP_ATANH, a, out)


def run_cosh(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("cosh", RefOpKind.REF_OP_COSH, a, out)


def run_sinh(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sinh", RefOpKind.REF_OP_SINH, a, out)


def run_tan(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("tan", RefOpKind.REF_OP_TAN, a, out)


def run_erf(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("erf", RefOpKind.REF_OP_ERF, a, out)


def run_erfc(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("erfc", RefOpKind.REF_OP_ERFC, a, out)


def run_expm1(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("expm1", RefOpKind.REF_OP_EXPM1, a, out)


def run_log1p(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("log1p", RefOpKind.REF_OP_LOG1P, a, out)


def run_log2(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("log2", RefOpKind.REF_OP_LOG2, a, out)


def run_log10(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("log10", RefOpKind.REF_OP_LOG10, a, out)


def run_rsqrt(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("rsqrt", RefOpKind.REF_OP_RSQRT, a, out)


def run_sigmoid(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sigmoid", RefOpKind.REF_OP_SIGMOID, a, out)


def run_silu(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("silu", RefOpKind.REF_OP_SILU, a, out)


def run_sign(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sign", RefOpKind.REF_OP_SIGN, a, out)


def run_round(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("round", RefOpKind.REF_OP_ROUND, a, out)


def run_trunc(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("trunc", RefOpKind.REF_OP_TRUNC, a, out)


def run_tanh(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("tanh", RefOpKind.REF_OP_TANH, a, out)


def run_floor(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("floor", RefOpKind.REF_OP_FLOOR, a, out)


def run_ceil(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("ceil", RefOpKind.REF_OP_CEIL, a, out)


def run_reciprocal(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("reciprocal", RefOpKind.REF_OP_RECIPROCAL, a, out)


def run_relu(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("relu", RefOpKind.REF_OP_RELU, a, out)

def run_angle(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("angle", RefOpKind.REF_OP_ANGLE, a, out)


def run_conj(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("conj", RefOpKind.REF_OP_CONJ, a, out)


def run_conj_physical(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise(
        "conj_physical", RefOpKind.REF_OP_CONJ_PHYSICAL, a, out
    )


def run_deg2rad(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("deg2rad", RefOpKind.REF_OP_DEG2RAD, a, out)


def run_digamma(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("digamma", RefOpKind.REF_OP_DIGAMMA, a, out)


def run_erfinv(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("erfinv", RefOpKind.REF_OP_ERFINV, a, out)


def run_exp2(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("exp2", RefOpKind.REF_OP_EXP2, a, out)


def run_frac(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("frac", RefOpKind.REF_OP_FRAC, a, out)


def run_i0(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("i0", RefOpKind.REF_OP_I0, a, out)


def run_lgamma(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("lgamma", RefOpKind.REF_OP_LGAMMA, a, out)


def run_logit(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("logit", RefOpKind.REF_OP_LOGIT, a, out)


def run_nan_to_num(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("nan_to_num", RefOpKind.REF_OP_NAN_TO_NUM, a, out)


def run_positive(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("positive", RefOpKind.REF_OP_POSITIVE, a, out)


def run_rad2deg(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("rad2deg", RefOpKind.REF_OP_RAD2DEG, a, out)


def run_real(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("real", RefOpKind.REF_OP_REAL, a, out)


def run_sgn(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sgn", RefOpKind.REF_OP_SGN, a, out)


def run_sinc(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sinc", RefOpKind.REF_OP_SINC, a, out)


def run_square(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("square", RefOpKind.REF_OP_SQUARE, a, out)


def run_conv2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    stride: object = 1,
    padding: object = 0,
    dilation: object = 1,
    groups: int = 1,
) -> None:
    if (
        input_tensor.dtype is not torch.float32
        or weight.dtype is not torch.float32
        or out.dtype is not torch.float32
    ):
        raise RefBackendError("conv2d supports only torch.float32 tensors")
    if input_tensor.ndim != 4 or weight.ndim != 4 or out.ndim != 4:
        raise RefBackendError("conv2d requires 4D input, weight, and output")
    if (
        not input_tensor.is_contiguous()
        or not weight.is_contiguous()
        or not out.is_contiguous()
    ):
        raise RefBackendError("conv2d requires contiguous tensors")
    stride_pair = _normalize_conv2d_param("stride", stride)
    padding_pair = _normalize_conv2d_param("padding", padding)
    dilation_pair = _normalize_conv2d_param("dilation", dilation)
    if (
        stride_pair[0] <= 0
        or stride_pair[1] <= 0
        or dilation_pair[0] <= 0
        or dilation_pair[1] <= 0
        or padding_pair[0] < 0
        or padding_pair[1] < 0
    ):
        raise RefBackendError(
            "conv2d expects stride and dilation to be positive and padding to be non-negative"
        )
    if groups <= 0:
        raise RefBackendError("conv2d requires positive groups")
    expected_shape = _conv2d_output_shape(
        input_tensor, weight, stride_pair, padding_pair, dilation_pair, groups
    )
    if out.shape != expected_shape:
        raise RefBackendError("conv2d requires output shape (N, C_out, H_out, W_out)")
    params = RefConv2dParams(
        stride_h=stride_pair[0],
        stride_w=stride_pair[1],
        padding_h=padding_pair[0],
        padding_w=padding_pair[1],
        dilation_h=dilation_pair[0],
        dilation_w=dilation_pair[1],
        groups=groups,
    )
    call, buffers = _build_call(
        (input_tensor, weight),
        (out,),
        params=ctypes.cast(ctypes.pointer(params), ctypes.c_void_p),
    )
    _ = (buffers, params)
    _get_library().run_op(RefOpKind.REF_OP_CONV2D, call)


def run_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if (
        a.dtype is not torch.float32
        or b.dtype is not torch.float32
        or out.dtype is not torch.float32
    ):
        raise RefBackendError("matmul supports only torch.float32 tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise RefBackendError("matmul requires 2D inputs and output")
    if a.shape[1] != b.shape[0]:
        raise RefBackendError("matmul requires inner dimensions to match")
    if out.shape != (a.shape[0], b.shape[1]):
        raise RefBackendError("matmul requires output shape (m, n)")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("matmul requires contiguous tensors")
    call, buffers = _build_call((a, b), (out,))
    _ = buffers
    _get_library().run_op(RefOpKind.REF_OP_MATMUL, call)


def run_bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if (
        a.dtype is not torch.float32
        or b.dtype is not torch.float32
        or out.dtype is not torch.float32
    ):
        raise RefBackendError("bmm supports only torch.float32 tensors")
    if a.ndim != 3 or b.ndim != 3 or out.ndim != 3:
        raise RefBackendError("bmm requires 3D inputs and output")
    if a.shape[0] != b.shape[0]:
        raise RefBackendError("bmm requires batch dimensions to match")
    if a.shape[2] != b.shape[1]:
        raise RefBackendError("bmm requires inner dimensions to match")
    if out.shape != (a.shape[0], a.shape[1], b.shape[2]):
        raise RefBackendError("bmm requires output shape (batch, m, n)")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("bmm requires contiguous tensors")
    call, buffers = _build_call((a, b), (out,))
    _ = buffers
    _get_library().run_op(RefOpKind.REF_OP_BMM, call)


def run_broadcast_in_dim(
    a: torch.Tensor, out: torch.Tensor, broadcast_dimensions: Tuple[int, ...]
) -> None:
    if a.dtype is not torch.float32 or out.dtype is not torch.float32:
        raise RefBackendError("broadcast_in_dim supports only torch.float32 tensors")
    if not a.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("broadcast_in_dim requires contiguous tensors")
    if len(broadcast_dimensions) != a.ndim:
        raise RefBackendError(
            "broadcast_in_dim expects broadcast_dimensions to match input rank"
        )
    if out.ndim < a.ndim:
        raise RefBackendError("broadcast_in_dim requires output rank >= input rank")

    dims = (ctypes.c_int32 * len(broadcast_dimensions))(*broadcast_dimensions)
    params = RefBroadcastInDimParams(
        n_dims=ctypes.c_int32(len(broadcast_dimensions)),
        broadcast_dimensions=dims,
    )

    call, buffers = _build_call(
        (a,),
        (out,),
        params=ctypes.cast(ctypes.pointer(params), ctypes.c_void_p),
    )
    _ = (buffers, dims, params)
    _get_library().run_op(RefOpKind.REF_OP_BROADCAST_IN_DIM, call)


def run_lstm(
    input_tensor: torch.Tensor,
    hx: Tuple[torch.Tensor, torch.Tensor],
    params: Tuple[torch.Tensor, ...],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
    out: torch.Tensor,
    h_n: torch.Tensor,
    c_n: torch.Tensor,
) -> None:
    if (
        input_tensor.dtype is not torch.float32
        or h_n.dtype is not torch.float32
        or c_n.dtype is not torch.float32
        or out.dtype is not torch.float32
    ):
        raise RefBackendError("lstm supports only torch.float32 tensors")
    if len(hx) != 2:
        raise RefBackendError("lstm expects hx to be a tuple of (h0, c0)")
    if len(params) != 4:
        raise RefBackendError("lstm expects params to be (weight_ih, weight_hh, bias_ih, bias_hh)")
    h0, c0 = hx
    weight_ih, weight_hh, bias_ih, bias_hh = params
    if (
        h0.dtype is not torch.float32
        or c0.dtype is not torch.float32
        or weight_ih.dtype is not torch.float32
        or weight_hh.dtype is not torch.float32
        or bias_ih.dtype is not torch.float32
        or bias_hh.dtype is not torch.float32
    ):
        raise RefBackendError("lstm supports only torch.float32 tensors")
    if input_tensor.ndim != 3:
        raise RefBackendError("lstm requires input to be 3D")
    if h0.ndim != 3 or c0.ndim != 3:
        raise RefBackendError("lstm requires h0 and c0 to be 3D")
    if weight_ih.ndim != 2 or weight_hh.ndim != 2:
        raise RefBackendError("lstm requires weight_ih and weight_hh to be 2D")
    if bias_ih.ndim != 1 or bias_hh.ndim != 1:
        raise RefBackendError("lstm requires bias_ih and bias_hh to be 1D")
    if not has_biases:
        raise RefBackendError("lstm supports only has_biases=True")
    if num_layers != 1:
        raise RefBackendError("lstm supports only num_layers=1")
    if dropout != 0.0:
        raise RefBackendError("lstm supports only dropout=0")
    if train:
        raise RefBackendError("lstm supports only train=False")
    if bidirectional:
        raise RefBackendError("lstm supports only bidirectional=False")

    if batch_first:
        batch = input_tensor.shape[0]
        seq_len = input_tensor.shape[1]
    else:
        seq_len = input_tensor.shape[0]
        batch = input_tensor.shape[1]
    input_size = input_tensor.shape[2]
    hidden_size = weight_hh.shape[1]
    gate_size = 4 * hidden_size

    if weight_ih.shape[0] != gate_size or weight_hh.shape[0] != gate_size:
        raise RefBackendError(
            "lstm requires weight_ih and weight_hh to have 4 * hidden_size rows"
        )
    if weight_ih.shape[1] != input_size:
        raise RefBackendError("lstm requires input_size to match weight_ih")
    if weight_hh.shape[1] != hidden_size:
        raise RefBackendError("lstm requires hidden_size to match weight_hh")
    if bias_ih.numel() != gate_size or bias_hh.numel() != gate_size:
        raise RefBackendError(
            "lstm requires bias_ih and bias_hh to have 4 * hidden_size elements"
        )
    if h0.shape != (1, batch, hidden_size) or c0.shape != (1, batch, hidden_size):
        raise RefBackendError("lstm requires h0 and c0 shape (1, batch, hidden_size)")
    if batch_first:
        expected_out = (batch, seq_len, hidden_size)
        if out.shape != expected_out:
            raise RefBackendError(
                "lstm requires output shape (batch, seq_len, hidden_size)"
            )
    else:
        expected_out = (seq_len, batch, hidden_size)
        if out.shape != expected_out:
            raise RefBackendError(
                "lstm requires output shape (seq_len, batch, hidden_size)"
            )
    if h_n.shape != (1, batch, hidden_size) or c_n.shape != (1, batch, hidden_size):
        raise RefBackendError("lstm requires h_n and c_n shape (1, batch, hidden_size)")
    if (
        not input_tensor.is_contiguous()
        or not h0.is_contiguous()
        or not c0.is_contiguous()
        or not weight_ih.is_contiguous()
        or not weight_hh.is_contiguous()
        or not bias_ih.is_contiguous()
        or not bias_hh.is_contiguous()
        or not out.is_contiguous()
        or not h_n.is_contiguous()
        or not c_n.is_contiguous()
    ):
        raise RefBackendError("lstm requires contiguous tensors")

    params_struct = RefLstmParams(
        has_biases=1,
        num_layers=num_layers,
        train=1 if train else 0,
        bidirectional=1 if bidirectional else 0,
        batch_first=1 if batch_first else 0,
        dropout=dropout,
    )
    call, buffers = _build_call(
        (input_tensor, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh),
        (out, h_n, c_n),
        params=ctypes.cast(ctypes.pointer(params_struct), ctypes.c_void_p),
    )
    _ = (buffers, params_struct)
    _get_library().run_op(RefOpKind.REF_OP_LSTM, call)
