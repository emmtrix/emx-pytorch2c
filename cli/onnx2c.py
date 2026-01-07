import argparse
import importlib
from pathlib import Path
from typing import Iterable, List

import torch

from codegen_backend import codegen_generic_backend
from codegen_backend.export import export_generic_c


class _InlineModuleTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return False


def _require_module(module_name: str, message: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _default_output_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(".c")


def _onnx_dtype_to_torch(onnx_module, elem_type: int) -> torch.dtype:
    mapping = {
        onnx_module.TensorProto.FLOAT: torch.float32,
        onnx_module.TensorProto.FLOAT16: torch.float16,
        onnx_module.TensorProto.DOUBLE: torch.float64,
        onnx_module.TensorProto.BFLOAT16: torch.bfloat16,
        onnx_module.TensorProto.INT8: torch.int8,
        onnx_module.TensorProto.UINT8: torch.uint8,
        onnx_module.TensorProto.INT16: torch.int16,
        onnx_module.TensorProto.INT32: torch.int32,
        onnx_module.TensorProto.INT64: torch.int64,
        onnx_module.TensorProto.BOOL: torch.bool,
    }
    if elem_type not in mapping:
        raise ValueError(f"Unsupported ONNX tensor type: {elem_type}")
    return mapping[elem_type]


def _shape_from_type(onnx_module, value_info, default_dim: int) -> List[int]:
    if not value_info.type.HasField("tensor_type"):
        raise ValueError(f"Unsupported ONNX input kind: {value_info.type}")
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: List[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            dims.append(default_dim)
    return dims


def _collect_example_inputs(onnx_module, model, default_dim: int) -> List[torch.Tensor]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs: List[torch.Tensor] = []
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        dtype = _onnx_dtype_to_torch(onnx_module, tensor_type.elem_type)
        shape = _shape_from_type(onnx_module, value_info, default_dim)
        inputs.append(torch.zeros(tuple(shape), dtype=dtype))
    return inputs


def _trace_module(torch_module: torch.nn.Module) -> torch.fx.GraphModule:
    tracer = _InlineModuleTracer()
    graph = tracer.trace(torch_module)
    return torch.fx.GraphModule(torch_module, graph)


def _random_input_like(example: torch.Tensor) -> torch.Tensor:
    shape = tuple(example.shape)
    dtype = example.dtype
    if dtype.is_floating_point or dtype is torch.bfloat16:
        return torch.randn(shape, dtype=dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int8).to(dtype)
    if dtype is torch.uint8:
        return torch.randint(0, 8, shape, dtype=dtype)
    return torch.randint(-8, 8, shape, dtype=dtype)


def _random_inputs(example_inputs: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    return [_random_input_like(example) for example in example_inputs]


def _run_self_test(
    torch_module: torch.nn.Module,
    example_inputs: Iterable[torch.Tensor],
    runs: int,
) -> None:
    if runs < 0:
        raise ValueError("self-test runs must be >= 0")
    if runs == 0:
        return
    print(f"[onnx2c] running torch.compile self-test ({runs} random inputs)...")
    compiled = torch.compile(torch_module, backend=codegen_generic_backend)
    with torch.no_grad():
        for run_index in range(1, runs + 1):
            random_inputs = _random_inputs(example_inputs)
            compiled_output = compiled(*random_inputs)
            eager_output = torch_module(*random_inputs)
            torch.testing.assert_close(compiled_output, eager_output)
            print(f"[onnx2c] self-test run {run_index}/{runs} OK")
    print("[onnx2c] self-test completed.")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model to C code using the codegen backend."
    )
    parser.add_argument("onnx_path", type=Path, help="Path to the ONNX model file.")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path for generated C code (defaults to <model>.c).",
    )
    parser.add_argument(
        "--default-dim",
        type=int,
        default=1,
        help="Default dimension to use when ONNX inputs specify dynamic shapes.",
    )
    parser.add_argument(
        "--function-name",
        default="entry",
        help="Name for the generated C entry function.",
    )
    parser.add_argument(
        "--self-test-runs",
        type=int,
        default=1,
        help=(
            "Number of random inputs to validate torch.compile output against eager. "
            "Set to 0 to disable the self-test."
        ),
    )
    parser.add_argument(
        "--truncate-weights-after",
        type=int,
        default=None,
        help=(
            "Truncate inline weight initializers after this many values (emit '...'). "
            "Intended for golden test fixtures."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    onnx_path: Path = args.onnx_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    out_path = args.out or _default_output_path(onnx_path)

    onnx_module = _require_module(
        "onnx",
        "onnx2c requires the 'onnx' package. Install it with `pip install onnx`.",
    )
    onnx2torch_module = _require_module(
        "onnx2torch",
        "onnx2c requires the 'onnx2torch' package. Install it with `pip install onnx2torch`.",
    )

    model = onnx_module.load(onnx_path)
    example_inputs = _collect_example_inputs(onnx_module, model, args.default_dim)
    torch_module = onnx2torch_module.convert(model)
    torch_module.eval()
    _run_self_test(torch_module, example_inputs, args.self_test_runs)
    graph_module = _trace_module(torch_module)
    export_generic_c(
        graph_module,
        example_inputs,
        str(out_path),
        function_name=args.function_name,
        truncate_weights_after=args.truncate_weights_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
