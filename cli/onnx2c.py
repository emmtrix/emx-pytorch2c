import argparse
import importlib
from pathlib import Path
from typing import Iterable, List

import torch

from codegen_backend import codegen_generic_backend
from codegen_backend.export import export_generic_c

torch.fx.wrap(len)


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


def _ensure_conv_kernel_shape(onnx_module, model) -> None:
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose"}:
            continue
        if any(attribute.name == "kernel_shape" for attribute in node.attribute):
            continue
        if len(node.input) < 2:
            continue
        weight_name = node.input[1]
        initializer = initializers.get(weight_name)
        if initializer is None or len(initializer.dims) < 3:
            continue
        kernel_shape = list(initializer.dims[2:])
        attribute = onnx_module.helper.make_attribute("kernel_shape", kernel_shape)
        node.attribute.extend([attribute])


def _patch_onnx2pytorch_for_fx(onnx2pytorch_module) -> None:
    try:
        onnx2pytorch_utils = onnx2pytorch_module.utils
    except AttributeError:
        return
    if getattr(onnx2pytorch_utils, "_onnx2c_fx_patched", False):
        return
    from torch.fx.proxy import Proxy
    from onnx2pytorch.operations import add as onnx2pytorch_add

    original_is_constant = onnx2pytorch_utils.is_constant
    original_add_forward = onnx2pytorch_add.Add.forward

    def _fx_safe_is_constant(value):
        if isinstance(value, Proxy):
            return False
        return original_is_constant(value)

    def _fx_safe_add_forward(self, *input):
        if any(isinstance(value, Proxy) for value in input):
            out = input[0]
            for inp in input[1:]:
                out = out + inp
            return out
        return original_add_forward(self, *input)

    onnx2pytorch_utils.is_constant = _fx_safe_is_constant
    onnx2pytorch_add.Add.forward = _fx_safe_add_forward
    onnx2pytorch_utils._onnx2c_fx_patched = True


def _disable_inplace_relu(torch_module: torch.nn.Module) -> None:
    for module in torch_module.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False


def _freeze_dynamic_reshape_shapes(
    graph_module: torch.fx.GraphModule, example_inputs: Iterable[torch.Tensor]
) -> None:
    try:
        from torch.fx.passes.shape_prop import ShapeProp
    except ImportError:
        return
    ShapeProp(graph_module).propagate(*example_inputs)
    for node in graph_module.graph.nodes:
        if node.op not in {"call_function", "call_method"}:
            continue
        if node.op == "call_function":
            target_name = getattr(node.target, "__name__", "")
            if node.target not in {
                torch.ops.aten.reshape.default,
                torch.ops.aten.view.default,
                torch.ops.aten._unsafe_view.default,
                torch.reshape,
                torch.Tensor.reshape,
                torch.Tensor.view,
            } and target_name not in {"reshape", "view"}:
                continue
            input_node = node.args[0]
        else:
            if node.target not in {"reshape", "view"}:
                continue
            input_node = node.args[0]
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            continue
        output_shape = tuple(tensor_meta.shape)
        node.op = "call_function"
        node.target = torch.ops.aten.reshape.default
        node.args = (input_node, output_shape)
        node.kwargs = {}
    graph_module.graph.lint()
    graph_module.recompile()


def _ensure_nonempty_graph(graph_module: torch.fx.GraphModule) -> None:
    has_op = any(
        node.op not in {"placeholder", "get_attr", "output"}
        for node in graph_module.graph.nodes
    )
    if has_op:
        return
    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    if not placeholders:
        return
    output_node = next(
        node for node in graph_module.graph.nodes if node.op == "output"
    )
    input_node = placeholders[0]
    with graph_module.graph.inserting_before(output_node):
        clone_node = graph_module.graph.call_function(
            torch.ops.aten.clone.default, (input_node,)
        )
    output_node.args = (clone_node,)
    graph_module.graph.lint()
    graph_module.recompile()


def _trace_module(
    torch_module: torch.nn.Module, example_inputs: Iterable[torch.Tensor]
) -> torch.fx.GraphModule:
    tracer = _InlineModuleTracer()
    graph = tracer.trace(torch_module, concrete_args=tuple(example_inputs))
    graph_module = torch.fx.GraphModule(torch_module, graph)
    _freeze_dynamic_reshape_shapes(graph_module, example_inputs)
    _ensure_nonempty_graph(graph_module)
    return graph_module


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
    onnx2pytorch_module = _require_module(
        "onnx2pytorch",
        "onnx2c requires the 'onnx2pytorch' package. Install it with `pip install onnx2pytorch`.",
    )

    model = onnx_module.load(onnx_path)
    _ensure_conv_kernel_shape(onnx_module, model)
    example_inputs = _collect_example_inputs(onnx_module, model, args.default_dim)
    torch_module = onnx2pytorch_module.ConvertModel(model)
    _patch_onnx2pytorch_for_fx(onnx2pytorch_module)
    _disable_inplace_relu(torch_module)
    torch_module.eval()
    _run_self_test(torch_module, example_inputs, args.self_test_runs)
    graph_module = _trace_module(torch_module, example_inputs)
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
