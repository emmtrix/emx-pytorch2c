import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from cli import onnx2c
from codegen_backend.export import export_generic_c

onnx = pytest.importorskip("onnx")
onnx2torch = pytest.importorskip("onnx2torch")

REFERENCE_DIR = Path(__file__).resolve().parent / "onnx2c_refs"
ONNX_DIR = Path(__file__).resolve().parent / "onnx2c"
ONNX_CASES = [
    pytest.param(
        ONNX_DIR / "test_onnx01_add_bias_tiny.onnx",
        id="onnx01_add_bias_tiny",
    ),
    pytest.param(
        ONNX_DIR / "test_onnx02_cnn_approx500k_c124_c248_fcin12288.onnx",
        id="onnx02_cnn_approx500k_c124_c248_fcin12288",
        marks=pytest.mark.skip(
            reason="onnx2torch batchnorm trace hits control flow"
        ),
    ),
    pytest.param(
        ONNX_DIR / "test_onnx03_conv2d_tiny.onnx",
        id="onnx03_conv2d_tiny",
    ),
    pytest.param(
        ONNX_DIR / "test_onnx04_identity_tiny.onnx",
        id="onnx04_identity_tiny",
    ),
    pytest.param(
        ONNX_DIR / "test_onnx05_lstm_approx500k_h144_seq16_in64_out16.onnx",
        id="onnx05_lstm_approx500k_h144_seq16_in64_out16",
        marks=pytest.mark.skip(
            reason="onnx2torch LSTM converter not implemented"
        ),
    ),
    pytest.param(
        ONNX_DIR / "test_onnx06_matmul_tiny.onnx",
        id="onnx06_matmul_tiny",
    ),
    pytest.param(
        ONNX_DIR / "test_onnx09_mnist_simplified.onnx",
        id="onnx09_mnist_simplified",
        marks=pytest.mark.skip(
            reason="reshape/linear not yet supported in codegen backend"
        ),
    ),
    pytest.param(
        ONNX_DIR / "test_onnx10_hello.onnx",
        id="onnx10_hello",
    ),
]


def _onnx_to_source(onnx_path: Path, tmp_path: Path) -> str:
    model = onnx.load(onnx_path)
    example_inputs = onnx2c._collect_example_inputs(onnx, model, default_dim=1)
    torch_module = onnx2torch.convert(model)
    torch_module.eval()
    graph_module = onnx2c._trace_module(torch_module)
    out_path = tmp_path / f"{onnx_path.stem}.c"
    return export_generic_c(graph_module, example_inputs, str(out_path)).lstrip()


def _assert_onnx2c_source_matches(onnx_path: Path, tmp_path: Path) -> None:
    reference_path = REFERENCE_DIR / f"{onnx_path.stem}.c"
    source = _onnx_to_source(onnx_path, tmp_path)
    if os.getenv("UPDATE_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = reference_path.read_text(encoding="utf-8")
    assert source == expected


@pytest.mark.parametrize("onnx_path", ONNX_CASES)
def test_onnx2c_golden(onnx_path, tmp_path):
    _assert_onnx2c_source_matches(onnx_path, tmp_path)
