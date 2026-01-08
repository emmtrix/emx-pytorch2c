# emx-pytorch2c

emx-pytorch2c generates simple, correct, generic, and easily analyzable C code from PyTorch workloads. Performance optimization of the generated C code is explicitly **not** a goal.

## Goals

* Simple, correctness-preserving C code
* Generic code that is easy to analyze and verify
* A solid foundation for further analysis and verification tools

## Non-Goals

* Performance optimization of the generated C code

## Features

* Supported operators (codegen backend): see [`tests/list_aten_core_ops_ref.md`](tests/list_aten_core_ops_ref.md) for core ops and [`tests/list_aten_ops_ref.md`](tests/list_aten_ops_ref.md) for all ATen ops.
* `torch.compile` backend for generating generic C code from PyTorch workloads.
* Export utility for emitting standalone C sources from Python functions.
* ONNX-to-C conversion via the `cli.onnx2c` command-line interface.
* Supported dtypes for codegen graphs: `torch.float32`, `torch.int8`, `torch.int16`, `torch.int32`, `torch.uint8`, `torch.uint16`, `torch.uint32`, `torch.bool`.
  * Index tensors for embedding/gather-style ops may also use `torch.int64`.

## Requirements

* Python >= 3.8
* PyTorch
* Jinja2
* Optional (for `cli.onnx2c`): onnx, onnx2pytorch

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ci.txt
pip install -e .
```

## Quickstart

```python
import torch
from codegen_backend.backend import codegen_generic_backend


def f(a, b):
    return a + b

compiled = torch.compile(f, backend=codegen_generic_backend)
print(compiled(torch.randn(2, 2), torch.randn(2, 2)))
```

## Usage

### 1) `torch.compile` (Codegen Backend)

```python
import torch
from codegen_backend.backend import codegen_generic_backend


def f(a, b):
    return a + b

compiled = torch.compile(f, backend=codegen_generic_backend)

a = torch.randn(4, 4)
b = torch.randn(4, 4)
print(compiled(a, b))
```

### 2) Export generic C code

```python
import torch
from codegen_backend.export import export_generic_c


def f(a, b):
    return a + b

example_inputs = (torch.randn(4, 4), torch.randn(4, 4))
result = export_generic_c(f, example_inputs)
print(result.c_source)
```

### 3) ONNX to C via CLI

Requires `onnx` and `onnx2pytorch`:

```bash
pip install onnx onnx2pytorch
```

```bash
python -m cli.onnx2c --help
```

Example:

```bash
python -m cli.onnx2c model.onnx -o model.c --self-test-runs 0
```

## Tests

```bash
PYTHONPATH=src pytest -n auto --maxfail=5 -q
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
