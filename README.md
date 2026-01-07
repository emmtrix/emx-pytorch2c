# Codegen Backend

This project generates simple, correct, generic, and easily analyzable C code from PyTorch workloads. Performance optimization of the generated C code is explicitly **not** a goal.

## Goals

* Simple, correctness-preserving C code
* Generic code that is easy to analyze and verify
* A solid foundation for further analysis and verification tools

## Non-Goals

* Performance optimization of the generated C code

## Setup

```bash
pip install -e .
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

```bash
python -m cli.onnx2c --help
```

## Tests

```bash
PYTHONPATH=src pytest -n auto --maxfail=5 -q
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
