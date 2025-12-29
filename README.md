# Reference Backend (Phase 1)

This repository provides a correctness-first Generic C Reference Backend integrated with TorchDynamo / `torch.compile`. The initial version supports a single operation: FP32 out-of-place `add` with contiguous tensors and matching shapes.

## Setup

```bash
pip install -e .
```

## Build the native library

The extension module (`ref_backend._ref_backend`) provides the C ABI and is built via setuptools.

```bash
python -m ref_backend.build
```

## Run tests

```bash
pytest -q
```

## Example usage

```python
import torch
from ref_backend.backend import ref_backend_backend


def f(a, b):
    return a + b

compiled = torch.compile(f, backend=ref_backend_backend)

a = torch.randn(4, 4)
b = torch.randn(4, 4)
print(compiled(a, b))
```

## Scope limitations

* CPU only
* `float32` only
* contiguous tensors only
* out-of-place `add(a, b) -> out`
* identical shapes only
