# emmtrix PyTorch to C Compiler

emx-pytorch2c generates simple, correct, generic, and easily analyzable C code from PyTorch workloads. Performance optimization of the generated C code is explicitly **not** a goal.

## Goals

* Simple, correctness-preserving C code
* Generic code that is easy to analyze and verify
* A solid foundation for further analysis and verification tools
* C code tuned for later vectorization via [emmtrix Code vectorizer (eCV)](https://www.emmtrix.com/tools/emmtrix-code-vectorizer)

## Non-Goals

* Performance optimization of the generated C code

## Features

* Supported operators: 
  * see [`tests/list_core_ops_ref.md`](tests/list_core_ops_ref.md) for core ops.
  * see [`tests/list_aten_ops_ref.md`](tests/list_aten_ops_ref.md) for all ATen ops.
* Export utility for emitting standalone C sources from Python functions.
* `torch.compile` backend for generating generic C code from PyTorch workloads for verification.
* ONNX-to-C conversion via the `cli.onnx2c` command-line interface.
* Supported dtypes:
  * `torch.float32` / `torch.float64`
  * `torch.int8` / `torch.uint8`
  * `torch.int16` / `torch.uint16`
  * `torch.int32` / `torch.uint32`
  * `torch.bool`
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

## Exported C Code Examples

### Example 1: Simple Add

Below is a minimal example of the emitted C for a simple add. It highlights two
key characteristics of the generated code: explicit fixed-size arrays in the
function signatures and straightforward loop nests over each tensor dimension.

```c
void node1_add_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[i0][i1], b[i0][i1]);
        }
    }
}
```

In other words, shapes are materialized as explicit array extents, and compute
is expressed as deterministic, nested `for` loops rather than vectorized or
opaque library calls.

### Example 2: Heap/Stack Temporary Allocation

The generic backend allocates intermediate buffers on the stack until they
exceed `temp_allocation_threshold`, at which point they are allocated with
`malloc` and freed after use. For example, generated C can contain both a
stack temporary and a heap temporary:

```c
void ref_codegen_main_f32(const float input_0[1], const float input_1[1],
                          const float input_2[1][2][2][5],
                          const float input_3[1][2][2][5],
                          float out[1][2][2][5]) {
    float tmp_0[1];
    float (*tmp_1)[2][2][5] = malloc(sizeof(float) * 1 * 2 * 2 * 5);
    node1_add_f32(input_0, input_1, tmp_0);
    node2_add_f32(input_2, input_3, tmp_1);
    node3_add_f32(tmp_0, tmp_1, out);
    free(tmp_1);
}
```


## Usage

### 1) Export generic C code

```python
import torch
from codegen_backend.export import export_generic_c


def f(a, b):
    return a + b

example_inputs = (torch.randn(4, 4), torch.randn(4, 4))
result = export_generic_c(f, example_inputs, temp_allocation_threshold=2048)
print(result.c_source)
```

### 2) ONNX to C via CLI

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

Tune temporary buffer allocation (defaults to 1024 bytes; set 0 for stack-only):

```bash
python -m cli.onnx2c model.onnx -o model.c --tmp-malloc-threshold 2048
```

### 3) `torch.compile` (for verification)

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

## Tests

```bash
PYTHONPATH=src pytest -n auto --maxfail=5 -q
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
