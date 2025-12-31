# Agent Instructions

## Tests

Build the backend before running tests:

```bash
PYTHONPATH=src python -m ref_backend.build
```

Run the test suite with:

```bash
PYTHONPATH=src pytest -q
```

## Operator Guidelines

When adding a new operator, follow the same structure as the existing `add`/`matmul` paths:

- **C implementation**: add a new `csrc/ops_<op>.c` with a `ref_run_<op>` entry point,
  separate raw numeric kernels (e.g. `add_f32`) from tensor-view validation/dispatch
  logic to keep the math reusable and easier to test, keep raw kernels typed to the
  PyTorch dtypes they implement (e.g. `float` for `torch.float32`), use `write_error`
  for validation failures, and keep messages consistent with the Python wrapper checks.
- **Dispatcher registration**: register the new op in `csrc/ref_backend.c` and assign
  a new `RefOpKind` value in `src/ref_backend/cffi_bindings.py`.
- **Python bindings**: add a `run_<op>` wrapper in `src/ref_backend/cffi_bindings.py`
  that enforces dtype/shape/contiguity rules before calling into the C backend.
- **Backend mapping**: wire the op in `src/ref_backend/backend.py` so torch ops map
  to the new implementation.
- **Tests**: add coverage in `tests/` for correct results and error handling
  (e.g. non-contiguous tensors, shape mismatches).
