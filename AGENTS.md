# Agent Instructions

## Tests

Build the backend before running tests:

```bash
PYTHONPATH=src python -m c_ref_backend.build
```

Run the test suite with:

```bash
PYTHONPATH=src pytest -q
```

### Codegen reference updates

The codegen backend tests compare generated C sources against reference files. To
refresh the references after intentional changes, set `UPDATE_REFS=1` when
running tests:

```bash
UPDATE_REFS=1 PYTHONPATH=src pytest -q
```

## Operator Guidelines

When adding a new operator, follow the same structure as the existing `add`/`matmul` paths:

- **C implementation**: add a new `csrc/ops_<op>.c` with a `ref_run_<op>` entry point,
  separate raw numeric kernels (e.g. `add_f32`) from tensor-view validation/dispatch
  logic to keep the math reusable and easier to test, keep raw kernels typed to the
  PyTorch dtypes they implement (e.g. `float` for `torch.float32`), use `write_error`
  for validation failures, and keep messages consistent with the Python wrapper checks.
- **Dispatcher registration**: register the new op in `csrc/c_ref_backend.c` and assign
  a new `RefOpKind` value in `src/c_ref_backend/cffi_bindings.py`.
- **Python bindings**: add a `run_<op>` wrapper in `src/c_ref_backend/cffi_bindings.py`
  that enforces dtype/shape/contiguity rules before calling into the C backend.
- **Backend mapping**: wire the op in `src/c_ref_backend/backend.py` so torch ops map
  to the new implementation.
- **Tests**: add coverage in `tests/` for correct results and error handling
  (e.g. non-contiguous tensors, shape mismatches).
