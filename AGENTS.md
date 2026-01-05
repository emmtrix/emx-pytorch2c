# Agent Instructions

## Tests

Build the backend only when needed (for example, after changing C sources or
backend build logic):

```bash
PYTHONPATH=src python -m c_ref_backend.build
```

Run the test suite with:

```bash
PYTHONPATH=src pytest -n auto --maxfail=5 -q
```

Documentation-only changes in `*.md` files do not require rerunning tests.

When reporting executed tests, include the test duration in your feedback.

### Golden reference updates

The codegen backend tests compare generated C sources against golden reference files. To
refresh the references after intentional changes, set `UPDATE_REFS=1` when
running tests:

```bash
UPDATE_REFS=1 PYTHONPATH=src pytest ...
```

## Operator Guidelines

Overview: there are two backends for new operators—C ref backend and codegen backend.
Rule of thumb: use C ref for handwritten reference semantics; use codegen for generated kernel support.
See the two subsections below for the correct workflow.

### C ref backend: adding a new operator

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

### Codegen backend: adding a new operator

- Register the op in `src/codegen_backend/backend.py` by adding it to `SUPPORTED_OPS` via `_binary_spec`, `_unary_spec`, or a custom `_OpSpec`.
- Register the new op in `tests/test_codegen_ops.py` (supported ops list / OpInfo coverage) so it is exercised by the standard test suite.
- Map targets in `TARGET_REGISTRY`, and wire in-place behavior with `inplace_targets`/`inplace_arg_index` when needed.
- Check whether a corresponding in-place variant exists (e.g. `aten.<op>_...` / `Tensor.<op>_`); if it does, register it via `inplace_targets`/`inplace_arg_index` and ensure it is covered by tests.
- Ensure dtype coverage via `_CODEGEN_DTYPES` and `_INTEGER_CODEGEN_DTYPES`, extending them if the op needs additional dtypes.
- If the op needs a new kernel shape or custom emission logic, prefer adding or updating a Jinja template under `src/codegen_backend/templates/*.c.j2` instead of inline string assembly.
- Test with reference update.

## Template Prompt Guidance

When writing a template prompt, check the `prompts/` directory for available prompt files.
