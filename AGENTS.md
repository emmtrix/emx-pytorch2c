# Agent Instructions

## Tests

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

Overview: new operators are implemented through the codegen backend.

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
