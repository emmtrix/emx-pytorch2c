# Architecture Overview

This repository focuses on generating simple, correctness-preserving C code from
PyTorch workloads. The high-level flow is:

1. A user provides a PyTorch function, either via `torch.compile` or the export
   utilities.
2. The codegen backend maps supported PyTorch operators to C-emission templates.
3. Templates emit generic C sources that prioritize readability and
   analyzability over performance.

## Major Areas

- `src/codegen_backend/`: Core code generation backend and templates for
  emitting C.
- `cli/`: Command-line tooling, including ONNX-to-C conversion.
- `src/`: Export helpers and other supporting utilities.
- `tests/`: Operator coverage, codegen validation, and golden reference checks.

## Design Principles

- **Simplicity and correctness first**: emitted C prioritizes clarity and
  verifiability.
- **Template-driven codegen**: new operator logic typically extends Jinja
  templates instead of string assembly.
- **Explicit operator coverage**: supported ops and dtype coverage are tracked
  in tests and backend registries.
