from pathlib import Path

import torch
from codegen_backend.export import export_generic_c
from test_codegen_golden import _assert_codegen_source_matches

REFERENCE_DIR = Path(__file__).resolve().parent / "exportc_refs"


def _normalize_source(source: str) -> str:
    return "\n".join(line.rstrip() for line in source.splitlines()) + "\n"


def _export_source(tmp_path: Path, *args, **kwargs) -> str:
    out_path = tmp_path / "exported_model.c"
    return export_generic_c(*args, out_path=str(out_path), **kwargs)


def test_export_generic_c_inlines_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    class AddmmModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 3))
            self.bias = torch.nn.Parameter(torch.randn(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.addmm(self.bias, x, self.weight)

    model = AddmmModule()
    model.eval()
    example_input = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    source_fn = lambda gm, example_inputs: _normalize_source(
        _export_source(tmp_path, gm, example_inputs)
    )
    _assert_codegen_source_matches(
        "exported_model.c",
        source_fn,
        model,
        (example_input,),
        reference_dir=REFERENCE_DIR,
    )


def test_export_generic_c_variable_dim_inputs(tmp_path: Path) -> None:
    class AddModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    model = AddModule()
    model.eval()
    example_input = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    source_fn = lambda gm, example_inputs: _normalize_source(
        _export_source(
            tmp_path,
            gm,
            example_inputs,
            variable_dim_inputs={0: [0]},
        )
    )
    _assert_codegen_source_matches(
        "exported_variable_model.c",
        source_fn,
        model,
        (example_input,),
        reference_dir=REFERENCE_DIR,
    )
