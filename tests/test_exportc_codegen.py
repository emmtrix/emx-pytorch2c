import os
from pathlib import Path

import torch
import torch.fx

from codegen_backend.export import export_generic_c

REFERENCE_DIR = Path(__file__).resolve().parent / "golden"


def _normalize_source(source: str) -> str:
    return "\n".join(line.rstrip() for line in source.splitlines()) + "\n"


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
    gm = torch.fx.symbolic_trace(model)
    out_path = tmp_path / "exported_model.c"
    source = export_generic_c(gm, (example_input,), str(out_path))
    source = _normalize_source(source)
    reference_path = REFERENCE_DIR / "exported_model.c"
    if os.getenv("UPDATE_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = _normalize_source(reference_path.read_text(encoding="utf-8"))
    assert source == expected
