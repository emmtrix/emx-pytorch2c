#!/usr/bin/env python
from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path

import torch
from torch.testing._internal.common_methods_invocations import op_db

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests import test_cref_ops  # noqa: E402


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _count_executed_tests() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"matches": 0, "invalid": 0})
    device = "cpu"

    for op in test_cref_ops.OPS_UNDER_TEST:
        constraints = test_cref_ops._constraints_for(op)
        allowed_dtypes = constraints["allowed_dtypes"]
        op_dtypes = set(op.dtypes)
        if allowed_dtypes is None:
            dtypes = sorted(op_dtypes, key=_dtype_name)
        else:
            dtypes = sorted(op_dtypes & set(allowed_dtypes), key=_dtype_name)

        for dtype in dtypes:
            for _ in test_cref_ops._iter_supported_samples(op, device, dtype, constraints):
                counts[op.name]["matches"] += 1

    return counts


def main() -> None:
    _ = op_db
    counts = _count_executed_tests()
    total = 0
    for op_name in sorted(counts):
        matches = counts[op_name]["matches"]
        invalid = counts[op_name]["invalid"]
        op_total = matches + invalid
        total += op_total
        print(
            f"{op_name}: total={op_total} "
            f"(matches_eager={matches}, invalid_shapes={invalid})"
        )
    print(f"all_ops_total: {total}")


if __name__ == "__main__":
    main()
