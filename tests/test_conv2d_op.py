import pytest
import torch

from c_ref_backend.cffi_bindings import RefBackendError, run_conv2d


def _allocate_conv2d_output(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride,
    padding,
    dilation,
    groups,
) -> torch.Tensor:
    expected = torch.nn.functional.conv2d(
        input_tensor,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return torch.empty_like(expected, memory_format=torch.contiguous_format)


def test_conv2d_matches_eager_grouped():
    input_tensor = torch.randn(2, 4, 6, 6, dtype=torch.float32)
    weight = torch.randn(6, 2, 3, 3, dtype=torch.float32)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 2
    out = _allocate_conv2d_output(
        input_tensor, weight, stride, padding, dilation, groups
    )

    run_conv2d(
        input_tensor,
        weight,
        out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    expected = torch.nn.functional.conv2d(
        input_tensor,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    torch.testing.assert_close(out, expected)


def test_conv2d_rejects_non_contiguous():
    input_tensor = torch.randn(1, 3, 5, 6, dtype=torch.float32).transpose(2, 3)
    weight = torch.randn(4, 3, 3, 3, dtype=torch.float32)
    out = _allocate_conv2d_output(input_tensor.contiguous(), weight, 1, 0, 1, 1)
    with pytest.raises(RefBackendError, match="contiguous"):
        run_conv2d(input_tensor, weight, out, stride=1, padding=0, dilation=1, groups=1)


def test_conv2d_rejects_channel_mismatch():
    input_tensor = torch.randn(1, 3, 5, 5, dtype=torch.float32)
    weight = torch.randn(4, 2, 3, 3, dtype=torch.float32)
    out = torch.empty((1, 1, 1, 1), dtype=torch.float32)
    with pytest.raises(
        RefBackendError,
        match=r"input channels to match weight channels \* groups",
    ):
        run_conv2d(input_tensor, weight, out, stride=1, padding=0, dilation=1, groups=1)
