import pytest
import torch

from c_ref_backend.cffi_bindings import RefBackendError, run_lstm


def _call_lstm(
    input_tensor: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    batch_first: bool,
):
    return torch.ops.aten.lstm.input(
        input_tensor,
        (h0, c0),
        [weight_ih, weight_hh, bias_ih, bias_hh],
        True,
        1,
        0.0,
        False,
        False,
        batch_first,
    )


@pytest.mark.parametrize("batch_first", [False, True])
def test_lstm_matches_eager(batch_first):
    seq_len = 3
    batch = 2
    input_size = 4
    hidden_size = 5
    if batch_first:
        input_tensor = torch.randn(batch, seq_len, input_size)
    else:
        input_tensor = torch.randn(seq_len, batch, input_size)
    h0 = torch.randn(1, batch, hidden_size)
    c0 = torch.randn(1, batch, hidden_size)
    weight_ih = torch.randn(4 * hidden_size, input_size)
    weight_hh = torch.randn(4 * hidden_size, hidden_size)
    bias_ih = torch.randn(4 * hidden_size)
    bias_hh = torch.randn(4 * hidden_size)

    if batch_first:
        out = torch.empty(batch, seq_len, hidden_size)
    else:
        out = torch.empty(seq_len, batch, hidden_size)
    h_n = torch.empty(1, batch, hidden_size)
    c_n = torch.empty(1, batch, hidden_size)
    run_lstm(
        input_tensor,
        (h0, c0),
        (weight_ih, weight_hh, bias_ih, bias_hh),
        True,
        1,
        0.0,
        False,
        False,
        batch_first,
        out,
        h_n,
        c_n,
    )
    result = (out, h_n, c_n)
    expected = _call_lstm(
        input_tensor,
        h0,
        c0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        batch_first,
    )

    assert len(result) == 3
    for actual, exp in zip(result, expected):
        torch.testing.assert_close(actual, exp)


def test_lstm_rejects_noncontiguous_input():
    seq_len = 3
    batch = 2
    input_size = 4
    hidden_size = 5
    input_tensor = torch.randn(seq_len * 2, batch, input_size)[::2]
    h0 = torch.randn(1, batch, hidden_size)
    c0 = torch.randn(1, batch, hidden_size)
    weight_ih = torch.randn(4 * hidden_size, input_size)
    weight_hh = torch.randn(4 * hidden_size, hidden_size)
    bias_ih = torch.randn(4 * hidden_size)
    bias_hh = torch.randn(4 * hidden_size)
    out = torch.empty(seq_len, batch, hidden_size)
    h_n = torch.empty(1, batch, hidden_size)
    c_n = torch.empty(1, batch, hidden_size)

    with pytest.raises(RefBackendError, match="lstm requires contiguous tensors"):
        run_lstm(
            input_tensor,
            (h0, c0),
            (weight_ih, weight_hh, bias_ih, bias_hh),
            True,
            1,
            0.0,
            False,
            False,
            False,
            out,
            h_n,
            c_n,
        )
