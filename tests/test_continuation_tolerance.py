"""Numerical tolerance must preserve distribution and greedy-token safety gates."""
import pytest
import torch

from rcuf.engine import _check_continuation_logits


def test_bf16_roundoff_at_low_probability_token_is_accepted():
    full = torch.tensor([4.0, 1.0, -1.0])
    continued = torch.tensor([4.0, 1.0, -0.8])
    report = _check_continuation_logits(continued, full, torch.bfloat16)
    assert report["split_full_first_token_matches"]
    assert report["split_full_kl"] < 0.001


def test_fp32_retains_original_absolute_tolerance():
    with pytest.raises(RuntimeError, match="Split vs full"):
        _check_continuation_logits(
            torch.tensor([4.0, 1.0, -0.8]), torch.tensor([4.0, 1.0, -1.0]), torch.float32
        )


def test_distribution_drift_fails_even_when_each_logit_is_close():
    full = torch.tensor([4.01, 4.0, 4.0, 4.0])
    continued = torch.tensor([4.25, 3.76, 3.76, 3.76])
    assert torch.allclose(full, continued, atol=0.25, rtol=0.02)
    with pytest.raises(RuntimeError, match="KL="):
        _check_continuation_logits(continued, full, torch.bfloat16)


def test_greedy_token_change_fails_despite_small_logit_error():
    with pytest.raises(RuntimeError, match="argmax match=False"):
        _check_continuation_logits(
            torch.tensor([4.0, 4.01]), torch.tensor([4.01, 4.0]), torch.bfloat16
        )
