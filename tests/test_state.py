import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache

from rcuf.state import clone_cache, edit_cache, edit_state, make_anchor, recurrent_states


def test_projection_is_on_key_axis_and_idempotent():
    torch.manual_seed(31)
    # Deliberately non-square: a silent transpose mistake must fail here.
    french = torch.randn(1, 2, 7, 11)
    english = torch.randn_like(french)
    state = torch.randn_like(french)
    anchor = make_anchor(french, english, rank=3)
    edited = edit_state(state, anchor, "replace", 1.0)
    u = anchor.basis
    projection = u @ u.transpose(-1, -2)
    complement = torch.eye(7) - projection
    torch.testing.assert_close(u.transpose(-1, -2) @ edited, u.transpose(-1, -2) @ french)
    torch.testing.assert_close(complement @ edited, complement @ state, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(edit_state(edited, anchor, "replace", 1.0), edited, atol=2e-6, rtol=2e-6)


def test_soft_replacement_is_not_idempotent():
    torch.manual_seed(2)
    anchor = make_anchor(torch.randn(1, 2, 7, 11), torch.randn(1, 2, 7, 11), 3)
    state = torch.randn(1, 2, 7, 11)
    once = edit_state(state, anchor, "replace", 0.5)
    twice = edit_state(once, anchor, "replace", 0.5)
    assert not torch.allclose(once, twice)


def test_zero_contrast_never_uses_arbitrary_svd_directions():
    reference = torch.randn(1, 2, 7, 11)
    anchor = make_anchor(reference, reference.clone(), rank=3)
    state = torch.randn_like(reference)
    assert int(anchor.effective_rank.sum()) == 0
    torch.testing.assert_close(edit_state(state, anchor, "replace", 1.0), state, atol=0, rtol=0)


def test_cache_forks_are_independent_and_only_recurrence_is_edited():
    config = Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"])
    cache = DynamicCache(config=config)
    state = torch.randn(1, 2, 7, 11)
    conv = torch.randn(1, 9, 4)
    keys = torch.randn(1, 2, 5, 8)
    values = torch.randn_like(keys)
    cache.update_recurrent_state(state, 0)
    cache.update_conv_state(conv, 0, conv_kernel_size=4)
    cache.update(keys, values, 1)
    first, second = clone_cache(cache, "cpu"), clone_cache(cache, "cpu")
    anchor = make_anchor(torch.randn_like(state), torch.randn_like(state), 3)
    edit_cache(first, {0: anchor}, "replace", 1.0)
    torch.testing.assert_close(recurrent_states(second)[0], recurrent_states(cache)[0])
    assert not torch.equal(recurrent_states(first)[0], recurrent_states(cache)[0])
    torch.testing.assert_close(first.layers[1].keys, keys)
    torch.testing.assert_close(first.layers[1].values, values)
    torch.testing.assert_close(first.layers[0].conv_states[0], conv)
    assert first.get_seq_length() == second.get_seq_length() == 5
