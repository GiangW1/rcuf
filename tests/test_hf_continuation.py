"""Actual HF model/cache integration on CPU, using tiny random weights, not a mock cache."""

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from rcuf.engine import Engine, check_cache_paths
from rcuf.state import clone_cache, recurrent_states


def tiny_engine(chunk_size=8):
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        },
        pad_token_id=0,
        eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    engine = Engine.__new__(Engine)
    engine.device = torch.device("cpu")
    engine.config = {"prefill_chunk_size": chunk_size, "max_new_tokens": 4}
    engine.model = Qwen3_5ForCausalLM(config).eval()
    engine.model.requires_grad_(False)
    return engine


def test_multi_token_cached_continuation_matches_unsplit_forward():
    engine = tiny_engine()
    tokens = list(range(3, 32))
    full, full_logits = engine.forward(tokens)
    base, _ = engine.forward(tokens[:13])
    split, split_logits = engine.forward(tokens[13:], base)
    torch.testing.assert_close(split_logits, full_logits, atol=3e-6, rtol=2e-5)
    assert full.get_seq_length() == split.get_seq_length() == len(tokens)
    for i, state in recurrent_states(full).items():
        torch.testing.assert_close(state, recurrent_states(split)[i], atol=4e-6, rtol=3e-5)


def test_chunked_prefill_and_runtime_sanity_gate():
    engine = tiny_engine(chunk_size=7)
    report = check_cache_paths(engine, list(range(3, 30)), split=18)
    assert report["repeat_max_logits_error"] == 0
    assert report["perturbation_max_logits_effect"] > 1e-6
    assert report["split_full_first_token_matches"]


def test_cached_edit_survives_multi_token_suffix():
    engine = tiny_engine()
    tokens = list(range(3, 28))
    base, _ = engine.forward(tokens[:18])
    edited = clone_cache(base)
    recurrent_states(edited)[0].add_(0.1)
    _, original_logits = engine.forward(tokens[18:], base)
    _, edited_logits = engine.forward(tokens[18:], edited)
    assert not torch.allclose(original_logits, edited_logits)


def test_conditional_checkpoint_loads_all_text_weights(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5VisionConfig

    text = tiny_engine().model.config
    text.tie_word_embeddings = True  # matches the real 4B checkpoint, which omits a separate LM head
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=48,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
    )
    config = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True)
    source = Qwen3_5ForConditionalGeneration(config).eval()
    source.save_pretrained(tmp_path)
    loaded, audit = Qwen3_5ForCausalLM.from_pretrained(
        tmp_path, config=text, output_loading_info=True, attn_implementation="sdpa"
    )
    assert not audit["missing_keys"] and not audit["mismatched_keys"]
    torch.testing.assert_close(
        loaded.model.embed_tokens.weight, source.model.language_model.embed_tokens.weight
    )
    for i in range(4):
        torch.testing.assert_close(
            loaded.model.layers[i].mlp.down_proj.weight,
            source.model.language_model.layers[i].mlp.down_proj.weight,
        )
