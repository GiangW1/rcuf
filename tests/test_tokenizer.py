"""Optional real tokenizer checks, no model weights. Enable RCUF_TOKENIZER_TESTS=1."""

import os

import pytest
from transformers import AutoTokenizer

from rcuf.data import records_for, render_prompt, system_tokens

pytestmark = pytest.mark.skipif(
    os.environ.get("RCUF_TOKENIZER_TESTS") != "1", reason="Opt-in network tokenizer integration"
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", use_fast=True)


def test_anchor_prefix_uses_valid_official_template(tokenizer):
    french = system_tokens(tokenizer, "French")
    english = system_tokens(tokenizer, "English")
    assert len(french) == len(english)
    assert french != english
    decoded = tokenizer.decode(french)
    assert "French" in decoded
    assert "Anchor query" not in decoded and "<|im_start|>user" not in decoded


def test_split_is_exact_prefix_of_full_non_thinking_prompt(tokenizer):
    records, target = records_for(0, 1729)
    reference = "Some ordinary reference material.\n\nQuestion:\nAn unrelated quotation."
    prompt = render_prompt(tokenizer, reference, records, target)
    prefix = tokenizer.decode(prompt["input_ids"][: prompt["question_cut"]])
    suffix = tokenizer.decode(prompt["input_ids"][prompt["question_cut"] :])
    assert "</records>" in prefix
    assert "What are the authorization code" in suffix
    assert "<think>\n\n</think>" in suffix
    system = system_tokens(tokenizer, "French")
    assert prompt["input_ids"][: len(system)] == system
