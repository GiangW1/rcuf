import pytest

from rcuf.scoring import score

RECORD = {"project": "VEGA-731", "code": "ZUM-482", "budget": 42700}


@pytest.mark.parametrize("budget", ["42700", "42 700", "42,700", "42.700", "42\u202f700"])
def test_french_and_normalized_facts(budget):
    result = score(
        f"Le code d'autorisation du projet VEGA-731 est ZUM-482. Son budget approuvé est de {budget} euros.",
        RECORD,
    )
    assert result["language_ok"] and result["content_ok"] and result["joint_ok"]
    assert result["needs_review"]


def test_language_and_facts_are_separate():
    result = score(
        "The authorization code for project VEGA-731 is ZUM-482. Its approved budget is 42,700 euros.", RECORD
    )
    assert result["content_ok"] and not result["language_ok"]


def test_wrong_or_conflicting_facts_fail():
    result = score("Le code est ZUM-482 ou LEX-915. Le budget est de 42700 euros.", RECORD)
    assert not result["content_ok"]
    assert not score("Le code est ZUM-482. Le budget est de 42900 euros.", RECORD)["content_ok"]


@pytest.mark.parametrize(
    "text",
    [
        "Le code d’autorisation du projet VEGA-731 n’est pas ZUM-482. Son budget approuvé n’est pas de 42 700 euros.",
        "Le code d’autorisation du projet ORION-264 est ZUM-482. Son budget approuvé est de 42 700 euros.",
    ],
)
def test_correct_tokens_in_wrong_prose_are_not_joint_successes(text):
    result = score(text, RECORD)
    assert result["fact_tokens_match"]
    assert not result["content_ok"] and not result["joint_ok"]
    assert result["needs_review"]


def test_bare_fields_and_truncation_require_review():
    result = score("ZUM-482 42700", RECORD)
    assert result["content_ok"] and not result["language_ok"] and result["needs_review"]
    result = score(
        "Le code d'autorisation du projet est ZUM-482. Son budget approuvé est de 42700 euros.", RECORD, True
    )
    assert "generation_truncated" in result["review_reasons"]
