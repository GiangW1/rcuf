"""Automatic scores are preliminary; ambiguous outputs require blind review."""

from __future__ import annotations

import re
from functools import lru_cache

CODE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
IDENTIFIER = re.compile(r"\b[A-Za-z]{3,12}-\d{3}\b")
NUMBER = re.compile(r"(?<![\w-])\d{1,3}(?:[ ,.\u00a0\u202f]\d{3})+(?!\d)|(?<![\w-])\d+(?!\w)")


@lru_cache(maxsize=1)
def language_identifier():
    from langid.langid import LanguageIdentifier, model

    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def score(text: str, record: dict, truncated: bool = False) -> dict:
    clean = IDENTIFIER.sub(" ", text)
    numbers = [int(re.sub(r"\D", "", match.group())) for match in NUMBER.finditer(clean)]
    codes = CODE.findall(text)
    content_ok = (
        bool(codes)
        and set(codes) == {record["code"]}
        and bool(numbers)
        and set(numbers) == {record["budget"]}
    )
    clean = NUMBER.sub(" ", clean)
    clean = re.sub(r"\b(?:EUR|euros?)\b", " ", clean, flags=re.IGNORECASE)
    words = re.findall(r"[^\W\d_]+", clean, flags=re.UNICODE)
    language, confidence = language_identifier().classify(clean) if words else ("unknown", 0.0)
    sentences = [part.strip() for part in re.split(r"[.!?\n]+", clean) if part.strip()]
    sentence_languages = []
    mixed = False
    for sentence in sentences:
        if len(re.findall(r"[^\W\d_]+", sentence)) >= 5:
            lang, probability = language_identifier().classify(sentence)
            sentence_languages.append({"language": lang, "confidence": probability})
            mixed |= lang != "fr" and probability >= 0.9
    language_ok = language == "fr" and not mixed and len(words) >= 4
    reasons = []
    if len(words) < 8:
        reasons.append("short_answer")
    if confidence < 0.9:
        reasons.append("uncertain_language")
    if mixed:
        reasons.append("mixed_language")
    if truncated:
        reasons.append("generation_truncated")
    if not content_ok:
        reasons.append("content_mismatch_or_unparsed")
    return {
        "language_ok": language_ok,
        "content_ok": content_ok,
        "joint_ok": language_ok and content_ok,
        "detected_language": language,
        "language_confidence": float(confidence),
        "sentence_languages": sentence_languages,
        "codes_found": codes,
        "numbers_found": numbers,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }
