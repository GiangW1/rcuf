"""Provisional bilingual prose labels and strict, language-independent format checks."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

from .scoring import IDENTIFIER, NUMBER, language_identifier
from .scoring import score as prose_score


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def parse_format(text, policy):
    try:
        if policy == "json":
            value = json.loads(text, object_pairs_hook=unique_object)
            if not isinstance(value, dict) or set(value) != {"code", "budget_eur"}:
                return None
            if type(value["budget_eur"]) is not int or not isinstance(value["code"], str):
                return None
            code, budget = value["code"], value["budget_eur"]
        else:
            if re.search(r"<!|<\?", text):
                return None
            root = ET.fromstring(text)
            children = list(root)
            if (
                root.tag != "answer"
                or root.attrib
                or (root.text or "").strip()
                or len(children) != 2
                or {c.tag for c in children} != {"code", "budget_eur"}
                or any(c.attrib or len(c) or (c.tail or "").strip() for c in children)
            ):
                return None
            fields = {c.tag: c.text or "" for c in children}
            code, number = fields["code"].strip(), fields["budget_eur"].strip()
            if not re.fullmatch(r"[+-]?[0-9]+", number):
                return None
            budget = int(number)
        return {"code": code, "budget_eur": budget}
    except (ValueError, TypeError, ET.ParseError):
        return None


def score(text, example, truncated=False):
    preliminary = prose_score(text, example["target"], truncated)
    if example["family"] == "language":
        detected = preliminary["detected_language"]
        clean = NUMBER.sub(" ", IDENTIFIER.sub(" ", text))
        clean = re.sub(r"\b(?:EUR|euros?)\b", " ", clean, flags=re.IGNORECASE)
        words = re.findall(r"[^\W\d_]+", clean)
        sentence_labels = [
            language_identifier().classify(sentence)
            for sentence in re.split(r"[.!?\n]+", clean)
            if len(re.findall(r"[^\W\d_]+", sentence)) >= 3
        ]
        confident = {language for language, probability in sentence_labels if probability >= 0.9}
        if len(confident) > 1 or (confident and detected not in confident):
            observed = "mixed"
        elif preliminary["language_confidence"] < 0.9 or detected not in {"en", "fr"}:
            observed = "other" if preliminary["language_confidence"] >= 0.9 else "unknown"
        elif len(words) < 4:
            observed = "unknown"
        else:
            observed = detected
        rule = None if example["policy"] == "none" else observed == example["policy"] and not truncated
        return {
            "response_language": observed,
            "rule_ok": rule,
            "content_ok": preliminary["content_ok"],
            "joint_ok": rule is True and preliminary["content_ok"],
            "language_confidence": preliminary["language_confidence"],
            "reviewed": False,
        }
    parsed = parse_format(text, example["policy"])
    rule = parsed is not None and not truncated
    content = (
        parsed == {"code": example["target"]["code"], "budget_eur": example["target"]["budget"]}
        if parsed is not None
        else preliminary["content_ok"]
    )
    return {"rule_ok": rule, "content_ok": content, "joint_ok": rule and content, "reviewed": False}
