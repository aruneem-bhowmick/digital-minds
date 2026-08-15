"""Tests for prism.prompts -- prompt templates and the control set (REQ-4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from prism.prompts import (
    DEFAULT_CONTROL_QUESTIONS_PATH,
    MIN_CONTROL_QUESTIONS,
    detection_prompt,
    naming_subtask_prompt,
    unrelated_control_prompt,
)

# --- detection_prompt / naming_subtask_prompt -------------------------------


def test_detection_prompt_returns_a_nonempty_string() -> None:
    assert isinstance(detection_prompt(), str)
    assert detection_prompt().strip() != ""


def test_detection_prompt_does_not_presuppose_an_injection() -> None:
    # REQ-7's no-injection baseline reuses this exact prompt; wording that
    # already names an injection would make the baseline's false-positive
    # rate meaningless.
    text = detection_prompt().lower()
    for tell in ("inject", "steer", "concept has been", "we have added"):
        assert tell not in text


def test_detection_prompt_asks_for_a_yes_or_no_verdict() -> None:
    # The detection/naming split (ADR-0015) depends on this prompt asking
    # for a verdict before any description.
    text = detection_prompt().lower()
    assert "yes" in text
    assert "no" in text


def test_detection_prompt_is_stable_across_calls() -> None:
    assert detection_prompt() == detection_prompt()


def test_naming_subtask_prompt_returns_a_nonempty_string() -> None:
    assert isinstance(naming_subtask_prompt(), str)
    assert naming_subtask_prompt().strip() != ""


def test_naming_subtask_prompt_differs_from_detection_prompt() -> None:
    assert naming_subtask_prompt() != detection_prompt()


# N/A: comparing this project's wording against Lindsey (2025)'s own prompt
# text would require transcribing that copyrighted wording into a test
# fixture, which defeats the point of writing independent phrasing in the
# first place (CLAUDE.md §2). Originality here is a human judgment call made
# once at authoring time, not something a fixture-based test can check.
@pytest.mark.skip(
    reason="N/A: verifying non-reproduction of Lindsey (2025)'s wording requires "
    "quoting that source text into the test itself, which the wording "
    "requirement exists to avoid"
)
def test_prompts_do_not_reproduce_lindsey_wording() -> None:
    pass


# --- unrelated_control_prompt: the real, checked-in file --------------------


def test_unrelated_control_prompt_loads_the_real_config_file() -> None:
    data = unrelated_control_prompt(DEFAULT_CONTROL_QUESTIONS_PATH)

    assert isinstance(data["version"], str) and data["version"].strip()
    assert len(data["questions"]) >= MIN_CONTROL_QUESTIONS
    for entry in data["questions"]:
        assert entry["expected_answer"] == "no"
        assert entry["question"].strip() != ""
        assert entry["id"].strip() != ""


def test_unrelated_control_prompt_real_file_has_unique_ids_and_questions() -> None:
    data = unrelated_control_prompt(DEFAULT_CONTROL_QUESTIONS_PATH)
    ids = [entry["id"] for entry in data["questions"]]
    questions = [entry["question"] for entry in data["questions"]]

    assert len(ids) == len(set(ids))
    assert len(questions) == len(set(questions))


def test_unrelated_control_prompt_real_file_spans_more_than_one_topic() -> None:
    # A set that accidentally narrowed to one topic would only catch a
    # narrow form of yes-bias.
    data = unrelated_control_prompt(DEFAULT_CONTROL_QUESTIONS_PATH)
    topics = {entry.get("topic") for entry in data["questions"]}

    assert len(topics) > 1


# --- unrelated_control_prompt: validation against malformed fixtures --------


def _valid_question_set() -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "questions": [
            {"id": f"q{i}", "question": f"Is {i} greater than {i + 100}?", "expected_answer": "no"}
            for i in range(MIN_CONTROL_QUESTIONS)
        ],
    }


def _write_yaml(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "control_questions.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_unrelated_control_prompt_accepts_a_well_formed_fixture(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _valid_question_set())

    data = unrelated_control_prompt(path)

    assert len(data["questions"]) == MIN_CONTROL_QUESTIONS


def test_unrelated_control_prompt_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="top-level mapping"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_missing_version(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    del fixture["version"]
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="version"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_blank_version(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    fixture["version"] = "   "
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="version"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_missing_questions_key(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    del fixture["questions"]
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="questions"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_too_few_questions(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    fixture["questions"] = fixture["questions"][:3]
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match=str(MIN_CONTROL_QUESTIONS)):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_non_mapping_question_entry(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    fixture["questions"][0] = "not a mapping"
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="index 0"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_question_missing_a_required_field(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    del fixture["questions"][0]["expected_answer"]
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="expected_answer"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_duplicate_id(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    fixture["questions"][1]["id"] = fixture["questions"][0]["id"]
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="duplicate question id"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_duplicate_question(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    fixture["questions"][1]["question"] = fixture["questions"][0]["question"]
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="duplicate question:"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_rejects_a_non_negative_expected_answer(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    fixture["questions"][0]["expected_answer"] = "yes"
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match="default-negative"):
        unrelated_control_prompt(path)


def test_unrelated_control_prompt_error_names_the_offending_question(tmp_path: Path) -> None:
    fixture = _valid_question_set()
    fixture["questions"][0]["expected_answer"] = "maybe"
    path = _write_yaml(tmp_path, fixture)

    with pytest.raises(ValueError, match=fixture["questions"][0]["id"]):
        unrelated_control_prompt(path)
