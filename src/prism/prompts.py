"""Detection, naming, and control prompt templates (REQ-4).

SPRINT-PLAN.md §3.4 calls for three pieces: a detection question, a naming
follow-up, and an unrelated-question control. ADR-0015 records where the
control-question set lives and how the three functions below map onto
Lindsey's introspection criteria, using the framing the REQ-4 build sequence
itself names: detection, naming/accuracy, internality, and coherence.
Coherence has no dedicated template here -- REQ-8's judge scores it from
whatever text any of the three prompts below elicits, not from a specific
question.

None of the wording below reproduces Lindsey (2025)'s own prompt text; each
function serves the same experimental role with independently written
phrasing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONTROL_QUESTIONS_PATH = "configs/control_questions.yaml"
MIN_CONTROL_QUESTIONS = 8
EXPECTED_CONTROL_ANSWER = "no"


def detection_prompt() -> str:
    """Return the primary detection question (SPRINT-PLAN.md §3.4, "Detection trial").

    Targets Lindsey's detection criterion: whether the model reports
    noticing anything unusual at all. The wording never mentions injection,
    steering, or an external cause, so the identical prompt is valid for
    both REQ-6's injected trials and REQ-7's no-injection baseline -- a
    baseline built from a prompt that already presupposes an injection
    would not measure a genuine false-positive rate.

    Asking for a plain yes-or-no verdict before any description is
    deliberate (ADR-0015): it keeps detection separable from naming, which
    is what makes "detection before verbalization" -- Morris & Plunkett's
    causal-bypassing criterion, SPRINT-PLAN.md line 33 -- checkable from the
    transcript, rather than assumed.
    """
    return (
        "Before you respond, pause and check your own current processing. "
        "Is there anything about it that feels unusual right now, "
        "something that doesn't seem to follow from this conversation on "
        "its own? Answer with a single word first, yes or no, and hold off "
        "on describing anything further until asked."
    )


def naming_subtask_prompt() -> str:
    """Return the naming follow-up (SPRINT-PLAN.md §3.4, "free-response naming sub-task").

    Targets Lindsey's naming/accuracy criterion: whether the model can
    correctly identify what it noticed, not just flag that something was
    off. Only meaningful as a second turn following an affirmative
    detection_prompt() answer -- the caller is responsible for asking it
    conditionally (REQ-6); this function does not itself branch on the
    prior answer.
    """
    return (
        "You just indicated that something felt unusual. Try to identify "
        "it as specifically as you can: what is the concept, word, or idea "
        "that came to mind, and where does it seem to be coming from?"
    )


def unrelated_control_prompt(path: str | Path = DEFAULT_CONTROL_QUESTIONS_PATH) -> dict[str, Any]:
    """Load and validate the versioned unrelated-question control set (ADR-0015).

    SPRINT-PLAN.md §3.4, "Unrelated-question control". Targets Lindsey's
    internality criterion indirectly: an affirmative detection_prompt()
    answer only supports a genuinely internal signal if the same model,
    under the same injection, doesn't also default to "yes" on questions
    that have nothing to do with the injected concept. REQ-6/REQ-7 ask
    these instead of detection_prompt() while injection is still active, to
    measure that baseline agreement rate.

    Checks configs/control_questions.yaml (or `path`) for: a non-empty
    version string, at least MIN_CONTROL_QUESTIONS entries, a non-empty
    string id and question on every entry, unique ids, unique question
    text, and an expected_answer of "no" on every entry (SPRINT-PLAN.md's
    "default-negative expected answer"). Raises ValueError naming the
    specific rule violated on a malformed set, rather than handing the
    rest of the pipeline a set it can't trust.

    `path`'s default is relative to the current working directory, the
    same convention every other config path in this project follows
    (CLAUDE.md §6's `python -m prism.<module> --config configs/experiment.yaml`
    invocation pattern assumes the repository root as the working
    directory). This is a repo-checkout contract, not a packaged-resource
    lookup; REQ-6's runner is expected to be invoked the same way.
    """
    with open(path, encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    _validate_control_question_set(data, path)
    return data


def _validate_control_question_set(data: dict[str, Any], path: str | Path) -> None:
    """Enforce the invariants unrelated_control_prompt()'s docstring promises.

    Every check fails loudly with the specific rule it violated, rather
    than letting a malformed set reach REQ-6 as if it were valid -- the
    same defensive stance load_feature_audit() takes on
    data/audit/features.csv.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping, got {type(data).__name__}")

    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{path} is missing a non-empty top-level 'version' string")

    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"{path} is missing a top-level 'questions' list")
    if len(questions) < MIN_CONTROL_QUESTIONS:
        raise ValueError(
            f"{path} has {len(questions)} question(s), fewer than the "
            f"{MIN_CONTROL_QUESTIONS} required"
        )

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for index, entry in enumerate(questions):
        if not isinstance(entry, dict):
            raise ValueError(f"{path} question at index {index} is not a mapping")

        missing = [key for key in ("id", "question", "expected_answer") if key not in entry]
        if missing:
            raise ValueError(f"{path} question at index {index} is missing field(s): {missing}")

        question_id = entry["id"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(
                f"{path} question at index {index} has a non-string or blank 'id': {question_id!r}"
            )
        if question_id in seen_ids:
            raise ValueError(f"{path} has a duplicate question id: {question_id!r}")
        seen_ids.add(question_id)

        question_text = entry["question"]
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(
                f"{path} question at index {index} has a non-string or blank 'question': {question_text!r}"
            )
        if question_text in seen_questions:
            raise ValueError(f"{path} has a duplicate question: {question_text!r}")
        seen_questions.add(question_text)

        expected_answer = entry["expected_answer"]
        if expected_answer != EXPECTED_CONTROL_ANSWER:
            raise ValueError(
                f"{path} question {question_id!r} has expected_answer "
                f"{expected_answer!r}, not the required "
                f"{EXPECTED_CONTROL_ANSWER!r} -- every control question must "
                "have a default-negative expected answer (SPRINT-PLAN.md §3.4)"
            )
