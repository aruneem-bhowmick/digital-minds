"""Tests for prism.judge -- LLM-judge scoring and the human-validation gate (REQ-8)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from prism.judge import (
    _CONTROL_SCHEMA,
    _DETECTION_SCHEMA,
    _build_control_prompt,
    _build_grounding_text,
    _build_two_turn_prompt,
    _count_report_trials,
    _read_all_records,
    _write_all_records,
    collect_concept_grounding,
    load_concept_grounding,
    save_concept_grounding,
    save_scoring_provenance,
    score_all_pending,
    score_trial,
    validate_judge_subsample,
    write_validation_flag,
)


# --- a fake judge client, faithful enough to exercise real control flow -----


class _FakeJudgeClient:
    """Returns canned responses in call order and records every request made,
    so tests can assert on both the parsed result and what was actually sent.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            text = item
            stop_reason = "end_turn"
        else:
            payload = dict(item)
            stop_reason = payload.pop("_stop_reason", "end_turn")
            text = json.dumps(payload)
        return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


def _detection_trial(*, feature_id: int = 7, affirmative: bool = True, naming: dict | None = "default") -> dict:
    if naming == "default":
        naming = {"prompt": "What is it?", "response": "It felt like oceans."} if affirmative else None
    return {
        "trial_id": "detection::feature7::layer4::strength2::seed0",
        "feature_id": feature_id,
        "prompt_type": "detection",
        "model_response": {
            "detection": {"prompt": "Notice anything?", "response": "Yes, something feels off."},
            "affirmative": affirmative,
            "naming": naming,
        },
        "judge_scores": None,
    }


def _baseline_trial(*, feature_id: int = 7) -> dict:
    record = _detection_trial(feature_id=feature_id)
    record["trial_id"] = "baseline::feature7::layer4::seed0"
    record["prompt_type"] = "baseline"
    return record


def _control_trial(*, question_id: str = "q1") -> dict:
    return {
        "trial_id": f"control::feature7::layer4::strength2::seed0::q-{question_id}",
        "feature_id": 7,
        "prompt_type": "control",
        "model_response": {
            "question_id": question_id,
            "question": "Is the sky usually orange at noon?",
            "expected_answer": "no",
            "response": "No.",
        },
        "judge_scores": None,
    }


_DETECTION_SCORE = {
    "detected": True,
    "concept_identified": True,
    "concept_confidence": "high",
    "identified_before_verbalizing": True,
    "coherent": True,
    "reasoning": "matches the reference evidence",
}
_CONTROL_SCORE = {"affirmative": False, "coherent": True, "reasoning": "answered no as expected"}


# --- _build_grounding_text ---------------------------------------------------


def test_build_grounding_text_renders_snippets_ranked() -> None:
    text = _build_grounding_text([{"activation": 0.9, "snippet": "the vast ocean"}, {"activation": 0.5, "snippet": "sailing at sea"}])

    assert "the vast ocean" in text
    assert "0.900" in text


def test_build_grounding_text_reports_absence_explicitly_for_empty_snippets() -> None:
    text = _build_grounding_text([])

    assert "No strongly activating examples" in text


# --- _build_two_turn_prompt / _build_control_prompt --------------------------


def test_build_two_turn_prompt_uses_injected_framing_for_detection() -> None:
    prompt = _build_two_turn_prompt(_detection_trial(), [])

    assert "deliberately pushed a concept" in prompt
    assert "It felt like oceans." in prompt


def test_build_two_turn_prompt_uses_no_injection_framing_for_baseline() -> None:
    prompt = _build_two_turn_prompt(_baseline_trial(), [])

    assert "No concept was actually injected" in prompt


def test_build_two_turn_prompt_states_no_naming_turn_on_negative_detection() -> None:
    prompt = _build_two_turn_prompt(_detection_trial(affirmative=False, naming=None), [])

    assert "no follow-up naming question was asked" in prompt


def test_build_two_turn_prompt_includes_grounding_snippets() -> None:
    prompt = _build_two_turn_prompt(_detection_trial(), [{"activation": 0.7, "snippet": "deep blue water"}])

    assert "deep blue water" in prompt


def test_build_control_prompt_has_no_naming_or_detection_framing() -> None:
    prompt = _build_control_prompt(_control_trial())

    assert "Is the sky usually orange at noon?" in prompt
    assert "deliberately pushed" not in prompt


# --- score_trial: schema selection, grounding lookup, error handling --------


def test_score_trial_detection_uses_the_detection_schema() -> None:
    client = _FakeJudgeClient([_DETECTION_SCORE])

    result = score_trial(_detection_trial(), client)

    assert result == _DETECTION_SCORE
    assert client.calls[0]["output_config"]["format"]["schema"] is _DETECTION_SCHEMA


def test_score_trial_baseline_also_uses_the_detection_schema() -> None:
    # Same two-turn shape as detection -- ADR-0018 splits the schema by turn
    # structure, not by whether an injection actually happened.
    client = _FakeJudgeClient([_DETECTION_SCORE])

    result = score_trial(_baseline_trial(), client)

    assert result == _DETECTION_SCORE
    assert client.calls[0]["output_config"]["format"]["schema"] is _DETECTION_SCHEMA


def test_score_trial_control_uses_the_control_schema() -> None:
    client = _FakeJudgeClient([_CONTROL_SCORE])

    result = score_trial(_control_trial(), client)

    assert result == _CONTROL_SCORE
    assert client.calls[0]["output_config"]["format"]["schema"] is _CONTROL_SCHEMA


def test_score_trial_rejects_an_unrecognized_prompt_type() -> None:
    client = _FakeJudgeClient([])
    record = _control_trial()
    record["prompt_type"] = "something-else"

    with pytest.raises(ValueError, match="unrecognized prompt_type"):
        score_trial(record, client)


def test_score_trial_looks_up_grounding_by_string_feature_id() -> None:
    client = _FakeJudgeClient([_DETECTION_SCORE])
    grounding = {"7": [{"activation": 0.6, "snippet": "coral reefs"}]}

    score_trial(_detection_trial(feature_id=7), client, grounding)

    sent_prompt = client.calls[0]["messages"][0]["content"]
    assert "coral reefs" in sent_prompt


def test_score_trial_treats_a_missing_grounding_entry_as_no_evidence_found() -> None:
    client = _FakeJudgeClient([_DETECTION_SCORE])
    grounding = {"999": [{"activation": 0.6, "snippet": "irrelevant feature"}]}  # not feature 7

    score_trial(_detection_trial(feature_id=7), client, grounding)

    sent_prompt = client.calls[0]["messages"][0]["content"]
    assert "No strongly activating examples" in sent_prompt


def test_score_trial_treats_a_missing_grounding_argument_the_same_way() -> None:
    client = _FakeJudgeClient([_DETECTION_SCORE])

    score_trial(_detection_trial(), client, grounding=None)

    sent_prompt = client.calls[0]["messages"][0]["content"]
    assert "No strongly activating examples" in sent_prompt


def test_score_trial_raises_on_a_judge_refusal() -> None:
    client = _FakeJudgeClient([{**_DETECTION_SCORE, "_stop_reason": "refusal"}])

    with pytest.raises(RuntimeError, match="refused"):
        score_trial(_detection_trial(), client)


def test_score_trial_raises_on_a_response_missing_required_fields() -> None:
    client = _FakeJudgeClient(['{"detected": true}'])

    with pytest.raises(ValueError, match="missing required field"):
        score_trial(_detection_trial(), client)


def test_score_trial_raises_a_clear_error_on_max_tokens_truncation() -> None:
    client = _FakeJudgeClient([{**_CONTROL_SCORE, "_stop_reason": "max_tokens"}])

    with pytest.raises(ValueError, match="truncated"):
        score_trial(_control_trial(), client)


def test_score_trial_wraps_a_json_decode_failure_with_context() -> None:
    # Not a real refusal or truncation -- just genuinely malformed output --
    # so this should surface as a debuggable ValueError, not a bare
    # json.JSONDecodeError with no trial context.
    client = _FakeJudgeClient(["not valid json {{{"])

    with pytest.raises(ValueError, match="not valid JSON"):
        score_trial(_control_trial(), client)


def test_score_trial_control_trial_never_has_a_naming_field_to_grade() -> None:
    # N/A: control trials never carry a "naming" key at all (ADR-0017's
    # record shape for prompt_type="control" has no detection/naming turn),
    # so there is no "control trial with a naming turn" case to construct --
    # confirmed by the fact _build_control_prompt() only ever reads
    # question/response, never model_response["naming"].
    record = _control_trial()
    assert "naming" not in record["model_response"]
    assert "detection" not in record["model_response"]


# --- score_all_pending: resumability -----------------------------------------


def test_score_all_pending_only_scores_records_missing_judge_scores(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    already_scored = {**_control_trial(question_id="a"), "judge_scores": _CONTROL_SCORE}
    pending = _control_trial(question_id="b")
    _write_all_records(path, [already_scored, pending])
    client = _FakeJudgeClient([_CONTROL_SCORE])

    result = score_all_pending(path, client)

    assert result == {"scored": 1, "skipped": 1, "refused": 0}
    assert len(client.calls) == 1


def test_score_all_pending_second_run_scores_nothing(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    _write_all_records(path, [_control_trial()])

    score_all_pending(path, _FakeJudgeClient([_CONTROL_SCORE]))
    result = score_all_pending(path, _FakeJudgeClient([]))  # empty queue: any call would raise IndexError

    assert result == {"scored": 0, "skipped": 1, "refused": 0}


def test_score_all_pending_persists_progress_before_a_later_failure(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    first = _control_trial(question_id="a")
    second = _control_trial(question_id="b")
    _write_all_records(path, [first, second])
    client = _FakeJudgeClient([_CONTROL_SCORE, RuntimeError("simulated API failure")])

    with pytest.raises(RuntimeError, match="simulated API failure"):
        score_all_pending(path, client)

    records = _read_all_records(path)
    scored_ids = {r["trial_id"]: r["judge_scores"] is not None for r in records}
    assert scored_ids[first["trial_id"]] is True
    assert scored_ids[second["trial_id"]] is False


def test_score_all_pending_excludes_a_refused_trial_and_keeps_going(tmp_path) -> None:
    # A judge refusal is a content-based signal about the trial, not a bug --
    # unlike a genuine failure (the previous test), it must not take the rest
    # of the batch down with it.
    path = tmp_path / "trials.jsonl"
    refused = _control_trial(question_id="a")
    scorable = _control_trial(question_id="b")
    _write_all_records(path, [refused, scorable])
    client = _FakeJudgeClient([{**_CONTROL_SCORE, "_stop_reason": "refusal"}, _CONTROL_SCORE])

    result = score_all_pending(path, client)

    assert result == {"scored": 1, "skipped": 0, "refused": 1}
    records = {r["trial_id"]: r for r in _read_all_records(path)}
    assert records[refused["trial_id"]]["judge_scores"] is None
    assert records[refused["trial_id"]]["excluded"] is True
    assert "refused" in records[refused["trial_id"]]["exclusion_reason"]
    assert records[scorable["trial_id"]]["judge_scores"] == _CONTROL_SCORE
    assert records[scorable["trial_id"]].get("excluded", False) is False


def test_score_all_pending_clears_stale_exclusion_on_a_successful_retry(tmp_path) -> None:
    # A trial refused on a prior run and excluded then, but the judge no
    # longer refuses it on this run -- the stale exclusion must not survive
    # alongside a real score, or downstream code filtering on `excluded`
    # would silently drop a validly-scored trial.
    path = tmp_path / "trials.jsonl"
    previously_refused = {
        **_control_trial(),
        "excluded": True,
        "exclusion_reason": "judge refused to grade trial '...' (category='bio'): stale reason",
    }
    _write_all_records(path, [previously_refused])
    client = _FakeJudgeClient([_CONTROL_SCORE])

    score_all_pending(path, client)

    record = _read_all_records(path)[0]
    assert record["judge_scores"] == _CONTROL_SCORE
    assert record["excluded"] is False
    assert record["exclusion_reason"] is None


def test_score_all_pending_flushes_progress_on_finally_even_mid_batch(tmp_path) -> None:
    # since_last_write hasn't reached the batch-write threshold when the
    # second trial raises -- the `finally` block, not the periodic write,
    # is what must persist the first trial's score here.
    path = tmp_path / "trials.jsonl"
    first = _control_trial(question_id="a")
    second = _control_trial(question_id="b")
    _write_all_records(path, [first, second])
    client = _FakeJudgeClient([_CONTROL_SCORE, RuntimeError("simulated mid-batch failure")])

    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        score_all_pending(path, client)

    records = {r["trial_id"]: r for r in _read_all_records(path)}
    assert records[first["trial_id"]]["judge_scores"] == _CONTROL_SCORE


def test_score_all_pending_never_duplicates_rows(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    _write_all_records(path, [_control_trial(question_id="a"), _control_trial(question_id="b")])

    score_all_pending(path, _FakeJudgeClient([_CONTROL_SCORE, _CONTROL_SCORE]))

    assert len(_read_all_records(path)) == 2


# --- _read_all_records / _write_all_records ----------------------------------


def test_read_all_records_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert _read_all_records(path) == [{"a": 1}, {"a": 2}]


def test_write_then_read_round_trips(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    records = [{"trial_id": "a"}, {"trial_id": "b"}]

    _write_all_records(path, records)

    assert _read_all_records(path) == records


# --- concept grounding: collect / save / load --------------------------------


def test_collect_concept_grounding_keys_by_string_feature_id(monkeypatch) -> None:
    import prism.audit_build as audit_build
    import prism.models as models

    monkeypatch.setattr(audit_build, "_load_corpus", lambda *a, **k: ([], {"corpus_dataset": "test/corpus"}))
    monkeypatch.setattr(
        models, "top_activating_snippets", lambda *a, **k: {7: [{"activation": 0.4, "snippet": "s"}], 9: []}
    )
    monkeypatch.setattr("prism.judge._git_commit", lambda: "cafef00d")

    config = {
        "model": {"name": "m", "checkpoint_revision": "r"},
        "sae": {"checkpoint_repo": "repo", "checkpoint_revision": "r2", "checkpoint_sha256": "s", "hook_name": "h"},
    }
    grounding, provenance = collect_concept_grounding(config, loaded=object(), feature_ids=[7, 9], k=3)

    assert grounding == {"7": [{"activation": 0.4, "snippet": "s"}], "9": []}
    assert provenance["feature_ids"] == [7, 9]
    assert provenance["k"] == 3
    assert provenance["git_commit"] == "cafef00d"
    assert provenance["corpus_dataset"] == "test/corpus"


def test_save_then_load_concept_grounding_round_trips(tmp_path) -> None:
    path = tmp_path / "grounding.json"
    grounding = {"7": [{"activation": 0.4, "snippet": "s"}]}

    save_concept_grounding(grounding, {"k": 5}, path)

    assert load_concept_grounding(path) == grounding


# --- save_scoring_provenance: judge-run reproducibility ----------------------


def test_save_scoring_provenance_records_model_commit_and_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("prism.judge._git_commit", lambda: "cafef00d")
    path = tmp_path / "provenance.json"

    save_scoring_provenance(
        "claude-opus-4-8",
        {"scored": 5, "skipped": 2, "refused": 1},
        trials_path="data/trials/trials.jsonl",
        output_path=path,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["judge_model"] == "claude-opus-4-8"
    assert record["git_commit"] == "cafef00d"
    assert record["trials_path"] == "data/trials/trials.jsonl"
    assert record["scored"] == 5
    assert record["skipped"] == 2
    assert record["refused"] == 1
    assert record["timestamp"]


# --- _count_report_trials: source of truth for --confirm-validated -----------


def test_count_report_trials_counts_per_trial_headers(tmp_path) -> None:
    path = tmp_path / "report.md"
    path.write_text(
        "# Judge validation sample (2 trials)\n\n## trial-a\n\ncontent\n\n## trial-b\n\ncontent\n",
        encoding="utf-8",
    )

    assert _count_report_trials(path) == 2


# --- validate_judge_subsample -------------------------------------------------


def test_validate_judge_subsample_raises_when_nothing_is_scored(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    _write_all_records(path, [_control_trial()])

    with pytest.raises(ValueError, match="no scored trials"):
        validate_judge_subsample(5, path, output_path=None)


def test_validate_judge_subsample_caps_at_available_scored_trials(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    scored = {**_control_trial(), "judge_scores": _CONTROL_SCORE}
    _write_all_records(path, [scored])

    sample = validate_judge_subsample(15, path, output_path=None)

    assert len(sample) == 1


def test_validate_judge_subsample_spans_multiple_prompt_types(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    records = [
        {**_detection_trial(), "judge_scores": _DETECTION_SCORE},
        {**_baseline_trial(), "judge_scores": _DETECTION_SCORE},
        {**_control_trial(), "judge_scores": _CONTROL_SCORE},
    ]
    _write_all_records(path, records)

    sample = validate_judge_subsample(3, path, output_path=None)

    assert {r["prompt_type"] for r in sample} == {"detection", "baseline", "control"}


def test_validate_judge_subsample_never_exceeds_n_even_with_fewer_types_than_n_allows(tmp_path) -> None:
    # The old `max(1, n // len(types))` forced at least one record per type
    # regardless of n, so n=1 against 3 present prompt_types returned 3
    # records instead of 1. divmod-based allocation must not repeat that.
    path = tmp_path / "trials.jsonl"
    records = [
        {**_detection_trial(), "judge_scores": _DETECTION_SCORE},
        {**_baseline_trial(), "judge_scores": _DETECTION_SCORE},
        {**_control_trial(), "judge_scores": _CONTROL_SCORE},
    ]
    _write_all_records(path, records)

    sample = validate_judge_subsample(1, path, output_path=None)

    assert len(sample) == 1


def test_validate_judge_subsample_is_deterministic_under_a_fixed_seed(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    records = [
        {**_control_trial(question_id=str(i)), "judge_scores": _CONTROL_SCORE} for i in range(10)
    ]
    _write_all_records(path, records)

    first = validate_judge_subsample(4, path, seed=0, output_path=None)
    second = validate_judge_subsample(4, path, seed=0, output_path=None)

    assert [r["trial_id"] for r in first] == [r["trial_id"] for r in second]


def test_validate_judge_subsample_writes_a_report_file(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    report_path = tmp_path / "report.md"
    scored = {**_control_trial(), "judge_scores": _CONTROL_SCORE}
    _write_all_records(trials_path, [scored])

    validate_judge_subsample(5, trials_path, output_path=report_path)

    content = report_path.read_text(encoding="utf-8")
    assert scored["trial_id"] in content
    assert "Agreement notes" in content


# --- write_validation_flag: the human-confirmation gate ----------------------


def test_write_validation_flag_rejects_an_empty_reviewer_note(tmp_path) -> None:
    with pytest.raises(ValueError, match="must describe what was actually reviewed"):
        write_validation_flag("   ", sample_size=15, output_path=tmp_path / "flag")


def test_write_validation_flag_records_the_reviewer_note(tmp_path) -> None:
    path = tmp_path / "judge_validated.flag"

    write_validation_flag("checked 15/15, judge matches my read", sample_size=15, output_path=path)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["reviewer_note"] == "checked 15/15, judge matches my read"
    assert record["sample_size"] == 15
    assert record["reviewed_at"]


def test_no_scoring_or_validation_function_writes_the_flag_as_a_side_effect(tmp_path) -> None:
    # score_all_pending() and validate_judge_subsample() must never create
    # judge_validated.flag themselves -- only write_validation_flag(), on an
    # explicit call, does that.
    path = tmp_path / "trials.jsonl"
    _write_all_records(path, [_control_trial()])
    score_all_pending(path, _FakeJudgeClient([_CONTROL_SCORE]))
    validate_judge_subsample(1, path, output_path=None)

    assert not (tmp_path / "judge_validated.flag").exists()


# --- integration: the real judge API, when credentials are available --------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY")
def test_score_trial_against_the_real_judge_api() -> None:
    import anthropic

    client = anthropic.Anthropic()
    record = _control_trial()

    result = score_trial(record, client)

    assert isinstance(result["affirmative"], bool)
    assert isinstance(result["coherent"], bool)
    assert isinstance(result["reasoning"], str) and result["reasoning"]
