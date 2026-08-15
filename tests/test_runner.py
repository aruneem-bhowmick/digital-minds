"""Tests for prism.runner -- the systematic, baseline, and control trial batches (REQ-6, REQ-7)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from prism.runner import (
    DEFAULT_TRIALS_PATH,
    _append_record,
    _build_record,
    _load_existing_trial_ids,
    _trial_id,
    is_affirmative,
    run_baseline_trials,
    run_control_trials,
    run_control_trial,
    run_systematic_trials,
    run_two_turn_trial,
)

CONFIG_PATH = "configs/experiment.yaml"


# --- a fake model, faithful enough to exercise real control flow ------------
#
# Real generation isn't needed to test the runner's orchestration (looping,
# resumability, trial IDs, record shape) -- inject.py's own hook mechanics
# are already covered in test_inject.py. This fake round-trips text through
# a shared word-level vocabulary so to_tokens()/to_string() behave
# consistently with each other and with real token-ID concatenation, without
# needing a real tokenizer or a real forward pass.


class _FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._vocab: dict[str, int] = {"<bos>": 0}
        self._rvocab: dict[int, str] = {0: "<bos>"}
        self.cfg = SimpleNamespace(n_layers=6)
        self.hook_dict = dict.fromkeys(["blocks.4.hook_resid_pre"])

    def _ids(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                idx = len(self._vocab)
                self._vocab[word] = idx
                self._rvocab[idx] = word
            ids.append(self._vocab[word])
        return ids

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        ids = ([0] if prepend_bos else []) + self._ids(text)
        return torch.tensor([ids])

    def to_string(self, tensor: torch.Tensor) -> str:
        ids = tensor.reshape(-1).tolist()
        return " ".join(self._rvocab.get(i, "?") for i in ids)

    def generate(self, tokens: torch.Tensor, *, max_new_tokens: int, do_sample: bool, temperature: float, verbose: bool) -> torch.Tensor:
        del max_new_tokens, do_sample, temperature, verbose
        response = self._responses.pop(0)
        new_tokens = torch.tensor([self._ids(response)])
        return torch.cat([tokens, new_tokens], dim=1)

    @contextmanager
    def hooks(self, fwd_hooks: list):
        del fwd_hooks
        yield


def _fake_loaded(responses: list[str], n_features: int = 20, d_model: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        model=_FakeModel(responses),
        sae=SimpleNamespace(W_dec=torch.zeros(n_features, d_model)),
        hook_name="blocks.4.hook_resid_pre",
    )


def _base_config(**overrides) -> dict:
    config = {
        "model": {"name": "EleutherAI/pythia-70m-deduped", "checkpoint_revision": "deadbeef"},
        "sae": {
            "checkpoint_repo": "ghidav/pythia-70m-deduped-sae",
            "checkpoint_revision": "cafef00d",
            "checkpoint_sha256": "abc123",
        },
        "injection": {"strengths": [1, 2]},
        "sampling": {"seeds": [0, 1]},
        "generation": {"temperature_systematic": 1},
    }
    config.update(overrides)
    return config


def _sampled_features(feature_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"feature_id": feature_ids})


# --- is_affirmative -----------------------------------------------------


@pytest.mark.parametrize(
    "response",
    ["Yes.", "yes, something feels off", "Yes, I notice something unusual.", '"Yes" is my answer.'],
)
def test_is_affirmative_true_for_yes_first_word(response: str) -> None:
    assert is_affirmative(response) is True


@pytest.mark.parametrize(
    "response",
    ["No.", "no, nothing unusual", "I don't know.", "yesterday was an ordinary day", "", "   "],
)
def test_is_affirmative_false_for_non_yes_responses(response: str) -> None:
    assert is_affirmative(response) is False


# --- _trial_id: determinism, layer-awareness, cross-type collisions ------


def test_trial_id_is_deterministic_for_the_same_condition() -> None:
    first = _trial_id("detection", feature_id=5, layer=4, strength=2.0, seed=0)
    second = _trial_id("detection", feature_id=5, layer=4, strength=2.0, seed=0)

    assert first == second


def test_trial_id_differs_by_layer() -> None:
    # A later REQ-10 layer resolution must not collide with fallback-layer
    # trials already logged -- a new layer needs its own IDs so it reruns
    # instead of being silently skipped as "already done."
    layer4 = _trial_id("detection", feature_id=5, layer=4, strength=2.0, seed=0)
    layer3 = _trial_id("detection", feature_id=5, layer=3, strength=2.0, seed=0)

    assert layer4 != layer3


def test_trial_id_differs_across_prompt_types_for_the_same_condition() -> None:
    detection = _trial_id("detection", feature_id=5, layer=4, strength=2.0, seed=0)
    baseline = _trial_id("baseline", feature_id=5, layer=4, seed=0)
    control = _trial_id("control", feature_id=5, layer=4, strength=2.0, seed=0, question_id="ocean_size")

    assert len({detection, baseline, control}) == 3


def test_trial_id_differs_by_control_question() -> None:
    first = _trial_id("control", feature_id=5, layer=4, strength=2.0, seed=0, question_id="ocean_size")
    second = _trial_id("control", feature_id=5, layer=4, strength=2.0, seed=0, question_id="seven_even")

    assert first != second


def test_trial_id_strength_formatting_is_stable_across_int_and_float_input() -> None:
    from_int = _trial_id("detection", feature_id=5, layer=4, strength=2, seed=0)
    from_float = _trial_id("detection", feature_id=5, layer=4, strength=2.0, seed=0)

    assert from_int == from_float


# --- resumable JSONL log: append and dedup -------------------------------


def test_load_existing_trial_ids_on_a_missing_file_returns_empty_set(tmp_path) -> None:
    assert _load_existing_trial_ids(tmp_path / "does_not_exist.jsonl") == set()


def test_append_record_then_load_round_trips_the_trial_id(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    _append_record(path, {"trial_id": "a"})
    _append_record(path, {"trial_id": "b"})

    assert _load_existing_trial_ids(path) == {"a", "b"}


def test_append_record_never_rewrites_earlier_lines(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    _append_record(path, {"trial_id": "a", "note": "first"})
    _append_record(path, {"trial_id": "b", "note": "second"})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["note"] == "first"
    assert json.loads(lines[1])["note"] == "second"


# --- _build_record: schema completeness ----------------------------------


def test_build_record_has_every_adr_0005_field_plus_provenance() -> None:
    config = _base_config()
    record = _build_record(
        trial_id="t1",
        feature_id=5,
        layer=4,
        layer_source="adr-0009-fallback",
        strength=2.0,
        prompt_type="detection",
        seed=0,
        temperature=1.0,
        model_response={"detection": {"prompt": "p", "response": "r"}, "affirmative": False, "naming": None},
        config=config,
    )

    adr_0005_fields = {
        "trial_id",
        "feature_id",
        "layer",
        "strength",
        "prompt_type",
        "seed",
        "temperature",
        "model_response",
        "judge_scores",
        "timestamp",
        "git_commit",
        "excluded",
        "exclusion_reason",
    }
    assert adr_0005_fields <= record.keys()
    assert record["judge_scores"] is None
    assert record["excluded"] is False
    assert record["exclusion_reason"] is None
    assert record["model_checkpoint_revision"] == config["model"]["checkpoint_revision"]
    assert record["sae_checkpoint_sha256"] == config["sae"]["checkpoint_sha256"]
    assert record["git_commit"]
    assert record["timestamp"]


def test_build_record_strength_can_be_none_for_a_baseline_trial() -> None:
    record = _build_record(
        trial_id="t1",
        feature_id=5,
        layer=4,
        layer_source="adr-0009-fallback",
        strength=None,
        prompt_type="baseline",
        seed=0,
        temperature=1.0,
        model_response={},
        config=_base_config(),
    )

    assert record["strength"] is None


# --- run_two_turn_trial: the detection/naming control flow --------------


def test_run_two_turn_trial_asks_naming_only_on_an_affirmative_answer() -> None:
    loaded = _fake_loaded(responses=["Yes something feels off", "It is the concept of oceans"])

    result = run_two_turn_trial(loaded, lambda pos: [], seed=0, temperature=1.0, max_new_tokens=10)

    assert result["affirmative"] is True
    assert result["naming"] is not None
    assert result["naming"]["response"] == "It is the concept of oceans"


def test_run_two_turn_trial_skips_naming_on_a_negative_answer() -> None:
    loaded = _fake_loaded(responses=["No nothing unusual"])

    result = run_two_turn_trial(loaded, lambda pos: [], seed=0, temperature=1.0, max_new_tokens=10)

    assert result["affirmative"] is False
    assert result["naming"] is None
    # Only one generate() call should have happened -- the response queue
    # would raise IndexError on a second pop() if a naming turn ran anyway.
    assert loaded.model._responses == []


def test_run_two_turn_trial_builds_the_naming_continuation_from_token_ids(monkeypatch) -> None:
    # The continuation must be built by concatenating the detection turn's
    # actual output tokens with the naming prompt's tokens, never by
    # re-tokenizing pasted-together text -- retokenizing risks the join
    # landing on different token boundaries than the original generation
    # used, which would silently misalign token_start_pos on the second call.
    loaded = _fake_loaded(responses=["Yes I notice it", "It is oceans"])
    seen_lengths = []
    real_generate = loaded.model.generate

    def spying_generate(tokens, **kwargs):
        seen_lengths.append(tokens.shape[1])
        return real_generate(tokens, **kwargs)

    monkeypatch.setattr(loaded.model, "generate", spying_generate)

    from prism.prompts import naming_subtask_prompt

    run_two_turn_trial(loaded, lambda pos: [], seed=0, temperature=1.0, max_new_tokens=10)

    assert len(seen_lengths) == 2
    detection_call_len, naming_call_len = seen_lengths
    # detection_call_len tokens + the "Yes I notice it" response tokens (4
    # words) + the naming prompt's own tokens (no BOS) must equal exactly
    # what the second call received -- proving the continuation is the
    # first call's real output tokens plus the naming prompt, not a
    # re-tokenized string that could land on different boundaries.
    naming_prompt_len = loaded.model.to_tokens("\n\n" + naming_subtask_prompt(), prepend_bos=False).shape[1]
    assert naming_call_len == detection_call_len + 4 + naming_prompt_len


# --- run_control_trial: single turn, no naming ---------------------------


def test_run_control_trial_makes_a_single_generation() -> None:
    loaded = _fake_loaded(responses=["No"])
    question = {"id": "ocean_size", "question": "Is the Pacific Ocean smaller than Lake Michigan?", "expected_answer": "no"}

    result = run_control_trial(loaded, [], question, seed=0, temperature=1.0, max_new_tokens=10)

    assert result["question_id"] == "ocean_size"
    assert result["response"] == "No"
    assert result["expected_answer"] == "no"


# --- run_systematic_trials: grid coverage and resumability ---------------


def test_run_systematic_trials_produces_one_record_per_feature_strength_seed(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    config = _base_config()  # 2 strengths x 2 seeds
    sampled = _sampled_features([1, 2])  # 2 features -> 8 trials total
    loaded = _fake_loaded(responses=["No nothing unusual"] * 8)

    result = run_systematic_trials(config, loaded, sampled, trials_path=trials_path, max_new_tokens=10)

    assert result == {"run": 8, "skipped": 0}
    lines = trials_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    records = [json.loads(line) for line in lines]
    assert {r["prompt_type"] for r in records} == {"detection"}
    assert {r["feature_id"] for r in records} == {1, 2}
    assert {r["strength"] for r in records} == {1.0, 2.0}
    assert {r["seed"] for r in records} == {0, 1}
    assert len({r["trial_id"] for r in records}) == 8


def test_run_systematic_trials_resumes_without_recomputing_logged_trials(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    config = _base_config()
    sampled = _sampled_features([1, 2])

    first_pass = _fake_loaded(responses=["No nothing unusual"] * 8)
    run_systematic_trials(config, first_pass, sampled, trials_path=trials_path, max_new_tokens=10)

    # A second run gets a response queue with nothing in it -- if it tried
    # to regenerate any trial, popping an empty queue would raise.
    second_pass = _fake_loaded(responses=[])
    result = run_systematic_trials(config, second_pass, sampled, trials_path=trials_path, max_new_tokens=10)

    assert result == {"run": 0, "skipped": 8}
    lines = trials_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8  # nothing duplicated


def test_run_systematic_trials_partial_resume_only_runs_the_missing_trials(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    config = _base_config(injection={"strengths": [1]}, sampling={"seeds": [0]})  # 1 x 1 grid per feature
    first_feature_only = _sampled_features([1])
    both_features = _sampled_features([1, 2])

    first_pass = _fake_loaded(responses=["No"])
    run_systematic_trials(config, first_pass, first_feature_only, trials_path=trials_path, max_new_tokens=10)

    second_pass = _fake_loaded(responses=["No"])  # only feature 2's trial is missing
    result = run_systematic_trials(config, second_pass, both_features, trials_path=trials_path, max_new_tokens=10)

    assert result == {"run": 1, "skipped": 1}


# --- run_baseline_trials: no strength dimension, no-injection ------------


def test_run_baseline_trials_has_no_strength_dimension(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    config = _base_config()  # 2 seeds
    sampled = _sampled_features([1, 2])  # 2 features -> 4 trials (no strength sweep)
    loaded = _fake_loaded(responses=["No nothing unusual"] * 4)

    result = run_baseline_trials(config, loaded, sampled, trials_path=trials_path, max_new_tokens=10)

    assert result == {"run": 4, "skipped": 0}
    records = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").strip().splitlines()]
    assert all(r["strength"] is None for r in records)
    assert all(r["prompt_type"] == "baseline" for r in records)


def test_run_baseline_trials_and_systematic_trials_do_not_collide_in_the_same_log(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    config = _base_config()
    sampled = _sampled_features([1, 2])

    systematic_loaded = _fake_loaded(responses=["No"] * 8)
    run_systematic_trials(config, systematic_loaded, sampled, trials_path=trials_path, max_new_tokens=10)

    baseline_loaded = _fake_loaded(responses=["No"] * 4)
    result = run_baseline_trials(config, baseline_loaded, sampled, trials_path=trials_path, max_new_tokens=10)

    assert result == {"run": 4, "skipped": 0}  # none of the 4 baseline IDs collided with the 8 systematic ones
    lines = trials_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12


# --- run_control_trials: full question coverage, single seed ------------


def _control_questions_yaml(tmp_path, n: int = 8) -> Path:
    path = tmp_path / "control_questions.yaml"
    questions = [{"id": f"q{i}", "question": f"Is fact number {i} true?", "expected_answer": "no"} for i in range(n)]
    path.write_text(yaml.safe_dump({"version": "1.0.0", "questions": questions}), encoding="utf-8")
    return path


def test_run_control_trials_covers_every_configured_question(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    control_path = _control_questions_yaml(tmp_path, n=8)
    config = _base_config(injection={"strengths": [1, 2, 4, 8]}, sampling={"seeds": [0, 1]})
    sampled = _sampled_features([1, 2])  # 2 features x 4 strengths = 8 conditions, matching 8 questions
    loaded = _fake_loaded(responses=["No"] * 8)

    result = run_control_trials(
        config, loaded, sampled, control_questions_path=control_path, trials_path=trials_path, max_new_tokens=10
    )

    assert result == {"run": 8, "skipped": 0}
    records = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").strip().splitlines()]
    used_question_ids = {r["model_response"]["question_id"] for r in records}
    assert used_question_ids == {f"q{i}" for i in range(8)}


def test_run_control_trials_uses_only_the_first_configured_seed(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    control_path = _control_questions_yaml(tmp_path, n=8)
    config = _base_config(injection={"strengths": [1]}, sampling={"seeds": [7, 8, 9]})
    sampled = _sampled_features([1])
    loaded = _fake_loaded(responses=["No"])

    run_control_trials(
        config, loaded, sampled, control_questions_path=control_path, trials_path=trials_path, max_new_tokens=10
    )

    records = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").strip().splitlines()]
    assert records[0]["seed"] == 7


def test_run_control_trials_has_no_naming_field(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    control_path = _control_questions_yaml(tmp_path, n=8)
    config = _base_config(injection={"strengths": [1]}, sampling={"seeds": [0]})
    sampled = _sampled_features([1])
    loaded = _fake_loaded(responses=["No"])

    run_control_trials(
        config, loaded, sampled, control_questions_path=control_path, trials_path=trials_path, max_new_tokens=10
    )

    record = json.loads(trials_path.read_text(encoding="utf-8").strip())
    assert "naming" not in record["model_response"]
    assert "detection" not in record["model_response"]


# --- integration: real model, real SAE, real generate() ---------------------


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def loaded_pair(config):
    from prism.models import load_model_and_sae

    return load_model_and_sae(config)


@pytest.fixture()
def tiny_sampled_features():
    return pd.DataFrame({"feature_id": [10769, 9253]})  # real feature IDs from REQ-5's pilot


@pytest.mark.integration
def test_run_systematic_trials_real_model_produces_schema_complete_records(
    tmp_path, config, loaded_pair, tiny_sampled_features
) -> None:
    trials_path = tmp_path / "trials.jsonl"
    small_config = {**config, "injection": {**config["injection"], "strengths": [1]}, "sampling": {"seeds": [0]}}

    result = run_systematic_trials(
        small_config, loaded_pair, tiny_sampled_features, trials_path=trials_path, max_new_tokens=6
    )

    assert result == {"run": 2, "skipped": 0}
    records = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").strip().splitlines()]
    for record in records:
        assert record["judge_scores"] is None
        assert record["git_commit"]
        assert record["layer_source"] == "adr-0009-fallback"
        assert isinstance(record["model_response"]["affirmative"], bool)
        assert record["model_response"]["detection"]["response"]


@pytest.mark.integration
def test_run_baseline_and_control_trials_real_model(tmp_path, config, loaded_pair, tiny_sampled_features) -> None:
    trials_path = tmp_path / "trials.jsonl"
    small_config = {**config, "injection": {**config["injection"], "strengths": [1]}, "sampling": {"seeds": [0]}}

    baseline_result = run_baseline_trials(
        small_config, loaded_pair, tiny_sampled_features, trials_path=trials_path, max_new_tokens=6
    )
    control_result = run_control_trials(
        small_config, loaded_pair, tiny_sampled_features, trials_path=trials_path, max_new_tokens=6
    )

    assert baseline_result == {"run": 2, "skipped": 0}
    assert control_result == {"run": 2, "skipped": 0}
    records = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").strip().splitlines()]
    prompt_types = {r["prompt_type"] for r in records}
    assert prompt_types == {"baseline", "control"}
