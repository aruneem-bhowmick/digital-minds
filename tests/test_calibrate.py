"""Tests for prism.inject's calibration pilot (REQ-5)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from prism.inject import (
    pilot_coherence_flag,
    run_calibration_pilot,
    save_pilot_records,
    select_pilot_features,
    summarize_pilot,
)
from prism.layers import get_fallback_layer
from prism.models import load_model_and_sae
from prism.prompts import detection_prompt

CONFIG_PATH = "configs/experiment.yaml"


def _sampled_df(rows_per_tertile: int = 4) -> pd.DataFrame:
    rows = []
    feature_id = 0
    for tertile in ("low", "medium", "high"):
        for i in range(rows_per_tertile):
            rows.append(
                {
                    "feature_id": feature_id,
                    "identifiability_score": 0.1 + 0.01 * i,
                    "decoder_norm": 1.0 + 0.1 * i,
                    "activation_frequency": 0.01,
                    "identifiability_tertile": tertile,
                }
            )
            feature_id += 1
    return pd.DataFrame(rows)


# --- select_pilot_features: tertile coverage and reproducibility -----------


def test_select_pilot_features_spans_all_three_tertiles() -> None:
    sampled = _sampled_df()

    pilot = select_pilot_features(sampled, n_features=5, seed=0)

    assert len(pilot) == 5
    assert set(pilot["identifiability_tertile"]) == {"low", "medium", "high"}


def test_select_pilot_features_splits_the_remainder_across_tertiles() -> None:
    sampled = _sampled_df()

    # 5 // 3 = 1 remainder 2 -> low and medium get one extra each.
    pilot = select_pilot_features(sampled, n_features=5, seed=0)
    counts = pilot["identifiability_tertile"].value_counts()

    assert counts["low"] == 2
    assert counts["medium"] == 2
    assert counts["high"] == 1


def test_select_pilot_features_is_reproducible_under_a_fixed_seed() -> None:
    sampled = _sampled_df()

    first = select_pilot_features(sampled, n_features=6, seed=7)
    second = select_pilot_features(sampled, n_features=6, seed=7)

    pd.testing.assert_frame_equal(
        first.sort_values("feature_id").reset_index(drop=True),
        second.sort_values("feature_id").reset_index(drop=True),
    )


def test_select_pilot_features_different_seeds_can_draw_different_features() -> None:
    sampled = _sampled_df(rows_per_tertile=8)

    first = select_pilot_features(sampled, n_features=6, seed=0)
    second = select_pilot_features(sampled, n_features=6, seed=1)

    assert set(first["feature_id"]) != set(second["feature_id"])


# --- select_pilot_features: validation --------------------------------------


def test_select_pilot_features_rejects_missing_tertile_column() -> None:
    sampled = _sampled_df().drop(columns=["identifiability_tertile"])

    with pytest.raises(ValueError, match="identifiability_tertile"):
        select_pilot_features(sampled, n_features=5)


def test_select_pilot_features_rejects_missing_decoder_norm_column() -> None:
    # run_calibration_pilot() reads decoder_norm off every row unconditionally;
    # this must fail here, before the model/SAE load, not as a KeyError
    # partway through a pilot run.
    sampled = _sampled_df().drop(columns=["decoder_norm"])

    with pytest.raises(ValueError, match="decoder_norm"):
        select_pilot_features(sampled, n_features=5)


def test_select_pilot_features_rejects_missing_feature_id_column() -> None:
    sampled = _sampled_df().drop(columns=["feature_id"])

    with pytest.raises(ValueError, match="feature_id"):
        select_pilot_features(sampled, n_features=5)


def test_select_pilot_features_rejects_n_features_below_three() -> None:
    sampled = _sampled_df()

    with pytest.raises(ValueError, match="n_features"):
        select_pilot_features(sampled, n_features=2)


def test_select_pilot_features_rejects_a_tertile_too_small_to_supply_its_share() -> None:
    sampled = _sampled_df(rows_per_tertile=1)  # only 1 feature per tertile

    with pytest.raises(ValueError, match="fewer than"):
        select_pilot_features(sampled, n_features=6)  # needs 2 per tertile


# --- pilot_coherence_flag: degenerate text ----------------------------------


def test_pilot_coherence_flag_flags_near_empty_text() -> None:
    flag = pilot_coherence_flag("no")

    assert flag["likely_degenerate"] is True
    assert flag["reason"] == "too_short"
    assert flag["repetition_rate"] is None


def test_pilot_coherence_flag_flags_empty_string() -> None:
    flag = pilot_coherence_flag("")

    assert flag["likely_degenerate"] is True
    assert flag["n_words"] == 0


def test_pilot_coherence_flag_flags_a_repetition_loop() -> None:
    # An exact-phrase loop like this is caught by the repeated-segment
    # check (the whole "the cat sat " block repeats verbatim), which runs
    # ahead of the trigram check -- see test_pilot_coherence_flag_
    # respects_a_custom_threshold for a case that exercises the trigram
    # path specifically.
    text = "the cat sat " * 20

    flag = pilot_coherence_flag(text)

    assert flag["likely_degenerate"] is True
    assert flag["reason"] == "repeated_segment"


def test_pilot_coherence_flag_flags_a_hyphenated_subword_loop() -> None:
    # A real strength-8 pilot output: whitespace-splitting treats this as
    # only 4 "words" (the hyphen-chain is one token), so the word-trigram
    # count alone never sees it -- this is exactly the shape a bare
    # trigram heuristic misses.
    text = "I'm not a good-cqe-cqe-cqe-cqe-cqe-cqe-cqe-cqe-cqe-cqe-cqe-cqe-cqe-"

    flag = pilot_coherence_flag(text)

    assert flag["likely_degenerate"] is True
    assert flag["reason"] == "repeated_segment"


def test_pilot_coherence_flag_flags_a_punctuation_loop_with_no_word_boundaries() -> None:
    text = "::::::::::::::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-"

    flag = pilot_coherence_flag(text)

    assert flag["likely_degenerate"] is True
    assert flag["reason"] == "repeated_segment"


def test_pilot_coherence_flag_does_not_crash_when_min_words_is_below_three() -> None:
    # min_words=1 lets a 2-word response past the too-short guard, which
    # previously fell through to a trigram computation on zero trigrams
    # (0/0 -> ZeroDivisionError). Must return a verdict, not raise.
    flag = pilot_coherence_flag("no thanks", min_words=1)

    assert flag["likely_degenerate"] is False
    assert flag["repetition_rate"] is None


def test_pilot_coherence_flag_does_not_flag_a_legitimately_doubled_word() -> None:
    text = "That that is a common typo but it is still ordinary English prose here"

    flag = pilot_coherence_flag(text)

    assert flag["likely_degenerate"] is False


# --- pilot_coherence_flag: ordinary coherent text ---------------------------


def test_pilot_coherence_flag_does_not_flag_ordinary_prose() -> None:
    text = (
        "Yes, something does feel a little different right now. It is hard "
        "to describe exactly, but there is a word that keeps coming to mind "
        "that does not seem to belong to this conversation."
    )

    flag = pilot_coherence_flag(text)

    assert flag["likely_degenerate"] is False
    assert flag["reason"] is None
    assert flag["repetition_rate"] < 0.3


def test_pilot_coherence_flag_respects_a_custom_threshold() -> None:
    text = "a b c a b c a b c"  # every trigram after the first repeats

    lenient = pilot_coherence_flag(text, repetition_threshold=0.9)
    strict = pilot_coherence_flag(text, repetition_threshold=0.1)

    assert lenient["likely_degenerate"] is False
    assert strict["likely_degenerate"] is True


# --- summarize_pilot / save_pilot_records -----------------------------------


def _fake_records() -> list[dict]:
    return [
        {
            "feature_id": 1,
            "identifiability_tertile": "low",
            "identifiability_score": 0.12,
            "decoder_norm": 1.0,
            "layer": 4,
            "layer_source": "adr-0009-fallback",
            "strength": 8.0,
            "prompt": detection_prompt(),
            "temperature": 0,
            "response_text": "no, nothing unusual.",
            "coherence": pilot_coherence_flag("no, nothing unusual."),
            "git_commit": "deadbeef",
            "timestamp": "2026-08-15T00:00:00+00:00",
        },
        {
            "feature_id": 2,
            "identifiability_tertile": "high",
            "identifiability_score": 0.98,
            "decoder_norm": 1.2,
            "layer": 4,
            "layer_source": "adr-0009-fallback",
            "strength": 64.0,
            "prompt": detection_prompt(),
            "temperature": 0,
            "response_text": "the the the the the the the",
            "coherence": pilot_coherence_flag("the the the the the the the"),
            "git_commit": "deadbeef",
            "timestamp": "2026-08-15T00:00:00+00:00",
        },
    ]


def test_summarize_pilot_groups_by_strength_and_flags_degenerate_output() -> None:
    summary = summarize_pilot(_fake_records())

    assert "=== strength 8.0 ===" in summary
    assert "=== strength 64.0 ===" in summary
    assert "[ok]" in summary
    assert "[DEGENERATE]" in summary


def test_save_pilot_records_writes_one_json_line_per_record(tmp_path) -> None:
    output_path = tmp_path / "calibration_pilot.jsonl"

    returned = save_pilot_records(_fake_records(), output_path)

    assert returned == output_path
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["feature_id"] == 1
    assert parsed[1]["feature_id"] == 2


def test_save_pilot_records_overwrites_rather_than_appends(tmp_path) -> None:
    output_path = tmp_path / "calibration_pilot.jsonl"

    save_pilot_records(_fake_records(), output_path)
    save_pilot_records(_fake_records()[:1], output_path)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


# --- integration: real model, real SAE, real generate() ---------------------


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def loaded_pair(config):
    return load_model_and_sae(config)


@pytest.mark.integration
def test_run_calibration_pilot_produces_a_full_record_per_feature_strength_pair(loaded_pair, config) -> None:
    sampled = _sampled_df(rows_per_tertile=1)
    pilot_features = select_pilot_features(sampled, n_features=3, seed=0)
    layer = get_fallback_layer(loaded_pair.model.cfg.n_layers)

    records = run_calibration_pilot(
        loaded_pair,
        pilot_features,
        strengths=[0.0, 5.0],
        layer=layer,
        layer_source="adr-0009-fallback",
        prompt=detection_prompt(),
        config=config,
        pilot_feature_seed=0,
        max_new_tokens=6,
    )

    assert len(records) == 3 * 2
    record = records[0]
    assert record["layer"] == layer
    assert record["layer_source"] == "adr-0009-fallback"
    assert record["temperature"] == 0
    assert record["model_name"] == config["model"]["name"]
    assert record["sae_checkpoint_sha256"] == config["sae"]["checkpoint_sha256"]
    assert record["pilot_feature_seed"] == 0
    assert isinstance(record["response_text"], str) and record["response_text"].strip() != ""
    assert record["git_commit"]
    assert record["timestamp"]
    assert "likely_degenerate" in record["coherence"]


@pytest.mark.integration
def test_run_calibration_pilot_zero_strength_is_deterministic_baseline_text(loaded_pair, config) -> None:
    sampled = _sampled_df(rows_per_tertile=1)
    pilot_features = select_pilot_features(sampled, n_features=3, seed=0).iloc[[0]]
    layer = get_fallback_layer(loaded_pair.model.cfg.n_layers)

    first = run_calibration_pilot(
        loaded_pair,
        pilot_features,
        strengths=[0.0],
        layer=layer,
        layer_source="adr-0009-fallback",
        prompt=detection_prompt(),
        config=config,
        pilot_feature_seed=0,
        max_new_tokens=6,
    )
    second = run_calibration_pilot(
        loaded_pair,
        pilot_features,
        strengths=[0.0],
        layer=layer,
        layer_source="adr-0009-fallback",
        prompt=detection_prompt(),
        config=config,
        pilot_feature_seed=0,
        max_new_tokens=6,
    )

    # Temperature 0 (do_sample=False): identical inputs must reproduce the
    # exact same generation.
    assert first[0]["response_text"] == second[0]["response_text"]
