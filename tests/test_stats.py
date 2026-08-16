"""Tests for prism.stats -- the regression and AUC comparison (REQ-9)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.stats import (
    DETECTION_TARGET_COLUMN,
    JudgeNotValidatedError,
    _add_detection_target,
    _detection_subset,
    build_analysis_table,
    compare_classifiers,
    fit_inference_model,
)

# --- fixtures -----------------------------------------------------------------


def _flag(tmp_path: Path, note: str = "reviewed the sample, judge grades match") -> Path:
    path = tmp_path / "judge_validated.flag"
    path.write_text(json.dumps({"reviewer_note": note}) + "\n", encoding="utf-8")
    return path


def _write_trials(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _audit_csv(path: Path, rows: dict[str, list]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


_DETECTED_TRUE = {
    "detected": True,
    "concept_identified": None,
    "concept_confidence": None,
    "identified_before_verbalizing": None,
    "coherent": True,
    "reasoning": "affirmed noticing something unusual",
}
_DETECTED_FALSE = {
    "detected": False,
    "concept_identified": None,
    "concept_confidence": None,
    "identified_before_verbalizing": None,
    "coherent": False,
    "reasoning": "no affirmative detection in the response",
}
_CONTROL_SCORE = {"affirmative": False, "coherent": True, "reasoning": "answered no as expected"}


def _trial(
    trial_id: str,
    feature_id: int,
    *,
    prompt_type: str = "detection",
    strength: "float | None" = 1.0,
    judge_scores: "dict | None" = None,
    excluded: bool = False,
    exclusion_reason: "str | None" = None,
) -> dict:
    return {
        "trial_id": trial_id,
        "feature_id": feature_id,
        "layer": 4,
        "layer_source": "adr-0009-fallback",
        "strength": strength,
        "prompt_type": prompt_type,
        "seed": 0,
        "temperature": 1.0,
        "model_name": "EleutherAI/pythia-70m-deduped",
        "model_checkpoint_revision": "e93a9fa",
        "sae_checkpoint_repo": "ghidav/pythia-70m-deduped-sae",
        "sae_checkpoint_revision": "473774a",
        "sae_checkpoint_sha256": "fdcb4553",
        "model_response": {"detection": {"prompt": "p", "response": "r"}, "affirmative": False, "naming": None},
        "judge_scores": judge_scores,
        "timestamp": "2026-08-15T00:00:00+00:00",
        "git_commit": "deadbeef",
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
    }


def _minimal_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """One feature, three prompt_types, a mix of detected outcomes -- the
    smallest fixture that exercises the join, the exclusion filter, and the
    prompt_type split all at once.
    """
    trials_path = tmp_path / "trials.jsonl"
    _write_trials(
        trials_path,
        [
            _trial("detection::f1::s1", 1, strength=1.0, judge_scores=_DETECTED_TRUE),
            _trial("detection::f1::s2", 1, strength=2.0, judge_scores=_DETECTED_FALSE),
            _trial("baseline::f1", 1, prompt_type="baseline", strength=None, judge_scores=_DETECTED_FALSE),
            _trial("control::f1::q1", 1, prompt_type="control", strength=1.0, judge_scores=_CONTROL_SCORE),
            _trial(
                "detection::f1::s4",
                1,
                strength=4.0,
                judge_scores=None,
                excluded=True,
                exclusion_reason="judge refusal (category=bio)",
            ),
        ],
    )
    audit_path = tmp_path / "features.csv"
    _audit_csv(
        audit_path,
        {
            "feature_id": [1],
            "identifiability_score": [0.5],
            "decoder_norm": [1.0],
            "activation_frequency": [0.001],
        },
    )
    return trials_path, audit_path, _flag(tmp_path)


# --- build_analysis_table ------------------------------------------------------


def test_build_analysis_table_raises_without_validation_flag(tmp_path) -> None:
    trials_path, audit_path, flag_path = _minimal_dataset(tmp_path)
    flag_path.unlink()

    with pytest.raises(JudgeNotValidatedError, match="does not exist"):
        build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)


def test_build_analysis_table_joins_feature_metadata_onto_every_trial(tmp_path) -> None:
    trials_path, audit_path, flag_path = _minimal_dataset(tmp_path)

    table = build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)

    assert (table["identifiability_score"] == 0.5).all()
    assert (table["decoder_norm"] == 1.0).all()
    assert (table["activation_frequency"] == 0.001).all()


def test_build_analysis_table_flattens_judge_scores_into_columns(tmp_path) -> None:
    trials_path, audit_path, flag_path = _minimal_dataset(tmp_path)

    table = build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)

    detection_row = table.loc[table["trial_id"] == "detection::f1::s1"].iloc[0]
    assert detection_row["judge_detected"] == True  # noqa: E712
    control_row = table.loc[table["trial_id"] == "control::f1::q1"].iloc[0]
    assert control_row["judge_affirmative"] == False  # noqa: E712


def test_build_analysis_table_drops_excluded_trials(tmp_path) -> None:
    trials_path, audit_path, flag_path = _minimal_dataset(tmp_path)

    table = build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)

    assert "detection::f1::s4" not in set(table["trial_id"])
    assert len(table) == 4  # five records written, one excluded


def test_build_analysis_table_raises_on_non_excluded_trial_missing_judge_scores(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    _write_trials(trials_path, [_trial("detection::f1::s1", 1, judge_scores=None, excluded=False)])
    audit_path = tmp_path / "features.csv"
    _audit_csv(audit_path, {"feature_id": [1], "identifiability_score": [0.5], "decoder_norm": [1.0], "activation_frequency": [0.001]})
    flag_path = _flag(tmp_path)

    with pytest.raises(ValueError, match="no judge_scores yet"):
        build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)


def test_build_analysis_table_raises_on_unmatched_feature_id(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    _write_trials(trials_path, [_trial("detection::f99::s1", 99, judge_scores=_DETECTED_TRUE)])
    audit_path = tmp_path / "features.csv"
    _audit_csv(audit_path, {"feature_id": [1], "identifiability_score": [0.5], "decoder_norm": [1.0], "activation_frequency": [0.001]})
    flag_path = _flag(tmp_path)

    with pytest.raises(ValueError, match=r"feature_id\(s\) \[99\]"):
        build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)


# --- _detection_subset / _add_detection_target ---------------------------------


def test_detection_subset_filters_to_detection_prompt_type(tmp_path) -> None:
    trials_path, audit_path, flag_path = _minimal_dataset(tmp_path)
    table = build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)

    subset = _detection_subset(table)

    assert set(subset["prompt_type"]) == {"detection"}
    assert len(subset) == 2


def test_detection_subset_raises_when_no_detection_rows() -> None:
    only_control = pd.DataFrame({"prompt_type": ["control", "control"], "strength": [1.0, 1.0]})

    with pytest.raises(ValueError, match="no prompt_type == 'detection' rows"):
        _detection_subset(only_control)


def test_detection_subset_raises_on_null_strength() -> None:
    malformed = pd.DataFrame({"prompt_type": ["detection", "detection"], "strength": [1.0, None]})

    with pytest.raises(ValueError, match="null strength value"):
        _detection_subset(malformed)


def test_add_detection_target_matches_judge_detected(tmp_path) -> None:
    trials_path, audit_path, flag_path = _minimal_dataset(tmp_path)
    table = build_analysis_table(trials_path, audit_path, validation_flag_path=flag_path)
    subset = _detection_subset(table)

    with_target = _add_detection_target(subset)

    by_trial = with_target.set_index("trial_id")[DETECTION_TARGET_COLUMN]
    assert bool(by_trial["detection::f1::s1"]) is True
    assert bool(by_trial["detection::f1::s2"]) is False


# --- fit_inference_model --------------------------------------------------------


def _synthetic_detection_table(n: int = 300, seed: int = 0, effect: float = 3.0) -> pd.DataFrame:
    """A table where detection depends strongly on identifiability_score and
    on nothing else, by construction -- the "right answer" a regression
    fit against it should recover: a clearly positive, non-degenerate
    coefficient on identifiability_score.
    """
    rng = np.random.default_rng(seed)
    identifiability_score = rng.uniform(0.1, 0.99, n)
    z = (identifiability_score - identifiability_score.mean()) / identifiability_score.std()
    prob = 1 / (1 + np.exp(-effect * z))
    detected = rng.uniform(size=n) < prob
    return pd.DataFrame(
        {
            "trial_id": [f"detection::f{i}::s1" for i in range(n)],
            "feature_id": range(n),
            "prompt_type": "detection",
            "strength": rng.choice([1.0, 2.0, 4.0, 8.0], n),
            "identifiability_score": identifiability_score,
            "decoder_norm": rng.uniform(0.1, 2.0, n),
            "activation_frequency": rng.uniform(1e-4, 1e-2, n),
            "judge_detected": detected,
        }
    )


def test_fit_inference_model_raises_without_validation_flag(tmp_path) -> None:
    flag_path = tmp_path / "judge_validated.flag"  # never written
    table = _synthetic_detection_table()

    with pytest.raises(JudgeNotValidatedError):
        fit_inference_model(table, validation_flag_path=flag_path)


def test_fit_inference_model_returns_cis_for_every_covariate(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    table = _synthetic_detection_table()

    result = fit_inference_model(table, validation_flag_path=flag_path)

    assert result["converged"] is True
    for covariate in ("identifiability_score", "decoder_norm", "activation_frequency", "strength"):
        coef = result["coefficients"][covariate]
        assert coef["ci_low"] < coef["estimate"] < coef["ci_high"]


def test_fit_inference_model_recovers_the_known_positive_effect(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    table = _synthetic_detection_table(effect=3.0)

    result = fit_inference_model(table, validation_flag_path=flag_path)

    coef = result["coefficients"]["identifiability_score"]
    assert coef["estimate"] > 0
    assert coef["ci_low"] > 0  # the true effect is strong enough that the CI should exclude zero


def test_fit_inference_model_restricts_to_detection_prompt_type(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    detection = _synthetic_detection_table(n=50)
    baseline = detection.copy()
    baseline["prompt_type"] = "baseline"
    baseline["strength"] = None
    mixed = pd.concat([detection, baseline], ignore_index=True)

    result = fit_inference_model(mixed, validation_flag_path=flag_path)

    assert result["n_trials"] == 50


def test_fit_inference_model_reports_a_zero_variance_covariate_instead_of_dividing_by_zero(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    n = 10
    table = pd.DataFrame(
        {
            "trial_id": [f"detection::f{i}::s1" for i in range(n)],
            "feature_id": range(n),
            "prompt_type": "detection",
            "strength": [1.0] * n,  # constant across this subset: zero variance
            "identifiability_score": np.linspace(0.1, 0.95, n),
            "decoder_norm": np.linspace(0.1, 2.0, n),
            "activation_frequency": np.linspace(1e-4, 1e-2, n),
            "judge_detected": [False] * 5 + [True] * 5,
        }
    )

    result = fit_inference_model(table, validation_flag_path=flag_path)

    assert result["converged"] is False
    assert "strength" in result["convergence_note"]
    assert result["coefficients"] == {}


def test_fit_inference_model_reports_nonconvergence_on_quasi_separated_data(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    n = 10
    table = pd.DataFrame(
        {
            "trial_id": [f"detection::f{i}::s1" for i in range(n)],
            "feature_id": range(n),
            "prompt_type": "detection",
            "strength": [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0, 1.0, 2.0],
            "identifiability_score": np.linspace(0.1, 0.95, n),
            "decoder_norm": np.linspace(0.1, 2.0, n),
            "activation_frequency": np.linspace(1e-4, 1e-2, n),
            "judge_detected": [False] * 5 + [True] * 5,
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_inference_model(table, validation_flag_path=flag_path)

    assert result["converged"] is False
    assert result["convergence_note"] != ""


# --- compare_classifiers --------------------------------------------------------


def test_compare_classifiers_raises_without_validation_flag(tmp_path) -> None:
    flag_path = tmp_path / "judge_validated.flag"
    table = _synthetic_detection_table()

    with pytest.raises(JudgeNotValidatedError):
        compare_classifiers(table, validation_flag_path=flag_path)


def test_compare_classifiers_returns_one_row_per_classifier(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    table = _synthetic_detection_table()

    result = compare_classifiers(table, validation_flag_path=flag_path)

    assert sorted(result["classifier"]) == ["decoder_norm", "identifiability_score"]
    assert list(result.columns) == ["classifier", "auc", "n_trials", "n_detections"]


def test_compare_classifiers_perfectly_separable_case_yields_auc_one(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    rng = np.random.default_rng(1)
    n = 40
    identifiability_score = np.concatenate([rng.uniform(0.01, 0.4, n // 2), rng.uniform(0.6, 0.99, n // 2)])
    table = pd.DataFrame(
        {
            "trial_id": [f"detection::f{i}::s1" for i in range(n)],
            "feature_id": range(n),
            "prompt_type": "detection",
            "strength": rng.choice([1.0, 2.0, 4.0, 8.0], n),
            "identifiability_score": identifiability_score,
            "decoder_norm": rng.uniform(0.1, 2.0, n),
            "activation_frequency": rng.uniform(1e-4, 1e-2, n),
            "judge_detected": np.array([False] * (n // 2) + [True] * (n // 2)),
        }
    )

    result = compare_classifiers(table, validation_flag_path=flag_path)

    identifiability_auc = result.set_index("classifier").loc["identifiability_score", "auc"]
    assert identifiability_auc == pytest.approx(1.0)


def test_compare_classifiers_no_signal_case_yields_auc_near_half(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    rng = np.random.default_rng(2)
    n = 2000
    table = pd.DataFrame(
        {
            "trial_id": [f"detection::f{i}::s1" for i in range(n)],
            "feature_id": range(n),
            "prompt_type": "detection",
            "strength": rng.choice([1.0, 2.0, 4.0, 8.0], n),
            "identifiability_score": rng.uniform(0, 1, n),
            "decoder_norm": rng.uniform(0.1, 2.0, n),
            "activation_frequency": rng.uniform(1e-4, 1e-2, n),
            "judge_detected": rng.uniform(size=n) < 0.3,  # independent of every covariate by construction
        }
    )

    result = compare_classifiers(table, validation_flag_path=flag_path)

    assert (result["auc"].between(0.4, 0.6)).all()


def test_compare_classifiers_uses_the_same_subset_as_fit_inference_model(tmp_path) -> None:
    flag_path = _flag(tmp_path)
    detection = _synthetic_detection_table(n=50)
    control = detection.copy()
    control["prompt_type"] = "control"
    mixed = pd.concat([detection, control], ignore_index=True)

    result = compare_classifiers(mixed, validation_flag_path=flag_path)

    assert (result["n_trials"] == 50).all()


# --- N/A: cases ruled out by an upstream guarantee ------------------------------


# N/A: load_feature_audit() (reused by build_analysis_table()) already
# rejects identifiability_score outside [0, 1] and negative decoder_norm
# before the join happens (tests/test_features.py already covers this at
# its source) -- an out-of-range value can never reach fit_inference_model()
# or compare_classifiers() through the normal build_analysis_table() path.
@pytest.mark.skip(
    reason="N/A: load_feature_audit() rejects this before build_analysis_table() can produce it"
)
def test_na_out_of_range_covariates_cannot_reach_the_regression() -> None:
    pass


# N/A: runner.py's resumability logic (REQ-6) guarantees trial_id
# uniqueness in trials.jsonl by construction (it skips any trial_id
# already present before writing a new one), so build_analysis_table()
# never needs to deduplicate rows by trial_id.
@pytest.mark.skip(reason="N/A: runner.py's resumable writer guarantees trial_id uniqueness upstream")
def test_na_duplicate_trial_id_in_trials_jsonl() -> None:
    pass


# N/A: ADR-0018's structured judge output schema makes `detected` a
# required field on every scored detection/baseline trial (never null,
# unlike concept_identified/identified_before_verbalizing, which are
# conditionally null) -- a non-excluded detection-type row with a missing
# judge_detected value cannot occur through the normal scoring path.
@pytest.mark.skip(
    reason="N/A: ADR-0018's structured output schema always populates judge_detected on a scored trial"
)
def test_na_null_judge_detected_on_a_non_excluded_detection_trial() -> None:
    pass
