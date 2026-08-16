"""Tests for prism.figures -- figure generation reading from data/results/
(Prompt 11, feeding REQ-12's report).

Per CLAUDE.md Section 3, figure generation sits at the bottom of this
project's test-priority list ("everything else, as time allows"), behind
the injection hook, the sampler, and the stats module. Coverage here still
targets the module's real logic (the predicted-probability curve and the
REQ-10 skip check) rather than padding with trivial matplotlib smoke tests
-- the drawing calls themselves are only checked for "did a real file get
written," not pixel content.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prism.figures as figures_module
from prism.figures import (
    layer_comparison_available,
    main,
    make_calibration_figure,
    make_identifiability_detection_figure,
    make_layer_comparison_figure_if_available,
    predicted_detection_curve,
    summarize_calibration_pilot,
)

# --- fixtures -----------------------------------------------------------------


def _regression_results(
    *,
    const_estimate: float = -1.0,
    estimate: float = 0.5,
    ci_low: float = -1.2,
    ci_high: float = 2.2,
    mean: float = 0.8,
    std: float = 0.1,
) -> dict:
    return {
        "coefficients": {
            "const": {"estimate": const_estimate, "ci_low": const_estimate - 1, "ci_high": const_estimate + 1},
            "identifiability_score": {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high},
        },
        "standardization": {
            "identifiability_score": {"mean": mean, "std": std},
        },
    }


def _analysis_table(n_negative: int = 20, n_positive: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.6, 1.0, size=n_negative + n_positive)
    detected = [False] * n_negative + [True] * n_positive
    return pd.DataFrame(
        {
            "prompt_type": ["detection"] * (n_negative + n_positive),
            "identifiability_score": scores,
            "judge_detected": detected,
            "layer": [4] * (n_negative + n_positive),
        }
    )


def _calibration_pilot_jsonl(path: Path, strengths: list[float], n_features: int = 3) -> None:
    rows = []
    for feature_id in range(n_features):
        for strength in strengths:
            n_words = max(1, int(50 - strength * 3 + feature_id))
            rows.append(
                {
                    "feature_id": feature_id,
                    "strength": strength,
                    "coherence": {
                        "likely_degenerate": strength >= 8,
                        "reason": "high_repetition" if strength >= 8 else None,
                        "repetition_rate": 0.1 * strength,
                        "n_words": n_words,
                    },
                }
            )
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --- predicted_detection_curve() -----------------------------------------------


def test_predicted_detection_curve_stays_in_zero_one_range() -> None:
    curve = predicted_detection_curve(_regression_results(), x_min=0.5, x_max=1.0, n_points=50)

    assert len(curve) == 50
    assert (curve["p_mean"].between(0, 1)).all()
    assert (curve["p_ci_low"].between(0, 1)).all()
    assert (curve["p_ci_high"].between(0, 1)).all()
    assert (curve["p_ci_low"] <= curve["p_mean"] + 1e-9).all()
    assert (curve["p_mean"] <= curve["p_ci_high"] + 1e-9).all()


def test_predicted_detection_curve_pinches_at_the_covariate_mean() -> None:
    # At x == mean, the standardized covariate is 0 for every curve --
    # the coefficient's CI bound can't move a term that's multiplied by
    # zero, so all three curves must agree exactly at that one point,
    # regardless of how wide the reported CI is.
    results = _regression_results(mean=0.8, std=0.1, estimate=0.5, ci_low=-1.2, ci_high=2.2)
    curve = predicted_detection_curve(results, x_min=0.7, x_max=0.9, n_points=3)

    midpoint = curve.iloc[1]
    assert midpoint["x"] == pytest.approx(0.8)
    assert midpoint["p_mean"] == pytest.approx(midpoint["p_ci_low"], abs=1e-9)
    assert midpoint["p_mean"] == pytest.approx(midpoint["p_ci_high"], abs=1e-9)


def test_predicted_detection_curve_is_increasing_for_a_positive_coefficient() -> None:
    results = _regression_results(estimate=2.0, mean=0.8, std=0.1)
    curve = predicted_detection_curve(results, x_min=0.6, x_max=1.0, n_points=40)

    assert np.all(np.diff(curve["p_mean"]) > 0)


def test_predicted_detection_curve_is_decreasing_for_a_negative_coefficient() -> None:
    results = _regression_results(estimate=-2.0, mean=0.8, std=0.1)
    curve = predicted_detection_curve(results, x_min=0.6, x_max=1.0, n_points=40)

    assert np.all(np.diff(curve["p_mean"]) < 0)


def test_predicted_detection_curve_raises_on_missing_const() -> None:
    results = _regression_results()
    del results["coefficients"]["const"]

    with pytest.raises(KeyError, match="const"):
        predicted_detection_curve(results, x_min=0.6, x_max=1.0)


def test_predicted_detection_curve_raises_on_missing_covariate_coefficient() -> None:
    with pytest.raises(KeyError, match="decoder_norm"):
        predicted_detection_curve(_regression_results(), x_min=0.6, x_max=1.0, covariate="decoder_norm")


def test_predicted_detection_curve_raises_on_missing_standardization_entry() -> None:
    results = _regression_results()
    del results["standardization"]["identifiability_score"]

    with pytest.raises(KeyError, match="standardization"):
        predicted_detection_curve(results, x_min=0.6, x_max=1.0)


# --- make_identifiability_detection_figure() -----------------------------------


def test_make_identifiability_detection_figure_writes_svg_and_png(tmp_path: Path) -> None:
    paths = make_identifiability_detection_figure(_analysis_table(), _regression_results(), output_dir=tmp_path)

    assert {p.suffix for p in paths} == {".svg", ".png"}
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_make_identifiability_detection_figure_raises_on_no_detection_rows(tmp_path: Path) -> None:
    table = _analysis_table()
    table["prompt_type"] = "control"

    with pytest.raises(ValueError, match="detection"):
        make_identifiability_detection_figure(table, _regression_results(), output_dir=tmp_path)


def test_make_identifiability_detection_figure_raises_on_missing_column(tmp_path: Path) -> None:
    table = _analysis_table().drop(columns=["judge_detected"])

    with pytest.raises(ValueError, match="judge_detected"):
        make_identifiability_detection_figure(table, _regression_results(), output_dir=tmp_path)


# --- summarize_calibration_pilot() ----------------------------------------------


def test_summarize_calibration_pilot_returns_one_row_per_strength(tmp_path: Path) -> None:
    pilot_path = tmp_path / "calibration_pilot.jsonl"
    _calibration_pilot_jsonl(pilot_path, strengths=[1.0, 2.0, 4.0], n_features=3)

    summary = summarize_calibration_pilot(pilot_path)

    assert sorted(summary["strength"]) == [1.0, 2.0, 4.0]
    assert "n_words" in summary.columns


def test_summarize_calibration_pilot_matches_a_hand_computed_mean(tmp_path: Path) -> None:
    pilot_path = tmp_path / "calibration_pilot.jsonl"
    # n_words = 50 - strength*3 + feature_id, feature_id in {0, 1, 2}
    _calibration_pilot_jsonl(pilot_path, strengths=[1.0], n_features=3)

    summary = summarize_calibration_pilot(pilot_path)

    expected_mean = ((50 - 3) + (50 - 3 + 1) + (50 - 3 + 2)) / 3
    assert summary.loc[summary["strength"] == 1.0, "n_words"].iloc[0] == pytest.approx(expected_mean)


def test_summarize_calibration_pilot_raises_on_empty_pilot_log(tmp_path: Path) -> None:
    pilot_path = tmp_path / "calibration_pilot.jsonl"
    pilot_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no pilot records"):
        summarize_calibration_pilot(pilot_path)


# --- make_calibration_figure() -------------------------------------------------


def test_make_calibration_figure_writes_svg_and_png(tmp_path: Path) -> None:
    pilot_path = tmp_path / "calibration_pilot.jsonl"
    _calibration_pilot_jsonl(pilot_path, strengths=[1.0, 2.0, 4.0, 8.0, 16.0])
    summary = summarize_calibration_pilot(pilot_path)

    paths = make_calibration_figure(pilot_path, summary, output_dir=tmp_path / "figures")

    assert {p.suffix for p in paths} == {".svg", ".png"}
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_make_calibration_figure_raises_on_empty_pilot_log(tmp_path: Path) -> None:
    pilot_path = tmp_path / "calibration_pilot.jsonl"
    pilot_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no pilot records"):
        make_calibration_figure(pilot_path, pd.DataFrame({"strength": [1.0], "n_words": [10.0]}), output_dir=tmp_path / "figures")


def test_make_calibration_figure_raises_without_a_precomputed_summary(tmp_path: Path) -> None:
    pilot_path = tmp_path / "calibration_pilot.jsonl"
    _calibration_pilot_jsonl(pilot_path, strengths=[1.0, 2.0])

    with pytest.raises(ValueError, match="summarize_calibration_pilot"):
        make_calibration_figure(pilot_path, None, output_dir=tmp_path / "figures")


# --- layer_comparison_available() / make_layer_comparison_figure_if_available() ---


def test_layer_comparison_available_false_for_a_single_layer() -> None:
    table = pd.DataFrame({"layer": [4, 4, 4, 4]})

    assert layer_comparison_available(table) is False


def test_layer_comparison_available_true_for_more_than_one_layer() -> None:
    table = pd.DataFrame({"layer": [4, 4, 5, 5]})

    assert layer_comparison_available(table) is True


def test_layer_comparison_available_raises_without_a_layer_column() -> None:
    table = pd.DataFrame({"strength": [1, 2]})

    with pytest.raises(ValueError, match="layer"):
        layer_comparison_available(table)


def test_layer_comparison_figure_skips_gracefully_when_req10_not_attempted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    table = pd.DataFrame({"layer": [4, 4, 4]})

    with caplog.at_level(logging.INFO, logger="prism.figures"):
        result = make_layer_comparison_figure_if_available(table, output_dir=tmp_path)

    assert result == []
    assert not list(tmp_path.glob("*"))
    assert "REQ-10 was not attempted" in caplog.text


def test_layer_comparison_figure_raises_when_real_multi_layer_data_appears(tmp_path: Path) -> None:
    # This is the honest failure mode: if a future run actually logs trials
    # at a second layer, the figure logic to compare them still doesn't
    # exist. That should be a loud NotImplementedError, not a silent skip
    # that looks identical to "REQ-10 wasn't attempted."
    table = pd.DataFrame({"layer": [4, 4, 5, 5]})

    with pytest.raises(NotImplementedError, match="more than one injection layer"):
        make_layer_comparison_figure_if_available(table, output_dir=tmp_path)


# --- main() (CLI) ---------------------------------------------------------------


def test_main_writes_the_two_available_figures_and_skips_the_third(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    analysis_table_path = tmp_path / "analysis_table.csv"
    _analysis_table().to_csv(analysis_table_path, index=False)

    regression_results_path = tmp_path / "regression_results.json"
    regression_results_path.write_text(json.dumps(_regression_results()), encoding="utf-8")

    calibration_pilot_path = tmp_path / "calibration_pilot.jsonl"
    _calibration_pilot_jsonl(calibration_pilot_path, strengths=[1.0, 2.0, 4.0, 8.0])

    output_dir = tmp_path / "figures"
    calibration_summary_path = tmp_path / "calibration_summary.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prism.figures",
            "--regression-results-path", str(regression_results_path),
            "--analysis-table-path", str(analysis_table_path),
            "--calibration-pilot-path", str(calibration_pilot_path),
            "--calibration-summary-path", str(calibration_summary_path),
            "--output-dir", str(output_dir),
        ],
    )

    main()

    written = sorted(p.name for p in output_dir.glob("*"))
    assert written == [
        "identifiability_vs_detection.png",
        "identifiability_vs_detection.svg",
        "strength_calibration.png",
        "strength_calibration.svg",
    ]
    assert calibration_summary_path.exists()
    summary = pd.read_csv(calibration_summary_path)
    assert sorted(summary["strength"]) == [1.0, 2.0, 4.0, 8.0]

    captured = capsys.readouterr()
    assert "skipped layer-comparison figure" in captured.out
