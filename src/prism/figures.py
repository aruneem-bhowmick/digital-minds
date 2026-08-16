"""Figure generation (Prompt 11, feeding REQ-12's report).

Every ``make_*_figure()`` function draws only from data that already
exists as a committed ``data/results/`` artifact -- none of them compute a
statistic themselves, per the split ADR-0005/ADR-0006 draw between data
collection, analysis, and presentation. The one number that needed
computing before it could be plotted (the REQ-5 pilot's per-strength mean
response length) has its own precomputation step,
``summarize_calibration_pilot()``, kept out of the plotting function and
persisted to ``data/results/calibration_summary.csv`` before
``make_calibration_figure()`` ever reads it -- the same pattern
``stats.py`` (REQ-9) uses to separate computing a result from presenting
it.

Three figures are in scope per Prompt 11:

1. ``make_identifiability_detection_figure()`` -- the primary REQ-9 result.
2. ``make_calibration_figure()`` -- the REQ-5 pilot sanity check.
3. ``make_layer_comparison_figure_if_available()`` -- REQ-10 (stretch), only
   if the analysis table actually has trials at more than one layer. It
   doesn't this session, so this returns no files and logs why.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGRESSION_RESULTS_PATH = "data/results/regression_results.json"
DEFAULT_ANALYSIS_TABLE_PATH = "data/results/analysis_table.csv"
DEFAULT_CALIBRATION_PILOT_PATH = "data/results/calibration_pilot.jsonl"
DEFAULT_CALIBRATION_SUMMARY_PATH = "data/results/calibration_summary.csv"
DEFAULT_FIGURES_DIR = "data/results/figures"

# Colors are the dataviz skill's validated default palette instance
# (references/palette.md) -- swap this block, not the plotting code, if the
# report ever adopts a different palette.
COLOR_SURFACE = "#fcfcfb"
COLOR_PRIMARY_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED_INK = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SERIES_BLUE = "#2a78d6"  # categorical slot 1 -- regression curve, pilot mean line
COLOR_SERIES_BLUE_BAND = "#9ec5f4"  # sequential blue step 200 -- CI band / calibrated-band fill
COLOR_ACCENT_RED = "#e34948"  # categorical slot 6 -- the one trial graded detected


def _apply_base_style(ax: "plt.Axes") -> None:
    """Shared chrome: light surface, hairline gridlines, no chart junk."""
    ax.set_facecolor(COLOR_SURFACE)
    ax.figure.set_facecolor(COLOR_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_MUTED_INK, labelsize=9)
    ax.grid(True, axis="y", color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save_figure(fig: "plt.Figure", output_dir: Path, stem: str) -> list[Path]:
    """Write ``stem.svg`` (vector, for the report template) and ``stem.png``
    (raster, for quick preview) to ``output_dir``, creating it if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("svg", None), ("png", 200)):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


# --- Primary figure: identifiability vs. detection ---------------------------


def predicted_detection_curve(
    regression_results: dict[str, Any],
    x_min: float,
    x_max: float,
    *,
    covariate: str = "identifiability_score",
    n_points: int = 200,
) -> pd.DataFrame:
    """Marginal predicted-P(detected) curve for ``covariate`` over
    ``[x_min, x_max]``, holding every other covariate at its fitted sample
    mean (standardized value 0) and the intercept at its point estimate.

    ``p_ci_low``/``p_ci_high`` are **not** a joint delta-method confidence
    interval on the predicted probability -- that needs the fitted model's
    full coefficient covariance matrix, which ``regression_results.json``
    does not carry (``fit_inference_model()`` reports per-coefficient CIs,
    not the covariance between them). They are the same curve traced
    through ``covariate``'s own reported ``ci_low``/``ci_high`` slope
    instead of its point estimate: a direct visualization of the
    coefficient CI ``stats.py`` already computed, not a new statistic.

    Raises ``KeyError`` if ``covariate`` or ``const`` is missing from
    ``regression_results``'s coefficients or standardization blocks -- a
    malformed or mismatched results file should fail loudly here rather
    than silently produce a wrong curve.
    """
    coefficients = regression_results["coefficients"]
    if "const" not in coefficients:
        raise KeyError("regression_results['coefficients'] has no 'const' entry")
    if covariate not in coefficients:
        raise KeyError(f"regression_results['coefficients'] has no '{covariate}' entry")
    standardization = regression_results["standardization"]
    if covariate not in standardization:
        raise KeyError(f"regression_results['standardization'] has no '{covariate}' entry")

    const = coefficients["const"]["estimate"]
    coef = coefficients[covariate]
    mean = standardization[covariate]["mean"]
    std = standardization[covariate]["std"]

    x = np.linspace(x_min, x_max, n_points)
    z = (x - mean) / std

    def _sigmoid(logit: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logit))

    p_mean = _sigmoid(const + coef["estimate"] * z)
    p_bound_a = _sigmoid(const + coef["ci_low"] * z)
    p_bound_b = _sigmoid(const + coef["ci_high"] * z)

    return pd.DataFrame(
        {
            "x": x,
            "p_mean": p_mean,
            "p_ci_low": np.minimum(p_bound_a, p_bound_b),
            "p_ci_high": np.maximum(p_bound_a, p_bound_b),
        }
    )


def make_identifiability_detection_figure(
    analysis_table: pd.DataFrame,
    regression_results: dict[str, Any],
    output_dir: "str | Path" = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Primary REQ-9 result: identifiability_score vs. detection, with the
    fitted regression's marginal curve and coefficient-CI band drawn over
    the real, sparse trial-level data.

    Plots every non-excluded detection trial as its own point (jittered and
    partly transparent, since 473 of 474 trials share ``judge_detected =
    False``) rather than an aggregate. The sprint's real result is a
    near-total absence of detections and a coefficient CI wide enough to
    span both signs; this figure is meant to look exactly like that, not
    like a cleaner trend than the data supports.
    """
    detection = analysis_table.loc[analysis_table["prompt_type"] == "detection"].copy()
    if detection.empty:
        raise ValueError("analysis_table has no prompt_type == 'detection' rows to plot")
    required = {"identifiability_score", "judge_detected"}
    missing = required - set(detection.columns)
    if missing:
        raise ValueError(f"analysis_table is missing required column(s): {sorted(missing)}")

    detection["judge_detected"] = detection["judge_detected"].astype(bool)
    x_min = float(detection["identifiability_score"].min())
    x_max = float(detection["identifiability_score"].max())
    curve = predicted_detection_curve(regression_results, x_min, x_max)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _apply_base_style(ax)

    ax.fill_between(
        curve["x"],
        curve["p_ci_low"],
        curve["p_ci_high"],
        color=COLOR_SERIES_BLUE_BAND,
        alpha=0.55,
        linewidth=0,
        zorder=1,
        label="identifiability_score coefficient 95% CI",
    )
    ax.plot(
        curve["x"],
        curve["p_mean"],
        color=COLOR_SERIES_BLUE,
        linewidth=2,
        zorder=3,
        label="predicted P(detected), other covariates at mean",
    )

    negatives = detection.loc[~detection["judge_detected"]]
    positives = detection.loc[detection["judge_detected"]]
    rng = np.random.default_rng(0)  # jitter only, purely visual -- fixed seed for a reproducible figure
    jitter = rng.uniform(-0.035, 0.035, size=len(negatives))
    ax.scatter(
        negatives["identifiability_score"],
        np.zeros(len(negatives)) + jitter,
        s=14,
        color=COLOR_SECONDARY_INK,
        alpha=0.18,
        linewidths=0,
        zorder=2,
        label=f"not detected (n={len(negatives)})",
    )
    if not positives.empty:
        ax.scatter(
            positives["identifiability_score"],
            np.ones(len(positives)),
            s=110,
            marker="*",
            color=COLOR_ACCENT_RED,
            edgecolors=COLOR_PRIMARY_INK,
            linewidths=0.6,
            zorder=4,
            label=f"detected (n={len(positives)})",
        )

    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["not detected", "detected"])
    ax.set_xlabel("identifiability score", color=COLOR_SECONDARY_INK)
    ax.set_title(
        "Identifiability shows no detectable relationship to detection here",
        color=COLOR_PRIMARY_INK,
        fontsize=12,
        loc="left",
        pad=28,
    )
    n_trials = len(detection)
    n_detections = int(detection["judge_detected"].sum())
    coef = regression_results["coefficients"]["identifiability_score"]
    ax.text(
        0.0,
        1.04,
        f"n={n_trials} detection trials, {n_detections} graded detected. "
        f"coefficient {coef['estimate']:.2f}, 95% CI [{coef['ci_low']:.2f}, {coef['ci_high']:.2f}]",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLOR_MUTED_INK,
        va="bottom",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), frameon=False, fontsize=8.5, labelcolor=COLOR_SECONDARY_INK, ncols=2)

    return _save_figure(fig, Path(output_dir), "identifiability_vs_detection")


# --- Calibration sanity figure ------------------------------------------------


def _load_calibration_pilot(path: "str | Path") -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["n_words"] = df["coherence"].apply(lambda c: c["n_words"])
    return df


def summarize_calibration_pilot(
    calibration_pilot_path: "str | Path" = DEFAULT_CALIBRATION_PILOT_PATH,
) -> pd.DataFrame:
    """Per-strength mean response length from the REQ-5 pilot log.

    A distinct precomputation step, not part of ``make_calibration_figure()``
    itself -- ADR-0005/ADR-0006's rule against recomputing a statistic
    inline in a plotting script applies to a groupby mean the same way it
    applies to a regression coefficient. This function's output is meant to
    be persisted (``save_calibration_summary()``) and loaded by the figure,
    not called from inside it.
    """
    df = _load_calibration_pilot(calibration_pilot_path)
    if df.empty:
        raise ValueError(f"{calibration_pilot_path} has no pilot records to summarize")
    return (
        df.groupby("strength")["n_words"]
        .mean()
        .reset_index()
        .sort_values("strength")
        .reset_index(drop=True)
    )


def save_calibration_summary(
    summary: pd.DataFrame, output_path: "str | Path" = DEFAULT_CALIBRATION_SUMMARY_PATH
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_path


def make_calibration_figure(
    calibration_pilot_path: "str | Path" = DEFAULT_CALIBRATION_PILOT_PATH,
    calibration_summary: "pd.DataFrame | None" = None,
    output_dir: "str | Path" = DEFAULT_FIGURES_DIR,
    *,
    chosen_strengths: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> list[Path]:
    """REQ-5 pilot sanity check: response length per trial against
    injection strength, on a log strength axis.

    Every point is a real pilot trial (5 features x 8 strengths, temperature
    0, ADR-0016) -- nothing is aggregated away, so the per-feature noise
    ``calibration_notes.md`` documents (e.g. feature 5015 looping at every
    strength tested) stays visible rather than smoothed into a single curve.
    ``chosen_strengths`` marks the band actually carried into
    ``configs/experiment.yaml`` for REQ-6's systematic trials.

    ``calibration_summary`` is required and must come from
    ``summarize_calibration_pilot()`` -- this function draws the mean line
    from it rather than computing the mean itself, so the number plotted
    always matches whatever was actually persisted to
    ``data/results/calibration_summary.csv``.
    """
    df = _load_calibration_pilot(calibration_pilot_path)
    if df.empty:
        raise ValueError(f"{calibration_pilot_path} has no pilot records to plot")
    if calibration_summary is None or calibration_summary.empty:
        raise ValueError(
            "calibration_summary must be precomputed via summarize_calibration_pilot() "
            "and passed in -- this function does not compute it"
        )

    means = calibration_summary.sort_values("strength")
    strengths_sorted = sorted(df["strength"].unique())

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _apply_base_style(ax)

    ax.axvspan(
        min(chosen_strengths),
        max(chosen_strengths),
        color=COLOR_SERIES_BLUE_BAND,
        alpha=0.28,
        zorder=0,
        label=f"strength band in configs/experiment.yaml ({[int(s) for s in sorted(chosen_strengths)]})",
    )

    rng = np.random.default_rng(0)  # jitter only, purely visual -- fixed seed for a reproducible figure
    jitter = rng.uniform(0.9, 1.11, size=len(df))
    ax.scatter(
        df["strength"] * jitter,
        df["n_words"],
        s=24,
        color=COLOR_SECONDARY_INK,
        alpha=0.6,
        linewidths=0,
        zorder=2,
        label="individual pilot trial (5 features)",
    )
    ax.plot(
        means["strength"],
        means["n_words"],
        color=COLOR_SERIES_BLUE,
        marker="o",
        markersize=5,
        linewidth=2,
        zorder=3,
        label="mean across features",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(strengths_sorted)
    ax.set_xticklabels([str(int(s)) for s in strengths_sorted])
    ax.set_xlabel("injection strength", color=COLOR_SECONDARY_INK)
    ax.set_ylabel("words generated (of 60 requested)", color=COLOR_SECONDARY_INK)
    ax.set_title(
        "Response length collapses at high injection strength (REQ-5 pilot)",
        color=COLOR_PRIMARY_INK,
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=COLOR_SECONDARY_INK)

    return _save_figure(fig, Path(output_dir), "strength_calibration")


# --- Layer-comparison figure (REQ-10, conditional) ----------------------------


def layer_comparison_available(analysis_table: pd.DataFrame) -> bool:
    """Whether ``analysis_table`` carries trials at more than one distinct
    injection layer -- i.e., whether REQ-10 (the geometry-grounded layer
    sweep) actually ran and logged a second layer's trials, per ADR-0009's
    rule against mixing layers under one "primary" label without saying so.

    A single distinct layer (every trial at the ADR-0009 fallback, which is
    the case for every trial in this repo as of this session) means there
    is nothing to compare, and the layer-comparison figure should not be
    produced.
    """
    if "layer" not in analysis_table.columns:
        raise ValueError("analysis_table is missing the 'layer' column")
    return analysis_table["layer"].nunique(dropna=True) > 1


def make_layer_comparison_figure_if_available(
    analysis_table: pd.DataFrame,
    output_dir: "str | Path" = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Builds the REQ-10 layer-comparison figure only if ``analysis_table``
    actually has trials at more than one layer.

    REQ-10 was dropped from this session's scope (SPRINT-PLAN.md's own
    cut-order guidance, applied per the parent session's instructions), so
    every trial in ``data/trials/trials.jsonl`` carries
    ``layer_source == "adr-0009-fallback"`` at the single fallback layer.
    Returns an empty list and logs why, rather than erroring or drawing a
    placeholder chart with nothing real to show.
    """
    if not layer_comparison_available(analysis_table):
        logger.info(
            "skipping layer-comparison figure: analysis_table has only one "
            "distinct injection layer (%s) -- REQ-10 was not attempted this "
            "session, so there is no second layer to compare against",
            sorted(int(layer) for layer in analysis_table["layer"].dropna().unique()),
        )
        return []
    # REQ-10's own comparison logic isn't implemented anywhere in this repo
    # yet. Reaching this branch would mean a later session logged trials at
    # a second layer without adding the figure to go with them -- a real
    # gap, not the "REQ-10 not attempted" skip path above, and it should
    # fail loudly rather than silently fall through to no figure at all.
    raise NotImplementedError(
        "analysis_table has trials at more than one injection layer, but "
        "the REQ-10 layer-comparison figure has not been implemented yet"
    )


# --- CLI -----------------------------------------------------------------------


def main() -> None:
    """CLI entry point: ``python -m prism.figures``.

    Reads ``regression_results.json``, ``analysis_table.csv``, and
    ``calibration_pilot.jsonl`` from ``data/results/``; also writes
    ``calibration_summary.csv`` (the one precomputation step this module
    owns) before the calibration figure reads it back. Writes every figure
    that has real data behind it to ``data/results/figures/``.
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-results-path", default=DEFAULT_REGRESSION_RESULTS_PATH)
    parser.add_argument("--analysis-table-path", default=DEFAULT_ANALYSIS_TABLE_PATH)
    parser.add_argument("--calibration-pilot-path", default=DEFAULT_CALIBRATION_PILOT_PATH)
    parser.add_argument("--calibration-summary-path", default=DEFAULT_CALIBRATION_SUMMARY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    analysis_table = pd.read_csv(args.analysis_table_path)
    with open(args.regression_results_path, encoding="utf-8") as handle:
        regression_results = json.load(handle)

    for path in make_identifiability_detection_figure(analysis_table, regression_results, args.output_dir):
        print(f"wrote {path}")

    calibration_summary = summarize_calibration_pilot(args.calibration_pilot_path)
    summary_path = save_calibration_summary(calibration_summary, args.calibration_summary_path)
    print(f"wrote {summary_path}")
    for path in make_calibration_figure(args.calibration_pilot_path, calibration_summary, args.output_dir):
        print(f"wrote {path}")

    layer_figures = make_layer_comparison_figure_if_available(analysis_table, args.output_dir)
    if not layer_figures:
        print("skipped layer-comparison figure -- REQ-10 was not attempted this session (see log above)")
    else:
        for path in layer_figures:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
