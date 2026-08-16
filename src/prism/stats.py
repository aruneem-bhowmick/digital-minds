"""Regression and AUC comparison (REQ-9).

``build_analysis_table()`` joins every non-excluded trial in
``data/trials/trials.jsonl`` onto its feature's identifiability score,
decoder norm, and activation frequency in ``data/audit/features.csv``,
producing the single regression-ready table both analyses in this module
consume (ADR-0005's consequence: one join, not one per caller).

``fit_inference_model()`` is the primary analysis SPRINT-PLAN.md Section
3.6 and ADR-0006 call for: a statsmodels logistic regression of
detection-correct on identifiability_score, with confidence intervals.
It restricts to the systematic injection trials and derives its binary
target from the judge's affirmative-detection field alone; see ADR-0020
for why.

Every public function here refuses to run before
``data/results/judge_validated.flag`` (REQ-8) exists -- CLAUDE.md's rule
against treating an unvalidated judge as ground truth, enforced at each
entry point rather than trusted to have been checked upstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from prism.features import load_feature_audit

DEFAULT_TRIALS_PATH = "data/trials/trials.jsonl"
DEFAULT_AUDIT_PATH = "data/audit/features.csv"
DEFAULT_VALIDATION_FLAG_PATH = "data/results/judge_validated.flag"

DETECTION_TARGET_COLUMN = "detection_correct"
INFERENCE_COVARIATES = ("identifiability_score", "decoder_norm", "activation_frequency", "strength")


class JudgeNotValidatedError(RuntimeError):
    """``data/results/judge_validated.flag`` is missing.

    Raised instead of silently running the analysis, per CLAUDE.md's rule
    against treating judge_scores as ground truth before a human has
    actually reviewed ``judge.validate_judge_subsample()``'s output.
    """


def _require_judge_validated(validation_flag_path: "str | Path") -> None:
    if not Path(validation_flag_path).exists():
        raise JudgeNotValidatedError(
            f"{validation_flag_path} does not exist -- run "
            "judge.validate_judge_subsample() and judge.write_validation_flag() "
            "(REQ-8) before trusting judge_scores as ground truth for any "
            "downstream analysis"
        )


def _read_trials(path: "str | Path") -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- REQ-9: the analysis table ----------------------------------------------


def build_analysis_table(
    trials_path: "str | Path" = DEFAULT_TRIALS_PATH,
    audit_path: "str | Path" = DEFAULT_AUDIT_PATH,
    *,
    validation_flag_path: "str | Path" = DEFAULT_VALIDATION_FLAG_PATH,
) -> pd.DataFrame:
    """Join every non-excluded trial onto its feature's audit row.

    Excluded trials (REQ-8's judge-refusal exclusions) are dropped here,
    not carried forward with null judge fields for every downstream caller
    to filter out again. Each trial's ``judge_scores`` dict is flattened
    into ``judge_*`` columns; ``model_response`` is dropped from the
    regression-ready table (it is free text, not a covariate) but
    ``trial_id`` stays, so a row can always be traced back to its full
    transcript in ``trials_path`` if needed.

    Raises ``JudgeNotValidatedError`` before touching a row of data if
    ``validation_flag_path`` is missing, and ``ValueError`` if a
    non-excluded trial has no judge score yet, or if a trial's
    ``feature_id`` has no match in ``audit_path`` -- both mean an upstream
    invariant broke, not something to paper over with a silent drop.
    """
    _require_judge_validated(validation_flag_path)

    records = _read_trials(trials_path)
    trials = pd.DataFrame.from_records(records)

    excluded = trials["excluded"]
    pending = trials.loc[~excluded, "judge_scores"].isna()
    if pending.any():
        bad_ids = trials.loc[(~excluded) & trials["judge_scores"].isna(), "trial_id"].tolist()
        raise ValueError(
            f"{trials_path} has non-excluded trial(s) with no judge_scores yet: "
            f"{bad_ids}. Run judge.score_all_pending() first."
        )

    trials = trials.loc[~excluded].reset_index(drop=True)
    judge_columns = pd.json_normalize(trials["judge_scores"].tolist()).add_prefix("judge_")
    trials = pd.concat([trials.drop(columns=["judge_scores", "model_response"]), judge_columns], axis=1)

    audit = load_feature_audit(audit_path)
    merged = trials.merge(audit, on="feature_id", how="left", validate="many_to_one")
    unmatched = sorted(merged.loc[merged["identifiability_score"].isna(), "feature_id"].unique())
    if unmatched:
        raise ValueError(
            f"{trials_path} has trial(s) for feature_id(s) {unmatched} with no "
            f"match in {audit_path} -- the audit table may be stale relative to "
            "the sampled features"
        )
    return merged


def _detection_subset(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the systematic injection trials the primary analyses run over.

    Baseline trials (REQ-7) carry ``strength = None`` since nothing was
    injected, and control trials are graded against a different judge
    schema entirely (``judge_affirmative``/``judge_coherent``, no
    ``judge_detected`` field, since there is no injected concept to
    detect). Neither has both a real strength value and a detection
    judgment, so neither belongs in this fit. See ADR-0020.
    """
    subset = trials_df.loc[trials_df["prompt_type"] == "detection"].copy()
    if subset.empty:
        raise ValueError("trials_df has no prompt_type == 'detection' rows to analyze")
    if subset["strength"].isna().any():
        raise ValueError(
            "detection-type trial(s) with a null strength value -- an inject.py/"
            "runner.py invariant broke"
        )
    return subset


def _add_detection_target(subset: pd.DataFrame) -> pd.DataFrame:
    """Derive the binary detection-correct target from the judge's own
    affirmative-detection field (ADR-0020).

    Lindsey's naming-accuracy criterion (``judge_concept_identified``) is
    not folded into this target: REQ-8's validated run recorded zero real
    naming turns across the dataset (every naming-eligible trial's
    ``judge_concept_identified`` is null), so requiring it would leave the
    target undefined everywhere, not merely rare.
    """
    subset = subset.copy()
    subset[DETECTION_TARGET_COLUMN] = subset["judge_detected"].astype(bool)
    return subset


# --- REQ-9: primary inference -----------------------------------------------


def fit_inference_model(
    trials_df: pd.DataFrame,
    *,
    validation_flag_path: "str | Path" = DEFAULT_VALIDATION_FLAG_PATH,
) -> dict[str, Any]:
    """Logistic regression of detection-correct on identifiability_score,
    with decoder_norm, activation_frequency, and strength as covariates
    (SPRINT-PLAN.md Section 3.6, ADR-0006).

    Covariates are z-scored before fitting -- their raw scales span several
    orders of magnitude (activation_frequency around 1e-3, decoder_norm
    around 1, strength up to 8), which the solver is sensitive to under
    this dataset's rare-event class imbalance (ADR-0020). Coefficients are
    reported in per-standard-deviation units, not raw covariate units; the
    standardization used is returned alongside them so a raw-unit estimate
    can always be recovered.

    Returns point estimates and 95% CIs for every covariate, plus whether
    the fit actually converged and how many trials/detections it is built
    on. Never raises on a convergence failure -- a failed or degenerate fit
    is itself a real, reportable result on data this sparse, not a bug to
    hide.
    """
    _require_judge_validated(validation_flag_path)

    subset = _add_detection_target(_detection_subset(trials_df))
    n_trials = len(subset)
    n_detections = int(subset[DETECTION_TARGET_COLUMN].sum())

    covariates = list(INFERENCE_COVARIATES)
    x_raw = subset[covariates].astype(float)
    x_mean = x_raw.mean()
    x_std = x_raw.std(ddof=0)
    x_standardized = (x_raw - x_mean) / x_std
    x_design = sm.add_constant(x_standardized, has_constant="add")
    y = subset[DETECTION_TARGET_COLUMN].astype(int)

    converged = False
    convergence_note = ""
    coefficients: dict[str, dict[str, float]] = {}
    try:
        fit_result = sm.Logit(y, x_design).fit(disp=0)
        converged = bool(fit_result.mle_retvals.get("converged", False))
        conf_int = fit_result.conf_int(alpha=0.05)
        for name in x_design.columns:
            coefficients[name] = {
                "estimate": float(fit_result.params[name]),
                "ci_low": float(conf_int.loc[name, 0]),
                "ci_high": float(conf_int.loc[name, 1]),
            }
        if not converged:
            convergence_note = "statsmodels reported mle_retvals['converged'] == False"
    except (PerfectSeparationError, np.linalg.LinAlgError) as exc:
        convergence_note = f"{type(exc).__name__}: {exc}"

    return {
        "n_trials": n_trials,
        "n_detections": n_detections,
        "covariates": covariates,
        "standardization": {
            name: {"mean": float(x_mean[name]), "std": float(x_std[name])} for name in covariates
        },
        "converged": converged,
        "convergence_note": convergence_note,
        "coefficients": coefficients,
    }
