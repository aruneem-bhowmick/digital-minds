"""Regression and AUC comparison (REQ-9).

``build_analysis_table()`` joins every non-excluded trial in
``data/trials/trials.jsonl`` onto its feature's identifiability score,
decoder norm, and activation frequency, producing the single
regression-ready table both analyses in this module consume (ADR-0005's
consequence: one join, not one per caller). ``data/trials/trials.jsonl``
can hold more than one model's trials (REQ-11 added Gemma Scope
alongside Pythia), and each model's own audit table indexes
``feature_id`` independently starting at 0 -- so the join key is
``(model_name, feature_id)``, not ``feature_id`` alone, and
``audit_paths`` maps each trial-bearing model's name to its own audit
CSV rather than pointing at a single default.

``fit_inference_model()`` and ``compare_classifiers()`` are the two
analyses SPRINT-PLAN.md Section 3.6 and ADR-0006 call for: a statsmodels
logistic regression with confidence intervals, and a scikit-learn AUC
comparison against the obvious decoder-norm confound. Both restrict to
the systematic injection trials and derive their binary target from the
judge's affirmative-detection field alone; see ADR-0020 for why.

Every public function here refuses to run before every relevant model's
own ``judge_validated*.flag`` (REQ-8) exists -- CLAUDE.md's rule against
treating an unvalidated judge as ground truth, enforced at each entry
point rather than trusted to have been checked upstream. A caller
analyzing more than one model's trials passes every one of that model's
flag paths, not just one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import MissingDataError, PerfectSeparationError

from prism.features import load_feature_audit

DEFAULT_TRIALS_PATH = "data/trials/trials.jsonl"
DEFAULT_AUDIT_PATH = "data/audit/features.csv"
DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEFAULT_VALIDATION_FLAG_PATH = "data/results/judge_validated.flag"

DETECTION_TARGET_COLUMN = "detection_correct"
INFERENCE_COVARIATES = ("identifiability_score", "decoder_norm", "activation_frequency", "strength")


class JudgeNotValidatedError(RuntimeError):
    """``data/results/judge_validated.flag`` is missing.

    Raised instead of silently running the analysis, per CLAUDE.md's rule
    against treating judge_scores as ground truth before a human has
    actually reviewed ``judge.validate_judge_subsample()``'s output.
    """


def _require_judge_validated(validation_flag_path: "str | Path | list[str | Path]") -> None:
    """Accepts one path, or a list of paths when a caller is analyzing more
    than one model's trials together (REQ-11) -- every one of them must
    exist, since a combined-model table is only as validated as its
    least-validated model.
    """
    paths = [validation_flag_path] if isinstance(validation_flag_path, (str, Path)) else list(validation_flag_path)
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise JudgeNotValidatedError(
            f"{missing} do not exist -- run judge.validate_judge_subsample() and "
            "judge.write_validation_flag() (REQ-8) for every model this analysis "
            "covers before trusting judge_scores as ground truth"
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
    *,
    audit_paths: dict[str, "str | Path"],
    validation_flag_path: "str | Path | list[str | Path]" = DEFAULT_VALIDATION_FLAG_PATH,
) -> pd.DataFrame:
    """Join every non-excluded trial onto its feature's audit row.

    Excluded trials (REQ-8's judge-refusal exclusions) are dropped here,
    not carried forward with null judge fields for every downstream caller
    to filter out again. Each trial's ``judge_scores`` dict is flattened
    into ``judge_*`` columns; ``model_response`` is dropped from the
    regression-ready table (it is free text, not a covariate) but
    ``trial_id`` stays, so a row can always be traced back to its full
    transcript in ``trials_path`` if needed.

    ``audit_paths`` maps each model_name present in ``trials_path`` to that
    model's own audit CSV (REQ-11): ``feature_id`` is indexed 0..N-1
    independently per SAE checkpoint, so two different models' dictionaries
    can share the same ``feature_id`` values without meaning the same
    feature -- the join key is ``(model_name, feature_id)``, not
    ``feature_id`` alone, and there is no single sensible default audit
    table once more than one model's trials can appear in the same file.

    Raises ``JudgeNotValidatedError`` before touching a row of data if any
    path in ``validation_flag_path`` is missing, and ``ValueError`` if a
    non-excluded trial has no judge score yet, or if a trial's
    ``(model_name, feature_id)`` has no match in ``audit_paths`` -- both
    mean an upstream invariant broke, not something to paper over with a
    silent drop.
    """
    _require_judge_validated(validation_flag_path)

    records = _read_trials(trials_path)
    trials = pd.DataFrame.from_records(records)

    excluded = trials["excluded"]
    unscored = (~excluded) & trials["judge_scores"].isna()
    if unscored.any():
        bad_ids = trials.loc[unscored, "trial_id"].tolist()
        raise ValueError(
            f"{trials_path} has non-excluded trial(s) with no judge_scores yet: "
            f"{bad_ids}. Run judge.score_all_pending() first."
        )

    trials = trials.loc[~excluded].reset_index(drop=True)
    judge_columns = pd.json_normalize(trials["judge_scores"].tolist()).add_prefix("judge_")
    trials = pd.concat([trials.drop(columns=["judge_scores", "model_response"]), judge_columns], axis=1)

    audit_frames = []
    for model_name, audit_path in audit_paths.items():
        audit = load_feature_audit(audit_path).copy()
        audit.insert(0, "model_name", model_name)
        audit_frames.append(audit)
    combined_audit = pd.concat(audit_frames, ignore_index=True)

    merged = trials.merge(
        combined_audit, on=["model_name", "feature_id"], how="left", validate="many_to_one", indicator=True
    )
    unmatched_rows = merged.loc[merged["_merge"] == "left_only", ["model_name", "feature_id"]].drop_duplicates()
    if not unmatched_rows.empty:
        unmatched = sorted(
            (str(row.model_name), int(row.feature_id)) for row in unmatched_rows.itertuples(index=False)
        )
        raise ValueError(
            f"{trials_path} has trial(s) for (model_name, feature_id) {unmatched} with no "
            f"match in audit_paths -- the audit table may be stale relative to the sampled "
            "features, or audit_paths is missing an entry for one of the trial file's models"
        )
    return merged.drop(columns=["_merge"])


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

    Raises ``ValueError`` if ``judge_detected`` is missing or contains a
    null value on a row this function is asked to derive a target for.
    ADR-0018's structured judge output schema guarantees it is always
    populated on a scored trial, but that guarantee lives in judge.py's
    prompt/schema, not in a type system -- a silent break there should
    not quietly become a wrong boolean here.
    """
    if "judge_detected" not in subset.columns:
        raise ValueError("subset is missing the judge_detected column")
    if subset["judge_detected"].isna().any():
        bad_ids = subset.loc[subset["judge_detected"].isna(), "trial_id"].tolist()
        raise ValueError(f"trial(s) with a null judge_detected value: {bad_ids}")

    subset = subset.copy()
    subset[DETECTION_TARGET_COLUMN] = subset["judge_detected"].astype(bool)
    return subset


# --- REQ-9: primary inference -----------------------------------------------


def fit_inference_model(
    trials_df: pd.DataFrame,
    *,
    validation_flag_path: "str | Path | list[str | Path]" = DEFAULT_VALIDATION_FLAG_PATH,
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
    standardization = {name: {"mean": float(x_mean[name]), "std": float(x_std[name])} for name in covariates}

    degenerate = [
        name for name in covariates if not np.isfinite(x_std[name]) or np.isclose(x_std[name], 0)
    ]
    converged = False
    convergence_note = ""
    coefficients: dict[str, dict[str, float]] = {}

    if degenerate:
        # Standardizing would divide by zero (or by a non-finite or
        # near-zero std -- floating-point noise around an otherwise-constant
        # covariate, not a real distribution) and hand statsmodels a
        # NaN/inf/enormous design matrix -- which raises whichever internal
        # exception happens to fire first (MissingDataError, LinAlgError,
        # ...) or silently "converges" to a numerically meaningless
        # coefficient. A degenerate covariate in this trial subset is itself
        # the reportable finding: there's no real variation for the fit to
        # attribute an effect to.
        convergence_note = (
            f"covariate(s) {degenerate} have zero, near-zero, or non-finite variance in this "
            "trial subset"
        )
    else:
        x_standardized = (x_raw - x_mean) / x_std
        x_design = sm.add_constant(x_standardized, has_constant="add")
        y = subset[DETECTION_TARGET_COLUMN].astype(int)
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
        except (PerfectSeparationError, MissingDataError, np.linalg.LinAlgError) as exc:
            convergence_note = f"{type(exc).__name__}: {exc}"

    return {
        "n_trials": n_trials,
        "n_detections": n_detections,
        "covariates": covariates,
        "standardization": standardization,
        "converged": converged,
        "convergence_note": convergence_note,
        "coefficients": coefficients,
    }


# --- REQ-9: identifiability vs. the norm confound ---------------------------


def compare_classifiers(
    trials_df: pd.DataFrame,
    *,
    validation_flag_path: "str | Path | list[str | Path]" = DEFAULT_VALIDATION_FLAG_PATH,
) -> pd.DataFrame:
    """AUC comparison between a classifier using only identifiability_score
    and one using only decoder_norm (SPRINT-PLAN.md Section 3.6, ADR-0006).

    Tests whether identifiability adds predictive value beyond the obvious
    norm confound. Uses the same trial subset and detection-correct target
    as ``fit_inference_model()``, scored in-sample: the dataset's positive
    class is too sparse to support a held-out split without either fold
    containing zero positives, so a train/test AUC would not measure
    anything a held-out split is meant to measure. Returns one row per
    classifier.
    """
    _require_judge_validated(validation_flag_path)

    subset = _add_detection_target(_detection_subset(trials_df))
    y = subset[DETECTION_TARGET_COLUMN].astype(int).to_numpy()
    n_trials = len(subset)
    n_detections = int(y.sum())

    single_class = n_detections in (0, n_trials)
    rows = []
    for covariate in ("identifiability_score", "decoder_norm"):
        if single_class:
            # LogisticRegression.fit() raises on a single-class target
            # ("needs samples of at least 2 classes"), so there is nothing
            # to fit -- an undefined AUC is the honest result, not a crash.
            auc = float("nan")
        else:
            x = subset[[covariate]].astype(float).to_numpy()
            classifier = LogisticRegression()
            classifier.fit(x, y)
            scores = classifier.predict_proba(x)[:, 1]
            auc = float(roc_auc_score(y, scores))
        rows.append({"classifier": covariate, "auc": auc, "n_trials": n_trials, "n_detections": n_detections})

    return pd.DataFrame(rows)


# --- CLI ---------------------------------------------------------------------


def _git_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    """CLI entry point: ``python -m prism.stats``.

    Builds the analysis table, runs both analyses, and writes all three
    outputs to ``data/results/`` -- nothing here is hand-edited, everything
    is regenerable by re-running this command.
    """
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-path", default=DEFAULT_TRIALS_PATH)
    parser.add_argument(
        "--audit-path",
        action="append",
        default=[],
        metavar="MODEL_NAME=PATH",
        help="one model's audit CSV, as MODEL_NAME=PATH; repeatable for a combined-model "
        "analysis (REQ-11). Defaults to Pythia's own audit table alone if omitted entirely.",
    )
    parser.add_argument(
        "--validation-flag-path",
        action="append",
        default=[],
        help="a judge_validated*.flag path; repeatable, one per model this analysis covers. "
        "Defaults to Pythia's own flag alone if omitted entirely.",
    )
    parser.add_argument("--analysis-table-path", default="data/results/analysis_table.csv")
    parser.add_argument("--regression-results-path", default="data/results/regression_results.json")
    parser.add_argument("--auc-comparison-path", default="data/results/auc_comparison.csv")
    args = parser.parse_args()

    audit_paths = {
        model_name: path for model_name, path in (entry.split("=", 1) for entry in args.audit_path)
    } or {DEFAULT_MODEL_NAME: DEFAULT_AUDIT_PATH}
    validation_flag_paths = args.validation_flag_path or [DEFAULT_VALIDATION_FLAG_PATH]

    table = build_analysis_table(
        args.trials_path, audit_paths=audit_paths, validation_flag_path=validation_flag_paths
    )
    analysis_table_path = Path(args.analysis_table_path)
    analysis_table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(analysis_table_path, index=False)
    print(f"wrote {analysis_table_path} ({len(table)} rows)")

    regression = fit_inference_model(table, validation_flag_path=validation_flag_paths)
    regression_record = {
        **regression,
        "trials_path": str(args.trials_path),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    regression_results_path = Path(args.regression_results_path)
    regression_results_path.parent.mkdir(parents=True, exist_ok=True)
    regression_results_path.write_text(json.dumps(regression_record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {regression_results_path} (converged={regression['converged']})")

    auc_table = compare_classifiers(table, validation_flag_path=validation_flag_paths)
    auc_table = auc_table.assign(
        trials_path=str(args.trials_path),
        git_commit=_git_commit(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    auc_comparison_path = Path(args.auc_comparison_path)
    auc_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    auc_table.to_csv(auc_comparison_path, index=False)
    print(f"wrote {auc_comparison_path}")
    print(auc_table.to_string(index=False))


if __name__ == "__main__":
    main()
