"""Regression and AUC comparison (REQ-9).

``build_analysis_table()`` joins every non-excluded trial in
``data/trials/trials.jsonl`` onto its feature's identifiability score,
decoder norm, and activation frequency in ``data/audit/features.csv``,
producing the single regression-ready table both analyses in this module
consume (ADR-0005's consequence: one join, not one per caller).

Every public function here refuses to run before
``data/results/judge_validated.flag`` (REQ-8) exists -- CLAUDE.md's rule
against treating an unvalidated judge as ground truth, enforced at each
entry point rather than trusted to have been checked upstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from prism.features import load_feature_audit

DEFAULT_TRIALS_PATH = "data/trials/trials.jsonl"
DEFAULT_AUDIT_PATH = "data/audit/features.csv"
DEFAULT_VALIDATION_FLAG_PATH = "data/results/judge_validated.flag"


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
