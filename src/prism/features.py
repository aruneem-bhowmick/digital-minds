"""Stratified feature sampling from the identifiability audit (REQ-2).

Reads the static, read-only ``data/audit/features.csv`` table (assembled by
``prism.audit_build`` per ADR-0011) and draws a sample of SAE features
stratified into low/medium/high identifiability tertiles, balanced on
decoder norm and activation frequency so neither covariate tracks
identifiability by accident (SPRINT-PLAN.md §3.2).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_COLUMNS = ("feature_id", "identifiability_score", "decoder_norm", "activation_frequency")
TERTILE_LABELS = ("low", "medium", "high")
DEFAULT_N_TOTAL = 40


def load_feature_audit(path: str) -> pd.DataFrame:
    """Load and validate ``data/audit/features.csv``.

    Every downstream function trusts this table's columns are present,
    unique per feature, and finite -- a malformed audit table needs to
    fail here, not propagate silently into the sampler or, later, the
    regression.
    """
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if df["feature_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate feature_id values")
    covariates = df[list(REQUIRED_COLUMNS)]
    if covariates.isna().any().any():
        raise ValueError(f"{path} has missing values in a required column")
    if not np.isfinite(covariates.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} has non-finite values in a required column")
    return df.sort_values("feature_id").reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n_total: int = DEFAULT_N_TOTAL, seed: int = 0) -> pd.DataFrame:
    """Sample n_total features, stratified into identifiability tertiles and
    balanced within each tertile on decoder_norm and activation_frequency.

    Tertile boundaries and the norm/frequency covariate bins are both
    computed over the full population in ``df``, not the sample, so a
    tertile's sample composition can be judged against the population it
    was drawn from. Returns the sampled rows with two added columns:
    ``identifiability_tertile`` (the stratum provenance) and
    ``covariate_bin`` (which norm x frequency cell the feature fell into,
    the mechanism that keeps the balance).
    """
    if n_total < 3:
        raise ValueError(
            f"n_total must be at least 3 to draw from all three tertiles, got {n_total}"
        )

    working = df.copy()
    working["identifiability_tertile"] = _assign_tertiles(working["identifiability_score"])
    working["covariate_bin"] = _covariate_bin_labels(working)

    rng = np.random.default_rng(seed)
    counts = _tertile_counts(n_total)

    sampled_parts = []
    for tertile in TERTILE_LABELS:
        stratum = working[working["identifiability_tertile"] == tertile]
        count = counts[tertile]
        if len(stratum) < count:
            raise ValueError(
                f"tertile {tertile!r} has only {len(stratum)} features, fewer than "
                f"the {count} requested; lower n_total or check the audit table"
            )
        sampled_parts.append(_balanced_within_stratum(stratum, count, rng))

    return pd.concat(sampled_parts, ignore_index=True)


def check_covariate_balance(sampled_df: pd.DataFrame) -> pd.DataFrame:
    """Report per-tertile mean and standard deviation of decoder_norm and
    activation_frequency on a sampled set.

    A diagnostic, not a gate: this function does not raise on an imbalance,
    it only surfaces one. Its output needs to actually be read before the
    sample is trusted, per CLAUDE.md's rule against treating a clean script
    exit as proof of correctness.
    """
    if "identifiability_tertile" not in sampled_df.columns:
        raise ValueError("sampled_df must carry identifiability_tertile (from stratified_sample)")
    summary = sampled_df.groupby("identifiability_tertile", observed=True).agg(
        n=("feature_id", "count"),
        decoder_norm_mean=("decoder_norm", "mean"),
        decoder_norm_std=("decoder_norm", "std"),
        activation_frequency_mean=("activation_frequency", "mean"),
        activation_frequency_std=("activation_frequency", "std"),
    )
    return summary.reindex(TERTILE_LABELS).reset_index()


def _assign_tertiles(scores: pd.Series) -> pd.Series:
    """Split a score column into low/medium/high by rank-based tercile.

    Ranking before binning (rather than binning the raw values) keeps bin
    sizes close to equal even when many features share the same score, and
    ``method="first"`` breaks ties by original row order so the split is
    deterministic rather than dependent on qcut's own tie-handling.
    """
    ranks = scores.rank(method="first")
    return pd.qcut(ranks, q=3, labels=list(TERTILE_LABELS))


def _covariate_bin_labels(df: pd.DataFrame) -> pd.Series:
    """Return a joint decoder_norm x activation_frequency bin label per row."""
    norm_bin = pd.qcut(df["decoder_norm"].rank(method="first"), q=3, labels=list(TERTILE_LABELS))
    freq_bin = pd.qcut(df["activation_frequency"].rank(method="first"), q=3, labels=list(TERTILE_LABELS))
    return norm_bin.astype(str) + "_" + freq_bin.astype(str)


def _tertile_counts(n_total: int) -> dict[str, int]:
    """Split n_total as evenly as possible across the three tertiles."""
    base, remainder = divmod(n_total, 3)
    counts = dict.fromkeys(TERTILE_LABELS, base)
    for tertile in TERTILE_LABELS[:remainder]:
        counts[tertile] += 1
    return counts


def _balanced_within_stratum(stratum: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample `count` rows from one identifiability tertile, spread across
    covariate bins rather than drawn uniformly, so the tertile's sample
    doesn't cluster in one corner of the norm/frequency space.
    """
    bin_sizes = stratum.groupby("covariate_bin", observed=True).size().to_dict()
    allocation = _round_robin_allocation(bin_sizes, count, rng)

    parts = []
    for covariate_bin, n in allocation.items():
        if n == 0:
            continue
        bin_rows = stratum[stratum["covariate_bin"] == covariate_bin]
        parts.append(bin_rows.sample(n=n, random_state=rng))
    return pd.concat(parts, ignore_index=False)


def _round_robin_allocation(
    capacity_by_bin: dict[str, int], total: int, rng: np.random.Generator
) -> dict[str, int]:
    """Allocate `total` picks across bins as evenly as possible, capped by
    each bin's capacity. Each round gives one pick to every bin that still
    has capacity, in a freshly randomized order, so when the remainder
    doesn't divide evenly no bin is systematically favored for the extra.
    """
    remaining = dict(capacity_by_bin)
    allocation = dict.fromkeys(capacity_by_bin, 0)
    picked = 0
    while picked < total:
        eligible = [b for b, capacity in remaining.items() if capacity > 0]
        if not eligible:
            break
        for b in rng.permutation(eligible):
            if picked >= total:
                break
            allocation[b] += 1
            remaining[b] -= 1
            picked += 1
    return allocation


def main() -> None:
    """CLI entry point: python -m prism.features --config configs/experiment.yaml."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--audit-csv", default="data/audit/features.csv")
    parser.add_argument("--output", default="data/results/sampled_features.csv")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    features_config = config.get("features", {})
    n_total = features_config.get("n_total", DEFAULT_N_TOTAL)
    seed = features_config.get("sample_seed", 0)

    audit = load_feature_audit(args.audit_csv)
    sample = stratified_sample(audit, n_total=n_total, seed=seed)
    balance = check_covariate_balance(sample)

    print(balance.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
