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
    unique per feature, finite, and within each covariate's valid range --
    a malformed audit table needs to fail here, not propagate silently
    into the sampler or, later, the regression.
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
    _validate_covariate_ranges(df, path)
    return df.sort_values("feature_id").reset_index(drop=True)


def _validate_covariate_ranges(df: pd.DataFrame, path: str) -> None:
    """Check each covariate against the range its own definition guarantees.

    identifiability_score is a mutual-coherence value (feature_coherence()
    in sae-bounding): the max absolute inner product between unit-normalized
    decoder atoms, bounded to [0, 1] by construction. decoder_norm is a
    vector norm, never negative. activation_frequency is a rate over
    tokens, bounded to [0, 1]. A value outside these ranges means the audit
    table is corrupted or was assembled from the wrong source, not that the
    definitions above need loosening.
    """
    if not df["identifiability_score"].between(0, 1).all():
        raise ValueError(f"{path} has identifiability_score values outside [0, 1]")
    if (df["decoder_norm"] < 0).any():
        raise ValueError(f"{path} has negative decoder_norm values")
    if not df["activation_frequency"].between(0, 1).all():
        raise ValueError(f"{path} has activation_frequency values outside [0, 1]")


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
    bins = sorted(working["covariate_bin"].unique())

    rng = np.random.default_rng(seed)
    counts = _tertile_counts(n_total)
    targets_by_count = _nested_bin_targets(bins, sorted(set(counts.values())), rng)

    sampled_parts = []
    for tertile in TERTILE_LABELS:
        stratum = working[working["identifiability_tertile"] == tertile]
        count = counts[tertile]
        if len(stratum) < count:
            raise ValueError(
                f"tertile {tertile!r} has only {len(stratum)} features, fewer than "
                f"the {count} requested; lower n_total or check the audit table"
            )
        sampled_parts.append(
            _balanced_within_stratum(stratum, targets_by_count[count], rng, tertile=tertile)
        )

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


def _nested_bin_targets(
    bins: list[str], counts: list[int], rng: np.random.Generator
) -> dict[int, dict[str, int]]:
    """Build a per-bin target for each distinct tertile count, so a larger
    count's target is always the smaller count's target plus additional
    picks layered on top -- never an independently re-randomized
    composition for the same set of bins.

    _tertile_counts() never produces more than two distinct values,
    differing by exactly 1 (n_total's remainder over 3 tertiles), so in
    practice this means every bin's target across tertiles differs by at
    most 1. The construction generalizes to any number of distinct counts:
    sorted ascending, each target starts from the previous (smaller)
    count's target and adds an evenly-split allocation of the difference,
    so no bin can ever have a *smaller* target at a larger count.
    """
    ordered = sorted(counts)
    targets: dict[int, dict[str, int]] = {}
    targets[ordered[0]] = _round_robin_allocation(dict.fromkeys(bins, ordered[0]), ordered[0], rng)
    for previous, count in zip(ordered, ordered[1:]):
        extra = count - previous
        extra_allocation = _round_robin_allocation(dict.fromkeys(bins, extra), extra, rng)
        targets[count] = {b: targets[previous][b] + extra_allocation[b] for b in bins}
    return targets


def _balanced_within_stratum(
    stratum: pd.DataFrame, target: dict[str, int], rng: np.random.Generator, *, tertile: str
) -> pd.DataFrame:
    """Draw exactly `target[bin]` rows from each covariate bin in one tertile.

    `target` is the same shared composition every tertile is held to
    (see _shared_bin_targets), not something this stratum gets to
    renegotiate. If identifiability correlates with decoder_norm or
    activation_frequency strongly enough that this tertile can't supply a
    bin's target count, that's reported as a clear error rather than
    silently drawing more from whichever bins this tertile happens to have
    available -- the latter would quietly reintroduce the exact
    identifiability/covariate confound this balancing exists to prevent.
    """
    parts = []
    for covariate_bin, n in target.items():
        if n == 0:
            continue
        bin_rows = stratum[stratum["covariate_bin"] == covariate_bin]
        if len(bin_rows) < n:
            raise ValueError(
                f"tertile {tertile!r} has only {len(bin_rows)} features in "
                f"covariate bin {covariate_bin!r}, fewer than the {n} needed to "
                "match the shared cross-tertile target; lower n_total, or check "
                "whether identifiability correlates too strongly with "
                "decoder_norm/activation_frequency in this audit table"
            )
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
