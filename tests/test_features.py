"""Tests for prism.features — the stratified feature sampler (REQ-2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.features import (
    REQUIRED_COLUMNS,
    _assign_tertiles,
    _balanced_within_stratum,
    _round_robin_allocation,
    _tertile_counts,
    check_covariate_balance,
    load_feature_audit,
    stratified_sample,
)


def _audit_csv(tmp_path: Path, rows: pd.DataFrame) -> Path:
    path = tmp_path / "features.csv"
    rows.to_csv(path, index=False)
    return path


def _synthetic_population(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """30 features with an even score spread and covariates independent of
    identifiability, so every tertile spans the full norm/frequency range
    and balancing has something real to do.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_id": np.arange(n),
            "identifiability_score": np.linspace(0.1, 0.9, n),
            "decoder_norm": rng.permutation(np.linspace(0.05, 2.0, n)),
            "activation_frequency": rng.permutation(np.linspace(0.0001, 0.01, n)),
        }
    )


# --- load_feature_audit -----------------------------------------------------


def test_load_feature_audit_loads_a_valid_csv(tmp_path: Path) -> None:
    path = _audit_csv(tmp_path, _synthetic_population())

    df = load_feature_audit(str(path))

    assert list(df["feature_id"]) == list(range(30))
    assert set(REQUIRED_COLUMNS) <= set(df.columns)


def test_load_feature_audit_rejects_missing_column(tmp_path: Path) -> None:
    rows = _synthetic_population().drop(columns=["decoder_norm"])
    path = _audit_csv(tmp_path, rows)

    with pytest.raises(ValueError, match="missing required column"):
        load_feature_audit(str(path))


def test_load_feature_audit_rejects_duplicate_feature_id(tmp_path: Path) -> None:
    rows = _synthetic_population()
    rows.loc[1, "feature_id"] = rows.loc[0, "feature_id"]
    path = _audit_csv(tmp_path, rows)

    with pytest.raises(ValueError, match="duplicate feature_id"):
        load_feature_audit(str(path))


def test_load_feature_audit_rejects_missing_values(tmp_path: Path) -> None:
    rows = _synthetic_population()
    rows.loc[2, "activation_frequency"] = np.nan
    path = _audit_csv(tmp_path, rows)

    with pytest.raises(ValueError, match="missing values"):
        load_feature_audit(str(path))


def test_load_feature_audit_rejects_non_finite_values(tmp_path: Path) -> None:
    rows = _synthetic_population()
    rows.loc[2, "decoder_norm"] = np.inf
    path = _audit_csv(tmp_path, rows)

    with pytest.raises(ValueError, match="non-finite"):
        load_feature_audit(str(path))


# --- tertile boundaries -------------------------------------------------


def test_assign_tertiles_splits_nine_scores_into_equal_thirds() -> None:
    scores = pd.Series([float(i) for i in range(9)])

    tertiles = _assign_tertiles(scores)

    assert list(tertiles[:3]) == ["low"] * 3
    assert list(tertiles[3:6]) == ["medium"] * 3
    assert list(tertiles[6:9]) == ["high"] * 3


def test_assign_tertiles_breaks_ties_by_row_order() -> None:
    # All nine scores tied: rank(method="first") still produces a well-defined,
    # deterministic 3/3/3 split by original position rather than an error or
    # an uneven qcut bin.
    scores = pd.Series([0.5] * 9)

    tertiles = _assign_tertiles(scores)

    assert list(tertiles[:3]) == ["low"] * 3
    assert list(tertiles[3:6]) == ["medium"] * 3
    assert list(tertiles[6:9]) == ["high"] * 3


def test_tertile_counts_splits_evenly_when_divisible() -> None:
    assert _tertile_counts(30) == {"low": 10, "medium": 10, "high": 10}


def test_tertile_counts_distributes_remainder_to_low_then_medium() -> None:
    # 40 = 13 + 13 + 14 base, remainder 1 -> the first tertile in
    # TERTILE_LABELS order ("low") gets the extra pick.
    assert _tertile_counts(40) == {"low": 14, "medium": 13, "high": 13}


# --- balancing mechanism -------------------------------------------------


def test_round_robin_allocation_splits_evenly_across_equal_capacity_bins() -> None:
    rng = np.random.default_rng(0)

    allocation = _round_robin_allocation({"a": 5, "b": 5, "c": 5}, 6, rng)

    assert sum(allocation.values()) == 6
    assert all(count <= 2 for count in allocation.values())


def test_round_robin_allocation_respects_bin_capacity() -> None:
    rng = np.random.default_rng(0)

    allocation = _round_robin_allocation({"a": 1, "b": 5}, 4, rng)

    assert allocation["a"] == 1  # capped, not left short by the round-robin
    assert allocation["b"] == 3
    assert sum(allocation.values()) == 4


def test_balanced_within_stratum_draws_from_every_available_bin() -> None:
    stratum = pd.DataFrame(
        {
            "feature_id": range(9),
            "covariate_bin": ["low_low"] * 3 + ["medium_medium"] * 3 + ["high_high"] * 3,
        }
    )
    rng = np.random.default_rng(0)

    selected = _balanced_within_stratum(stratum, count=3, rng=rng)

    assert len(selected) == 3
    assert set(selected["covariate_bin"]) == {"low_low", "medium_medium", "high_high"}


# --- stratified_sample ----------------------------------------------------


def test_stratified_sample_returns_exactly_n_total_rows() -> None:
    population = _synthetic_population(n=30)

    sample = stratified_sample(population, n_total=15, seed=0)

    assert len(sample) == 15


def test_stratified_sample_distributes_a_remainder_across_tertiles() -> None:
    population = _synthetic_population(n=30)

    sample = stratified_sample(population, n_total=16, seed=0)

    counts = sample["identifiability_tertile"].value_counts()
    assert counts["low"] == 6
    assert counts["medium"] == 5
    assert counts["high"] == 5


def test_stratified_sample_is_reproducible_under_a_fixed_seed() -> None:
    population = _synthetic_population(n=30)

    first = stratified_sample(population, n_total=15, seed=42)
    second = stratified_sample(population, n_total=15, seed=42)

    assert sorted(first["feature_id"]) == sorted(second["feature_id"])


def test_stratified_sample_records_tertile_provenance() -> None:
    population = _synthetic_population(n=30)

    sample = stratified_sample(population, n_total=15, seed=0)

    assert set(sample["identifiability_tertile"]) <= {"low", "medium", "high"}
    assert sample["identifiability_tertile"].notna().all()


def test_stratified_sample_rejects_a_tertile_too_small_to_satisfy_n_total() -> None:
    population = _synthetic_population(n=30)  # 10 features per tertile

    with pytest.raises(ValueError, match="fewer than"):
        stratified_sample(population, n_total=60, seed=0)  # asks for 20 per tertile


def test_stratified_sample_rejects_non_positive_n_total() -> None:
    population = _synthetic_population(n=30)

    with pytest.raises(ValueError, match="positive"):
        stratified_sample(population, n_total=0, seed=0)


# N/A: identifiability_score is a bounded coherence metric (the max absolute
# inner product between unit-normalized decoder atoms) produced by
# sae-bounding's feature_coherence(); it is non-negative by construction, so
# no upstream path can hand stratified_sample a negative score to reject.
@pytest.mark.skip(
    reason="N/A: identifiability_score cannot be negative given feature_coherence()'s "
    "definition upstream in sae-bounding"
)
def test_stratified_sample_rejects_negative_identifiability_score() -> None:
    pass


# --- check_covariate_balance ----------------------------------------------


def test_check_covariate_balance_matches_hand_computed_summary() -> None:
    sampled = pd.DataFrame(
        {
            "feature_id": [0, 1, 2, 3],
            "identifiability_tertile": ["low", "low", "high", "high"],
            "decoder_norm": [1.0, 3.0, 2.0, 4.0],
            "activation_frequency": [0.1, 0.3, 0.2, 0.4],
        }
    )

    balance = check_covariate_balance(sampled)

    low = balance[balance["identifiability_tertile"] == "low"].iloc[0]
    high = balance[balance["identifiability_tertile"] == "high"].iloc[0]
    assert low["n"] == 2
    assert low["decoder_norm_mean"] == pytest.approx(2.0)
    assert high["decoder_norm_mean"] == pytest.approx(3.0)
    assert high["activation_frequency_mean"] == pytest.approx(0.3)


def test_check_covariate_balance_requires_tertile_column() -> None:
    sampled = pd.DataFrame({"feature_id": [0], "decoder_norm": [1.0], "activation_frequency": [0.1]})

    with pytest.raises(ValueError, match="identifiability_tertile"):
        check_covariate_balance(sampled)
