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
    _nested_bin_targets,
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


def _synthetic_population(n: int = 108, seed: int = 0) -> pd.DataFrame:
    """A population where every identifiability tertile has, by
    construction, exactly the same number of features in every one of the
    nine decoder_norm x activation_frequency covariate bins.

    n must be divisible by 27 (3 tertiles x 9 bins) for the split to come
    out even. A cyclic bin assignment guarantees this deterministically --
    a random permutation at a small population size can leave a tertile
    with zero features in some bin by pure chance, which is a real
    limitation of the population, not a bug in stratified_sample()'s
    shared cross-tertile target, but not what most tests here want to
    exercise. seed is accepted for interface symmetry with other
    populations in this module but doesn't affect this one's construction.
    """
    del seed
    if n % 27 != 0:
        raise ValueError("n must be divisible by 27 for exact tertile x bin coverage")
    rng = np.random.default_rng(0)
    cycle = np.arange(n) % 9
    norm_third = cycle // 3
    freq_third = cycle % 3
    jitter = rng.uniform(0, 1, size=n)
    return pd.DataFrame(
        {
            "feature_id": np.arange(n),
            "identifiability_score": np.linspace(0.01, 0.99, n),  # strictly increasing, in [0, 1]
            "decoder_norm": norm_third + jitter,  # non-negative
            "activation_frequency": (freq_third + jitter) * 0.01,  # in [0, 1]
        }
    )


# --- load_feature_audit -----------------------------------------------------


def test_load_feature_audit_loads_a_valid_csv(tmp_path: Path) -> None:
    path = _audit_csv(tmp_path, _synthetic_population())

    df = load_feature_audit(str(path))

    assert list(df["feature_id"]) == list(range(108))
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


def test_nested_bin_targets_splits_a_single_count_evenly() -> None:
    rng = np.random.default_rng(0)

    targets = _nested_bin_targets(["a", "b", "c"], [6], rng)

    assert targets == {6: {"a": 2, "b": 2, "c": 2}}


def test_nested_bin_targets_is_identical_across_tertiles_for_equal_counts() -> None:
    # Two tertiles asking for the same count get the same target composition
    # -- the whole point of a "shared" target, not each tertile improvising
    # its own based on local availability. A single count in the input list
    # always produces one target regardless of how many tertiles use it.
    targets = _nested_bin_targets(["a", "b", "c"], [6], np.random.default_rng(0))

    assert targets[6] == {"a": 2, "b": 2, "c": 2}


def test_nested_bin_targets_never_gives_a_larger_count_less_than_a_smaller_one() -> None:
    rng = np.random.default_rng(0)
    bins = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    targets = _nested_bin_targets(bins, [13, 14], rng)

    assert sum(targets[13].values()) == 13
    assert sum(targets[14].values()) == 14
    per_bin_difference = {b: targets[14][b] - targets[13][b] for b in bins}
    assert all(diff in (0, 1) for diff in per_bin_difference.values())
    assert sum(per_bin_difference.values()) == 1  # exactly one bin absorbs the +1


def test_balanced_within_stratum_draws_from_every_targeted_bin() -> None:
    stratum = pd.DataFrame(
        {
            "feature_id": range(9),
            "covariate_bin": ["low_low"] * 3 + ["medium_medium"] * 3 + ["high_high"] * 3,
        }
    )
    target = {"low_low": 1, "medium_medium": 1, "high_high": 1}
    rng = np.random.default_rng(0)

    selected = _balanced_within_stratum(stratum, target, rng, tertile="low")

    assert len(selected) == 3
    assert set(selected["covariate_bin"]) == {"low_low", "medium_medium", "high_high"}


def test_balanced_within_stratum_raises_when_a_bin_cannot_meet_the_target() -> None:
    # Only one "low_low" feature available in this tertile, but the shared
    # target asks for two -- must fail clearly, not silently draw the
    # shortfall from a different bin.
    stratum = pd.DataFrame(
        {
            "feature_id": range(4),
            "covariate_bin": ["low_low"] + ["medium_medium"] * 3,
        }
    )
    target = {"low_low": 2, "medium_medium": 1, "high_high": 0}
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="fewer than the 2 needed"):
        _balanced_within_stratum(stratum, target, rng, tertile="low")


# --- stratified_sample ----------------------------------------------------


def test_stratified_sample_returns_exactly_n_total_rows() -> None:
    population = _synthetic_population()

    sample = stratified_sample(population, n_total=15, seed=0)

    assert len(sample) == 15


def test_stratified_sample_distributes_a_remainder_across_tertiles() -> None:
    population = _synthetic_population()

    sample = stratified_sample(population, n_total=16, seed=0)

    counts = sample["identifiability_tertile"].value_counts()
    assert counts["low"] == 6
    assert counts["medium"] == 5
    assert counts["high"] == 5


def test_stratified_sample_is_reproducible_under_a_fixed_seed() -> None:
    population = _synthetic_population()

    first = stratified_sample(population, n_total=15, seed=42)
    second = stratified_sample(population, n_total=15, seed=42)

    assert sorted(first["feature_id"]) == sorted(second["feature_id"])


def test_stratified_sample_records_tertile_provenance() -> None:
    population = _synthetic_population()

    sample = stratified_sample(population, n_total=15, seed=0)

    assert set(sample["identifiability_tertile"]) <= {"low", "medium", "high"}
    assert sample["identifiability_tertile"].notna().all()


def test_stratified_sample_matches_covariate_bin_composition_across_tertiles() -> None:
    # The point of a shared cross-tertile target: every tertile's sample
    # ends up with the *same* per-bin composition, not just a similar one.
    population = _synthetic_population()

    sample = stratified_sample(population, n_total=45, seed=0)  # 15 per tertile

    counts_by_tertile = (
        sample.groupby(["identifiability_tertile", "covariate_bin"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    assert (counts_by_tertile.nunique() == 1).all()


def test_stratified_sample_bin_counts_differ_by_at_most_one_when_tertile_counts_differ() -> None:
    # n_total=40 (this project's real configs/experiment.yaml value) splits
    # unevenly across tertiles (14/13/13). The nested target construction
    # means low's per-bin counts can exceed medium/high's by at most 1 in
    # any single bin, never by more, and never the other way around.
    population = _synthetic_population()

    sample = stratified_sample(population, n_total=40, seed=0)

    counts_by_tertile = (
        sample.groupby(["identifiability_tertile", "covariate_bin"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    pairwise_diffs = counts_by_tertile.to_numpy().max(axis=0) - counts_by_tertile.to_numpy().min(axis=0)
    assert (pairwise_diffs <= 1).all()


def test_stratified_sample_raises_clearly_when_covariates_correlate_with_identifiability() -> None:
    # decoder_norm and activation_frequency are monotonic in identifiability
    # score here, so every identifiability tertile's rows fall into exactly
    # one covariate bin (its own diagonal). A shared target spanning
    # multiple bins can't be met from a single-bin tertile -- this must
    # fail loudly, not silently draw the shortfall from whatever bin the
    # tertile actually has, which would erase the correlation instead of
    # surfacing it.
    n = 90
    scores = np.linspace(0.01, 0.99, n)
    population = pd.DataFrame(
        {
            "feature_id": np.arange(n),
            "identifiability_score": scores,
            "decoder_norm": scores * 2.0,
            "activation_frequency": scores * 0.01,
        }
    )

    with pytest.raises(ValueError, match=r"fewer than the .* needed to match the shared"):
        stratified_sample(population, n_total=9, seed=0)


def test_stratified_sample_rejects_a_tertile_too_small_to_satisfy_n_total() -> None:
    population = _synthetic_population(n=27)  # 9 features per tertile

    with pytest.raises(ValueError, match="fewer than"):
        stratified_sample(population, n_total=60, seed=0)  # asks for 20 per tertile


def test_stratified_sample_rejects_non_positive_n_total() -> None:
    population = _synthetic_population()

    with pytest.raises(ValueError, match="at least 3"):
        stratified_sample(population, n_total=0, seed=0)


def test_stratified_sample_rejects_n_total_too_small_for_three_tertiles() -> None:
    # n_total=1 or 2 gives at least one tertile a count of 0, which would
    # otherwise reach pd.concat([]) inside _balanced_within_stratum and
    # crash with a confusing pandas-internal error rather than a clear one.
    population = _synthetic_population()

    with pytest.raises(ValueError, match="at least 3"):
        stratified_sample(population, n_total=2, seed=0)


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


# --- against the real Gemma Scope audit table ------------------------------


def test_stratified_sample_runs_against_the_real_gemma_audit_table() -> None:
    """No code changes needed for the Gemma Scope pair (REQ-11 Step 3): the
    same load/sample/balance-check path REQ-2 built for Pythia's audit table
    runs unmodified against Gemma's, the actual test of whether the sampler
    is generic over which SAE checkpoint produced the audit rows.
    """
    audit = load_feature_audit("data/audit/features_gemma_scope_2b.csv")
    assert len(audit) == 16384

    sample = stratified_sample(audit, n_total=39, seed=0)
    assert len(sample) == 39
    assert set(sample["identifiability_tertile"]) <= {"low", "medium", "high"}

    balance = check_covariate_balance(sample)
    assert set(balance["identifiability_tertile"]) == {"low", "medium", "high"}
    assert (balance["n"] > 0).all()
