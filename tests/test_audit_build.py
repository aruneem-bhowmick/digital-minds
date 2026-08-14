"""Tests for prism.audit_build — assembling data/audit/features.csv (REQ-2)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import prism.audit_build as audit_build
from prism.models import LoadedPrismModel


def _fake_loaded(n_features: int) -> LoadedPrismModel:
    w_dec = torch.arange(n_features * 2, dtype=torch.float32).reshape(n_features, 2)
    return LoadedPrismModel(
        model=SimpleNamespace(),
        sae=SimpleNamespace(W_dec=w_dec),
        hook_name="blocks.4.hook_resid_pre",
    )


def _config() -> dict:
    return {
        "model": {"name": "test/model", "checkpoint_revision": "abc"},
        "sae": {
            "checkpoint_repo": "test/sae",
            "checkpoint_revision": "def",
            "checkpoint_sha256": "0" * 64,
            "hook_name": "blocks.4.hook_resid_pre",
        },
    }


def _write_identifiability(tmp_path: Path, feature_ids: list[int]) -> Path:
    path = tmp_path / "identifiability.csv"
    pd.DataFrame(
        {"feature_id": feature_ids, "identifiability_score": [0.1 * (i + 1) for i in feature_ids]}
    ).to_csv(path, index=False)
    return path


def test_build_feature_audit_table_rejects_missing_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "identifiability.csv"
    pd.DataFrame({"feature_id": [0, 1, 2]}).to_csv(path, index=False)
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))

    with pytest.raises(ValueError, match="missing required column"):
        audit_build.build_feature_audit_table(_config(), path, identifiability_source_commit="abc123")


def test_build_feature_audit_table_rejects_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_identifiability(tmp_path, [0, 1])  # only 2 rows, SAE has 3 features
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))

    with pytest.raises(ValueError, match="rows but the"):
        audit_build.build_feature_audit_table(_config(), path, identifiability_source_commit="abc123")


def test_build_feature_audit_table_rejects_non_contiguous_feature_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same row count as n_features (3), but 1-indexed instead of matching
    # the loaded SAE's own 0-indexed W_dec row order -- the exact case a
    # row-count-only check would miss.
    path = _write_identifiability(tmp_path, [1, 2, 3])
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))

    with pytest.raises(ValueError, match="not exactly 0"):
        audit_build.build_feature_audit_table(_config(), path, identifiability_source_commit="abc123")


def test_build_feature_audit_table_joins_aligned_features_by_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # feature_id deliberately out of row order in the source file; the join
    # must sort by feature_id before pairing with decoder_norm/activation_frequency.
    path = _write_identifiability(tmp_path, [2, 0, 1])
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))
    monkeypatch.setattr(
        audit_build, "measure_activation_frequencies", lambda loaded, batches: np.array([0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(audit_build, "_load_corpus", lambda *args, **kwargs: ([], {}))

    table, provenance = audit_build.build_feature_audit_table(
        _config(), path, identifiability_source_commit="abc123"
    )

    assert list(table["feature_id"]) == [0, 1, 2]
    np.testing.assert_allclose(table["activation_frequency"], [0.1, 0.2, 0.3])
    assert provenance["hook_name"] == "blocks.4.hook_resid_pre"


def test_build_feature_audit_table_records_source_identity_not_a_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_identifiability(tmp_path, [0, 1, 2])
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))
    monkeypatch.setattr(
        audit_build, "measure_activation_frequencies", lambda loaded, batches: np.array([0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(audit_build, "_load_corpus", lambda *args, **kwargs: ([], {}))

    _, provenance = audit_build.build_feature_audit_table(
        _config(),
        path,
        identifiability_source_commit="abc123",
        identifiability_source_repo="someone/sae-bounding-fork",
    )

    assert "identifiability_source_csv" not in provenance
    assert provenance["identifiability_source_repo"] == "someone/sae-bounding-fork"
    assert provenance["identifiability_source_commit"] == "abc123"
    assert provenance["identifiability_source_sha256"] == audit_build._sha256(path)


def test_build_feature_audit_table_defaults_the_source_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_identifiability(tmp_path, [0, 1, 2])
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))
    monkeypatch.setattr(
        audit_build, "measure_activation_frequencies", lambda loaded, batches: np.array([0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(audit_build, "_load_corpus", lambda *args, **kwargs: ([], {}))

    _, provenance = audit_build.build_feature_audit_table(
        _config(), path, identifiability_source_commit="abc123"
    )

    assert provenance["identifiability_source_repo"] == audit_build.DEFAULT_IDENTIFIABILITY_SOURCE_REPO
