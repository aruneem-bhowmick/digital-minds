"""Tests for prism.audit_build — assembling data/audit/features.csv (REQ-2, issue #17)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import prism.audit_build as audit_build
from prism.models import LoadedPrismModel

_STUBBED_GIT_COMMIT = "cafef00d" * 5
_SOURCE_COMMIT = "deadbeef" * 5  # 40 hex characters, a well-formed (not real) git SHA


def _fake_loaded(n_features: int) -> LoadedPrismModel:
    w_dec = torch.arange(n_features * 2, dtype=torch.float32).reshape(n_features, 2)
    return LoadedPrismModel(
        model=SimpleNamespace(),
        sae=SimpleNamespace(W_dec=w_dec),
        hook_name="blocks.4.hook_resid_pre",
    )


def _config(*, checksum: str, repo: str = "aruneem-bhowmick/sae-bounding") -> dict:
    return {
        "model": {"name": "test/model", "checkpoint_revision": "abc"},
        "sae": {
            "checkpoint_repo": "test/sae",
            "checkpoint_revision": "def",
            "checkpoint_sha256": "0" * 64,
            "hook_name": "blocks.4.hook_resid_pre",
        },
        "identifiability_source": {
            "repo": repo,
            "commit": _SOURCE_COMMIT,
            "checksum": checksum,
        },
    }


def _write_identifiability(tmp_path: Path, feature_ids: list[int]) -> Path:
    path = tmp_path / "identifiability.csv"
    pd.DataFrame(
        {"feature_id": feature_ids, "identifiability_score": [0.1 * (i + 1) for i in feature_ids]}
    ).to_csv(path, index=False)
    return path


def _config_matching(path: Path, **kwargs: object) -> dict:
    """A config whose pinned checksum actually matches path's real content."""
    return _config(checksum=audit_build._sha256(path), **kwargs)


def _stub_git_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the real `git rev-parse HEAD` subprocess call.

    Without this, build_feature_audit_table()'s provenance depends on
    actually running inside a git checkout with a git executable on PATH --
    incidental environment state these tests shouldn't need to exercise the
    behavior they're actually testing.
    """

    def _fake_check_output(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return _STUBBED_GIT_COMMIT + "\n"

    monkeypatch.setattr(audit_build.subprocess, "check_output", _fake_check_output)


def test_build_feature_audit_table_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    path = _write_identifiability(tmp_path, [0, 1, 2])
    config = _config(checksum="0" * 64)  # deliberately wrong

    with pytest.raises(ValueError, match="has checksum .* expected"):
        audit_build.build_feature_audit_table(config, path)


def test_build_feature_audit_table_rejects_missing_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "identifiability.csv"
    pd.DataFrame({"feature_id": [0, 1, 2]}).to_csv(path, index=False)
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))

    with pytest.raises(ValueError, match="missing required column"):
        audit_build.build_feature_audit_table(_config_matching(path), path)


def test_build_feature_audit_table_rejects_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_identifiability(tmp_path, [0, 1])  # only 2 rows, SAE has 3 features
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))

    with pytest.raises(ValueError, match="rows but the"):
        audit_build.build_feature_audit_table(_config_matching(path), path)


def test_build_feature_audit_table_rejects_non_contiguous_feature_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same row count as n_features (3), but 1-indexed instead of matching
    # the loaded SAE's own 0-indexed W_dec row order -- the exact case a
    # row-count-only check would miss.
    path = _write_identifiability(tmp_path, [1, 2, 3])
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))

    with pytest.raises(ValueError, match="not exactly 0"):
        audit_build.build_feature_audit_table(_config_matching(path), path)


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
    _stub_git_commit(monkeypatch)

    table, provenance = audit_build.build_feature_audit_table(_config_matching(path), path)

    assert list(table["feature_id"]) == [0, 1, 2]
    np.testing.assert_allclose(table["activation_frequency"], [0.1, 0.2, 0.3])
    assert provenance["hook_name"] == "blocks.4.hook_resid_pre"
    assert provenance["git_commit"] == _STUBBED_GIT_COMMIT


def test_build_feature_audit_table_records_source_identity_not_a_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_identifiability(tmp_path, [0, 1, 2])
    monkeypatch.setattr(audit_build, "load_model_and_sae", lambda config, device="cpu": _fake_loaded(3))
    monkeypatch.setattr(
        audit_build, "measure_activation_frequencies", lambda loaded, batches: np.array([0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(audit_build, "_load_corpus", lambda *args, **kwargs: ([], {}))
    _stub_git_commit(monkeypatch)
    config = _config_matching(path, repo="someone/sae-bounding-fork")

    _, provenance = audit_build.build_feature_audit_table(config, path)

    assert "identifiability_source_csv" not in provenance
    assert provenance["identifiability_source_repo"] == "someone/sae-bounding-fork"
    assert provenance["identifiability_source_commit"] == _SOURCE_COMMIT
    assert provenance["identifiability_source_sha256"] == audit_build._sha256(path)
