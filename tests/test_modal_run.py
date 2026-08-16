"""Tests for prism.modal_run — the Modal GPU execution wrapper (REQ-11).

Scope limited to _upload()'s destination-path validation: the one piece of
this module with real logic to get wrong (a path-traversal or absolute-path
destination escaping REPO_DIR). Everything else in this module is direct,
untested-by-design orchestration of the Modal SDK (git clone, pip install,
exec a caller-supplied command) -- an ops/infra tool, not experimental logic,
per this project's own stated test-priority order.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prism.modal_run import REPO_DIR, _upload


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "../outside.csv",
        "data/../../outside.csv",
        "/etc/passwd",
    ],
)
def test_upload_rejects_unsafe_destinations(bad_path: str) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        _upload(sandbox=None, local_path=None, repo_relative_path=bad_path)  # type: ignore[arg-type]


def test_upload_copies_a_valid_destination_into_the_repo_checkout() -> None:
    calls: dict[str, tuple] = {}
    fake_filesystem = SimpleNamespace(
        make_directory=lambda path, **kwargs: calls.__setitem__("make_directory", (path, kwargs)),
        copy_from_local=lambda local, remote: calls.__setitem__("copy_from_local", (local, remote)),
    )
    fake_sandbox = SimpleNamespace(filesystem=fake_filesystem)

    _upload(fake_sandbox, Path("local.csv"), "data/audit/new.csv")

    assert calls["make_directory"] == (f"{REPO_DIR}/data/audit", {"create_parents": True})
    assert calls["copy_from_local"] == (Path("local.csv"), f"{REPO_DIR}/data/audit/new.csv")
