"""Modal GPU execution for REQ-11's Gemma Scope pipeline.

This module contains no experimental logic of its own. It clones this
repository at a specific branch into a GPU-backed Modal Sandbox and runs
the same unmodified CLI commands and tests that would run locally --
``models.py``, ``inject.py``, ``runner.py``, and ``judge.py`` are exactly
the tested, reviewed code from every other REQ this sprint, just executed
somewhere with a GPU instead of this machine's CPU-only hardware. A
downloaded result's own provenance fields (``git_commit``, ``timestamp``)
are captured from inside the sandbox's real git checkout by the same
functions (``save_reconstruction_result()`` and similar) that already do
this for local runs -- this module adds no separate provenance mechanism
of its own, and reports which GPU a run used directly in its own console
output rather than only in a result schema.

Usage: ``python -m prism.modal_run --branch req-11/gemma-scope-2b-loading
--command "pytest tests/test_models.py -k gemma -m integration -v"``
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import modal
from dotenv import dotenv_values

app = modal.App("prism-gemma-pipeline")

# Anchors .env and pyproject.toml lookups to this repository regardless of
# the caller's own working directory, matching audit_build.py's own
# _REPO_ROOT convention (src/prism/modal_run.py -> src/prism -> src -> root).
_REPO_ROOT = Path(__file__).resolve().parents[2]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject(str(_REPO_ROOT / "pyproject.toml"))
)

REPO_URL = "https://github.com/aruneem-bhowmick/digital-minds.git"
REPO_DIR = "/root/digital-minds"

_REQUIRED_SECRET_KEYS = ("HF_TOKEN", "ANTHROPIC_API_KEY")


def _secrets() -> list[modal.Secret]:
    """Build Modal secrets from this machine's own .env, the same two
    credentials already used locally -- not a separate credential story to
    set up in the Modal dashboard. Uses python-dotenv's own parser rather
    than a hand-rolled one, so quoted values and spaces around ``=`` (both
    ordinary .env conventions) don't silently corrupt a secret.
    """
    env_values = dotenv_values(_REPO_ROOT / ".env")
    missing = [key for key in _REQUIRED_SECRET_KEYS if not env_values.get(key)]
    if missing:
        raise ValueError(f".env is missing required key(s): {sorted(missing)}")
    return [modal.Secret.from_dict({key: env_values[key] for key in _REQUIRED_SECRET_KEYS})]


def _run_step(sandbox: modal.Sandbox, *args: str, workdir: str | None = None, step_name: str) -> None:
    """Run one command to completion, streaming stdout and stderr concurrently.

    Draining the two streams sequentially (stdout fully, then stderr) risks
    a pipe-fill deadlock if the remote process writes enough to stderr
    while nothing is reading it yet (real for this project: huggingface_hub's
    download progress bars go to stderr while pytest's own output goes to
    stdout, both at once). Two threads read each stream as it arrives
    instead. Also used for every step (clone, install, the caller's own
    command) rather than one-off per-step blocks, since all three need the
    identical exec-stream-check-raise shape.
    """
    process = sandbox.exec(*args, workdir=workdir)

    captured_stderr: list[str] = []

    def _drain_stdout() -> None:
        for line in process.stdout:
            print(line, end="")

    def _drain_stderr() -> None:
        for line in process.stderr:
            captured_stderr.append(line)
            print(line, end="")

    stdout_thread = threading.Thread(target=_drain_stdout)
    stderr_thread = threading.Thread(target=_drain_stderr)
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with exit code {process.returncode}: "
            f"{''.join(captured_stderr) or '(no stderr captured)'}"
        )


def _download(sandbox: modal.Sandbox, repo_relative_path: str) -> Path:
    """Copy one file from the sandbox's repo checkout back to this machine's
    identical path, so a real result the remote GPU produced becomes a real
    local file this project's own commit/provenance discipline can track --
    not something left stranded on ephemeral sandbox storage.
    """
    local_path = Path(repo_relative_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sandbox.filesystem.copy_to_local(f"{REPO_DIR}/{repo_relative_path}", local_path)
    return local_path


def _upload(sandbox: modal.Sandbox, local_path: Path, repo_relative_path: str) -> None:
    """Copy one local file into the sandbox's repo checkout before the
    command runs -- the upload-side mirror of _download(). For local-only
    inputs the remote command needs but this repo doesn't track (e.g.
    sae-bounding's per-feature identifiability CSV, gitignored there).

    Creates the destination's parent directory first, mirroring _download()'s
    local-side mkdir(parents=True, exist_ok=True) -- a fresh git clone won't
    contain an untracked or gitignored destination directory on its own.
    """
    remote_path = f"{REPO_DIR}/{repo_relative_path}"
    remote_parent = remote_path.rsplit("/", 1)[0]
    sandbox.filesystem.make_directory(remote_parent, create_parents=True)
    sandbox.filesystem.copy_from_local(local_path, remote_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--command", required=True, help="shell command run from the repo root")
    parser.add_argument("--gpu", default="A10G")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--download",
        action="append",
        default=[],
        help="repo-relative path to copy back after the command succeeds; repeatable",
    )
    parser.add_argument(
        "--upload",
        nargs=2,
        action="append",
        default=[],
        metavar=("LOCAL_PATH", "REPO_RELATIVE_PATH"),
        help="copy a local file into the sandbox's repo checkout before running the "
        "command; repeatable",
    )
    args = parser.parse_args()

    with modal.enable_output(), app.run():
        sandbox = modal.Sandbox.create(
            app=app,
            image=image,
            gpu=args.gpu,
            secrets=_secrets(),
            timeout=args.timeout,
        )
        try:
            _run_step(
                sandbox,
                "git",
                "clone",
                "--branch",
                args.branch,
                "--depth",
                "1",
                REPO_URL,
                REPO_DIR,
                step_name="git clone",
            )
            _run_step(sandbox, "pip", "install", "-e", ".", workdir=REPO_DIR, step_name="pip install")

            for local_path, repo_relative_path in args.upload:
                _upload(sandbox, Path(local_path), repo_relative_path)
                print(f"[modal_run] uploaded {local_path} -> {repo_relative_path}")

            _run_step(sandbox, "bash", "-c", args.command, workdir=REPO_DIR, step_name="command")

            print(f"\n[modal_run] command succeeded on {args.gpu}")

            for repo_relative_path in args.download:
                local_path = _download(sandbox, repo_relative_path)
                print(f"[modal_run] downloaded {local_path}")
        finally:
            sandbox.terminate()


if __name__ == "__main__":
    main()
