"""Modal GPU execution for REQ-11's Gemma Scope pipeline.

This module contains no experimental logic of its own. It clones this
repository at a specific commit into a GPU-backed Modal Sandbox and runs
the same unmodified CLI commands and tests that would run locally --
``models.py``, ``inject.py``, ``runner.py``, and ``judge.py`` are exactly
the tested, reviewed code from every other REQ this sprint, just executed
somewhere with a GPU instead of this machine's CPU-only hardware. Every
result those modules produce still carries the same config/git-commit/
timestamp provenance; only the execution environment differs, which is
recorded explicitly rather than silently blended with the CPU-run Pythia
data (see ARCHITECTURE.md's REQ-11 ADR).

Usage: ``python -m prism.modal_run --branch req-11/gemma-scope-2b-loading
--command "pytest tests/test_models.py -k gemma -m integration -v"``
"""

from __future__ import annotations

import argparse
import os
import sys

import modal

app = modal.App("prism-gemma-pipeline")

_PINNED_DEPENDENCIES = [
    "transformer-lens==3.7.1",
    "sae-lens==6.49.1",
    "torch==2.6.0",
    "sentencepiece==0.2.0",
    "transformers==5.9.0",
    "numpy==2.5.2",
    "pandas==3.0.5",
    "statsmodels==0.14.6",
    "scikit-learn==1.9.0",
    "matplotlib==3.11.1",
    "anthropic==0.122.0",
    "pyyaml==6.0.3",
    "pytest==9.1.1",
    "datasets==5.0.1",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*_PINNED_DEPENDENCIES)
)

REPO_URL = "https://github.com/aruneem-bhowmick/digital-minds.git"
REPO_DIR = "/root/digital-minds"


def _secrets() -> list[modal.Secret]:
    """Build Modal secrets from this machine's own .env, the same two
    credentials already used locally (HF_TOKEN, ANTHROPIC_API_KEY) --
    not a separate credential story to set up in the Modal dashboard.
    """
    env_values: dict[str, str] = {}
    with open(".env", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key in ("HF_TOKEN", "ANTHROPIC_API_KEY"):
                env_values[key] = value
    missing = {"HF_TOKEN", "ANTHROPIC_API_KEY"} - set(env_values)
    if missing:
        raise ValueError(f".env is missing required key(s): {sorted(missing)}")
    return [modal.Secret.from_dict(env_values)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--command", required=True, help="shell command run from the repo root")
    parser.add_argument("--gpu", default="A10G")
    parser.add_argument("--timeout", type=int, default=1800)
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
            clone = sandbox.exec(
                "git", "clone", "--branch", args.branch, "--depth", "1", REPO_URL, REPO_DIR
            )
            for line in clone.stdout:
                print(line, end="")
            clone.wait()
            if clone.returncode != 0:
                raise RuntimeError(f"git clone failed with exit code {clone.returncode}")

            install = sandbox.exec("pip", "install", "-e", ".", workdir=REPO_DIR)
            for line in install.stdout:
                print(line, end="")
            install.wait()
            if install.returncode != 0:
                raise RuntimeError(f"pip install failed with exit code {install.returncode}")

            run = sandbox.exec("bash", "-c", args.command, workdir=REPO_DIR)
            for line in run.stdout:
                print(line, end="")
            for line in run.stderr:
                print(line, end="", file=sys.stderr)
            run.wait()
            if run.returncode != 0:
                raise RuntimeError(f"command failed with exit code {run.returncode}")

            print(f"\n[modal_run] command succeeded on {args.gpu}")
        finally:
            sandbox.terminate()


if __name__ == "__main__":
    main()
