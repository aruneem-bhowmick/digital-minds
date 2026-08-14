"""Assemble data/audit/features.csv from the identifiability audit and a
local activation-frequency measurement (REQ-2, ADR-0011, ADR-0013).

This is a one-time data-preparation step, not part of the per-run
experiment pipeline features.py exposes. The per-feature identifiability
score is produced upstream, in ``sae-bounding``, and read here as a fixed
input; activation frequency is measured here, over a pinned text corpus,
using REQ-1's ``load_model_and_sae()``; decoder norm is read directly off
the loaded SAE's own weights. The three are joined on feature index and
written to ``data/audit/features.csv``, which everything downstream in
this project (REQ-2's sampler onward) treats as a static, read-only input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import load_dataset

from prism.models import decoder_norms, load_model_and_sae, measure_activation_frequencies

# Anchors the git_commit provenance lookup to this repository regardless of
# the caller's own working directory (src/prism/audit_build.py -> src/prism -> src -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ADR-0013: NeelNanda/pile-10k, a 10,000-document sample of the Pile, which
# is Pythia's own training corpus. Pinned to a specific dataset revision, not
# a mutable branch ref, per CLAUDE.md's reproducibility rule.
DEFAULT_CORPUS_DATASET = "NeelNanda/pile-10k"
DEFAULT_CORPUS_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
DEFAULT_N_DOCUMENTS = 500
DEFAULT_MAX_TOKENS_PER_DOCUMENT = 256

# The only upstream repo this project currently reads a per-feature
# identifiability table from (issue #17 / ADR-0011).
DEFAULT_IDENTIFIABILITY_SOURCE_REPO = "aruneem-bhowmick/sae-bounding"


def build_feature_audit_table(
    config: dict[str, Any],
    identifiability_csv: Path,
    *,
    identifiability_source_commit: str,
    identifiability_source_repo: str = DEFAULT_IDENTIFIABILITY_SOURCE_REPO,
    n_documents: int = DEFAULT_N_DOCUMENTS,
    max_tokens_per_document: int = DEFAULT_MAX_TOKENS_PER_DOCUMENT,
    corpus_dataset: str = DEFAULT_CORPUS_DATASET,
    corpus_revision: str = DEFAULT_CORPUS_REVISION,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join identifiability, decoder norm, and activation frequency into one table.

    Returns the assembled dataframe plus a provenance record documenting
    exactly how it was produced, so the result can be reconstructed from
    logged config alone per CLAUDE.md's reproducibility rule.

    ``identifiability_source_commit`` (the sae-bounding commit that produced
    ``identifiability_csv``) has no default -- it changes with every
    regeneration on the sae-bounding side, and guessing it would silently
    misattribute provenance rather than surface that it wasn't recorded.
    The CSV's own content is checksummed here regardless, so the provenance
    record identifies what was actually read, not just where a local
    (and typically machine-specific) copy of it happened to sit on disk.
    """
    if len(identifiability_source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in identifiability_source_commit.lower()
    ):
        raise ValueError(
            f"identifiability_source_commit {identifiability_source_commit!r} is not a "
            "40-character git commit SHA -- this only checks the value is well-formed, "
            "not that it's the actual commit that produced identifiability_csv (issue #17, "
            "blocked on sae-bounding PR #6 merging before that can be verified)"
        )

    identifiability = pd.read_csv(identifiability_csv)
    missing_columns = {"feature_id", "identifiability_score"} - set(identifiability.columns)
    if missing_columns:
        raise ValueError(
            f"{identifiability_csv} is missing required column(s): {sorted(missing_columns)}"
        )

    loaded = load_model_and_sae(config, device=device)
    n_features = loaded.sae.W_dec.shape[0]
    if len(identifiability) != n_features:
        raise ValueError(
            f"identifiability table has {len(identifiability)} rows but the "
            f"loaded SAE has {n_features} features -- these must come from "
            "the same checkpoint"
        )
    if set(identifiability["feature_id"]) != set(range(n_features)):
        raise ValueError(
            f"{identifiability_csv}'s feature_id values are not exactly "
            f"0..{n_features - 1} -- the join below assigns decoder_norm and "
            "activation_frequency by position, which requires an exact, "
            "contiguous match to the loaded SAE's own feature indexing"
        )

    token_batches, corpus_provenance = _load_corpus(
        loaded,
        dataset_name=corpus_dataset,
        revision=corpus_revision,
        n_documents=n_documents,
        max_tokens_per_document=max_tokens_per_document,
    )

    table = identifiability.sort_values("feature_id").reset_index(drop=True)
    table["decoder_norm"] = decoder_norms(loaded)
    table["activation_frequency"] = measure_activation_frequencies(loaded, token_batches)

    provenance = {
        "model_name": config["model"]["name"],
        "model_checkpoint_revision": config["model"]["checkpoint_revision"],
        "sae_checkpoint_repo": config["sae"]["checkpoint_repo"],
        "sae_checkpoint_revision": config["sae"]["checkpoint_revision"],
        "sae_checkpoint_sha256": config["sae"]["checkpoint_sha256"],
        "hook_name": config["sae"]["hook_name"],
        "identifiability_source_repo": identifiability_source_repo,
        "identifiability_source_commit": identifiability_source_commit,
        "identifiability_source_sha256": _sha256(identifiability_csv),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=_REPO_ROOT
        ).strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **corpus_provenance,
    }
    return table, provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_corpus(
    loaded: Any,
    *,
    dataset_name: str,
    revision: str,
    n_documents: int,
    max_tokens_per_document: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Tokenize and truncate the first n_documents of a pinned HF dataset."""
    dataset = load_dataset(dataset_name, revision=revision, split="train")
    documents = dataset.select(range(min(n_documents, len(dataset))))

    token_batches = []
    total_tokens = 0
    for record in documents:
        tokens = loaded.model.to_tokens(record["text"])[:, :max_tokens_per_document]
        token_batches.append(tokens)
        total_tokens += tokens.shape[1]

    provenance = {
        "corpus_dataset": dataset_name,
        "corpus_revision": revision,
        "corpus_n_documents": len(token_batches),
        "corpus_max_tokens_per_document": max_tokens_per_document,
        "corpus_total_tokens": total_tokens,
    }
    return token_batches, provenance


def main() -> None:
    """CLI entry point: python -m prism.audit_build --identifiability-csv <path>."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--identifiability-csv", required=True)
    parser.add_argument(
        "--identifiability-source-commit",
        required=True,
        help="sae-bounding commit that produced --identifiability-csv (no default; must be stated explicitly)",
    )
    parser.add_argument(
        "--identifiability-source-repo", default=DEFAULT_IDENTIFIABILITY_SOURCE_REPO
    )
    parser.add_argument("--output", default="data/audit/features.csv")
    parser.add_argument(
        "--provenance-output", default="data/results/req2_feature_audit_provenance.json"
    )
    parser.add_argument("--n-documents", type=int, default=DEFAULT_N_DOCUMENTS)
    parser.add_argument(
        "--max-tokens-per-document", type=int, default=DEFAULT_MAX_TOKENS_PER_DOCUMENT
    )
    parser.add_argument("--corpus-dataset", default=DEFAULT_CORPUS_DATASET)
    parser.add_argument("--corpus-revision", default=DEFAULT_CORPUS_REVISION)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    table, provenance = build_feature_audit_table(
        config,
        Path(args.identifiability_csv),
        identifiability_source_commit=args.identifiability_source_commit,
        identifiability_source_repo=args.identifiability_source_repo,
        n_documents=args.n_documents,
        max_tokens_per_document=args.max_tokens_per_document,
        corpus_dataset=args.corpus_dataset,
        corpus_revision=args.corpus_revision,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    provenance_path = Path(args.provenance_output)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
