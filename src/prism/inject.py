"""Injection hook and strength calibration (REQ-3, REQ-5).

REQ-3 / ADR-0001 / ADR-0003: ``inject_concept`` builds a TransformerLens
forward hook that adds a scaled, normalized SAE decoder atom into a chosen
layer's residual stream, starting at the token position immediately before
the model's response and persisting through every token generated after
it -- not a single-token nudge. It returns the ``(hook_name, hook_fn)`` pair
TransformerLens's ``run_with_hooks`` / ``model.hooks()`` pattern expects;
this module never drives generation itself, so callers control their own
``model.generate()`` arguments (temperature, seed, max_new_tokens).

Strength calibration (REQ-5) is a separate concern layered on top of this
mechanism, implemented further down in this file (see the "REQ-5: strength
calibration" section below) rather than in a new module, per ADR-0007. The
mechanism above works correctly with any strength a caller supplies, whether
or not it came from that calibration pilot.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd
    import torch
    from transformer_lens import HookedTransformer


def inject_concept(
    model: "HookedTransformer",
    decoder_atom: "torch.Tensor | None",
    hook_name: str,
    strength: float,
    token_start_pos: int,
) -> "list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]]":
    """Build the forward hook for a persistent concept injection.

    ``decoder_atom=None`` is the no-injection passthrough (REQ-7's baseline
    trials): returns an empty hook list rather than a hook that happens to
    add zero, so callers always write
    ``with model.hooks(fwd_hooks=inject_concept(...)): model.generate(...)``
    with no branching on whether a trial is injected.

    ``hook_name`` must be the exact hook point the SAE's own dictionary was
    fit against (``models.LoadedPrismModel.hook_name``, e.g.
    ``blocks.4.hook_resid_pre`` for Pythia or ``blocks.20.hook_resid_post``
    for Gemma Scope) -- not reconstructed from a layer number, since that
    convention isn't uniform across checkpoints (ADR-0024: an earlier
    version of this function always assumed ``hook_resid_pre``, which
    silently injected Gemma trials one hook-tap before the site the SAE
    was actually trained on).

    Raises if ``hook_name`` has no matching hook point on ``model`` -- a
    wrong-hook config error needs to surface here, before a hook is ever
    attached, not several calls downstream during generation.
    """
    if decoder_atom is None:
        return []

    if decoder_atom.dim() != 1:
        raise ValueError(
            f"decoder_atom must be a single 1-D vector, got shape {tuple(decoder_atom.shape)}; "
            "pass one decoder row (e.g. sae.W_dec[feature_id]), not the full decoder matrix"
        )

    if token_start_pos < 0:
        raise ValueError(f"token_start_pos must be non-negative, got {token_start_pos}")

    if hook_name not in model.hook_dict:
        raise ValueError(
            f"{hook_name!r} has no matching hook point on this model; "
            "check the configured injection hook name against the model's depth/architecture"
        )

    injected_vector = _scaled_atom(decoder_atom, strength)
    return [(hook_name, _make_injection_hook(injected_vector, token_start_pos))]


def no_injection(
    model: "HookedTransformer", hook_name: str, token_start_pos: int
) -> "list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]]":
    """Explicit no-injection passthrough, sharing ``inject_concept``'s argument
    shape (REQ-7) so a baseline-trial call site reads the same as an injected
    one, differing only in which function name it calls.
    """
    del hook_name, token_start_pos  # kept for interface symmetry with inject_concept, unused here
    return []


def _scaled_atom(decoder_atom: "torch.Tensor", strength: float) -> "torch.Tensor":
    """Normalize before scaling, per ADR-0003.

    ADR-0010 recorded that this checkpoint's decoder atoms are not
    unit-normalized in the saved weights, so raw atom norms vary across the
    dictionary and are not comparable until each one is put on the same
    scale first. A genuinely zero-norm atom skips the division (which would
    otherwise be 0/0): scaling an all-zero vector by any strength is still
    all zero. Returns a clone rather than the input tensor itself -- in real
    usage ``decoder_atom`` is often a row-view into the SAE's full decoder
    matrix (e.g. ``sae.W_dec[feature_id]``), and the hook closure built from
    this return value would otherwise hold that view, and the whole decoder
    matrix's storage behind it, alive for as long as the hook is attached.
    """
    norm = decoder_atom.norm()
    if norm == 0:
        return decoder_atom.clone()
    return strength * (decoder_atom / norm)


def _make_injection_hook(
    injected_vector: "torch.Tensor", token_start_pos: int
) -> "Callable[[torch.Tensor, Any], torch.Tensor]":
    """Return a hook function that adds ``injected_vector`` to every position
    at or beyond ``token_start_pos``, and keeps doing so across every later
    call within the same generation.

    TransformerLens's ``generate()`` drives one forward call for the whole
    prompt, then one call per newly generated token once its KV cache is
    primed, and hands the hook only that call's chunk of the residual
    stream -- never the token's absolute position in the full sequence.
    This closure tracks how many positions it has already seen, starting
    fresh at zero every time ``inject_concept()`` is called (so nothing
    survives between trials), and advances that count by each chunk's
    length. That is exactly right for both the first, full-prompt call and
    every later, single-new-token call a cached ``generate()`` makes.

    This project's generation always goes through the cache, so after the
    first call every later chunk must be exactly one token; a longer later
    chunk means either ``generate()`` was called with
    ``use_past_kv_cache=False`` (each step re-forwards the whole growing
    sequence instead of just the new token) or this same hook list was
    reused across more than one ``generate()`` call. Either way, the
    position count above would already be past where the new chunk actually
    starts, and it would silently inject into positions -- including prompt
    tokens -- that must stay clean. Caught here, loudly, instead.
    """
    seen = 0
    converted_vector_by_key: dict[tuple[Any, Any], "torch.Tensor"] = {}

    def _hook(act: "torch.Tensor", hook: Any) -> "torch.Tensor":
        nonlocal seen
        del hook
        chunk_len = act.shape[1]
        if seen > 0 and chunk_len != 1:
            raise RuntimeError(
                f"inject_concept()'s hook received a chunk of length {chunk_len} "
                "after its first call, which only happens if generate() was "
                "called with use_past_kv_cache=False, or this hook list was "
                "reused across more than one generate() call. Build a fresh "
                "hook list per generate() call, with the KV cache enabled "
                "(the default)."
            )
        start, end = seen, seen + chunk_len
        seen = end

        if end <= token_start_pos:
            return act  # whole chunk still precedes the injection start

        key = (act.device, act.dtype)
        vector = converted_vector_by_key.get(key)
        if vector is None:
            vector = injected_vector.to(device=act.device, dtype=act.dtype)
            converted_vector_by_key[key] = vector

        if start >= token_start_pos:
            return act + vector  # whole chunk is at or past the start position

        first_injected_local_pos = token_start_pos - start
        act = act.clone()
        act[:, first_injected_local_pos:, :] += vector
        return act

    return _hook


# --- REQ-5: strength calibration ---------------------------------------------
#
# ADR-0007 places REQ-5's calibration pilot in this file, next to REQ-3's
# mechanism above, rather than a separate module. Everything below is a
# distinct concern from inject_concept()/no_injection(): a small,
# temperature-0 pilot that calls the mechanism above with a handful of
# candidate strengths and reports what came out, so a working strength band
# can be chosen and written into configs/experiment.yaml before REQ-6's
# systematic trials run at scale.


def select_pilot_features(pilot_source_df: "pd.DataFrame", n_features: int = 5, seed: int = 0) -> "pd.DataFrame":
    """Draw a small pilot set spanning REQ-2's identifiability tertiles.

    Pulls from ``pilot_source_df`` -- REQ-2's already-stratified,
    tertile-labeled sample (``features.stratified_sample()``'s output,
    ``data/results/sampled_features.csv`` by default) -- rather than
    drawing fresh from the full audit table, so the pilot's features are
    the same population REQ-6's systematic trials will eventually inject,
    not a separately randomized set that could land on entirely different
    features.

    Validates every column ``run_calibration_pilot()`` later reads off each
    row, not just ``identifiability_tertile`` -- a malformed pilot source
    needs to fail here, before the model and SAE are loaded, rather than as
    a ``KeyError`` partway through a pilot run that already spent the time
    to load them.
    """
    import pandas as pd
    from prism.features import _tertile_counts

    required_columns = ("feature_id", "identifiability_tertile", "identifiability_score", "decoder_norm")
    missing = [column for column in required_columns if column not in pilot_source_df.columns]
    if missing:
        raise ValueError(
            f"pilot_source_df is missing required column(s): {missing}; pass "
            "REQ-2's sampled-feature output (features.stratified_sample()'s "
            "result), not the raw audit table"
        )
    if n_features < 3:
        raise ValueError(f"n_features must be at least 3 to span all three tertiles, got {n_features}")

    counts = _tertile_counts(n_features)
    parts = []
    for tertile, count in counts.items():
        stratum = pilot_source_df[pilot_source_df["identifiability_tertile"] == tertile]
        if len(stratum) < count:
            raise ValueError(
                f"tertile {tertile!r} has only {len(stratum)} sampled features, "
                f"fewer than the {count} the pilot needs; lower n_features or "
                "check the REQ-2 sample"
            )
        parts.append(stratum.sample(n=count, random_state=seed))

    return pd.concat(parts, ignore_index=True)


_REPEATED_SEGMENT_RE = re.compile(r"(.{2,}?)\1{2,}")


def pilot_coherence_flag(text: str, *, repetition_threshold: float = 0.3, min_words: int = 3) -> dict[str, Any]:
    """A first-pass, automatic degenerate-output heuristic for one pilot generation.

    Three simple signals, not a coherence judgment, checked in this order:
    a run of 3+ immediate repeats of the same 2+ character segment anywhere
    in the raw text is flagged first, which catches character- and
    subword-level loops (``cqe-cqe-cqe-cqe``, ``::::::::``, ``lylylyly``)
    that a word-trigram count alone would miss, since whitespace-splitting
    treats a long hyphen-chain (or a spaceless run of punctuation) as a
    single "word" -- checking this ahead of the length check matters
    because that kind of garbage can be short in word count while still
    being exactly the "too strong" collapse this flag exists to catch. Text
    shorter than ``min_words`` words is flagged next (a generation
    collapsing to near-nothing is itself a sign of "too strong"). Otherwise
    the fraction of repeated word trigrams is checked against
    ``repetition_threshold`` -- a model stuck in a phrase-level repetition
    loop is the other documented shape of "too strong" (SPRINT-PLAN.md's
    "brain damage" failure mode). This flag is a starting point for the
    human read REQ-5's definition of done requires, not a replacement for
    it: it catches obvious collapse, not subtler incoherence.
    """
    from collections import Counter

    words = text.split()
    n_words = len(words)

    if _REPEATED_SEGMENT_RE.search(text):
        return {"likely_degenerate": True, "reason": "repeated_segment", "repetition_rate": None, "n_words": n_words}

    if n_words < min_words:
        return {"likely_degenerate": True, "reason": "too_short", "repetition_rate": None, "n_words": n_words}

    if n_words < 3:
        # Too few words to form a trigram; the segment-repeat check above
        # already covers character/subword loops short of full sentences.
        return {"likely_degenerate": False, "reason": None, "repetition_rate": None, "n_words": n_words}

    trigrams = [tuple(words[i : i + 3]) for i in range(n_words - 2)]
    counts = Counter(trigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    repetition_rate = repeated / len(trigrams)

    likely_degenerate = repetition_rate >= repetition_threshold
    return {
        "likely_degenerate": likely_degenerate,
        "reason": "high_repetition" if likely_degenerate else None,
        "repetition_rate": repetition_rate,
        "n_words": n_words,
    }


def run_calibration_pilot(
    loaded: Any,
    pilot_features: "pd.DataFrame",
    strengths: "list[float]",
    *,
    layer: int,
    layer_source: str,
    prompt: str,
    config: dict[str, Any],
    pilot_feature_seed: int,
    max_new_tokens: int = 60,
) -> "list[dict[str, Any]]":
    """Run the REQ-5 pilot: every sampled feature x every candidate strength,
    at temperature 0 (ADR-0008), returning one full record per pair.

    ``loaded`` is a ``models.LoadedPrismModel``, typed as ``Any`` here to
    avoid importing sae_lens/transformer_lens at this module's import time,
    matching the lazy-import convention the mechanism above already uses.
    ``layer_source`` is recorded on every record so a downstream reader --
    a human, or the eventual calibration figure -- can tell an ADR-0009
    fallback apart from a resolved REQ-10 layer without re-deriving it.
    ``config`` is the parsed ``configs/experiment.yaml`` (or equivalent)
    used to load ``loaded``; its model/SAE checkpoint identity is copied
    onto every record so each one is reconstructable on its own, per
    CLAUDE.md's reproducibility rule, without cross-referencing the git
    commit against whatever the config file happened to say at that commit.
    """
    import subprocess
    from datetime import datetime, timezone

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    model_name = config["model"]["name"]
    model_checkpoint_revision = config["model"]["checkpoint_revision"]
    sae_checkpoint_repo = config["sae"]["checkpoint_repo"]
    sae_checkpoint_revision = config["sae"]["checkpoint_revision"]
    sae_checkpoint_sha256 = config["sae"]["checkpoint_sha256"]

    tokens = loaded.model.to_tokens(prompt)
    token_start_pos = tokens.shape[1] - 1

    records: list[dict[str, Any]] = []

    for _, feature_row in pilot_features.iterrows():
        feature_id = int(feature_row["feature_id"])
        decoder_atom = loaded.sae.W_dec[feature_id]

        for strength in strengths:
            hooks = inject_concept(
                loaded.model,
                decoder_atom,
                hook_name=loaded.hook_name,
                strength=strength,
                token_start_pos=token_start_pos,
            )
            with loaded.model.hooks(fwd_hooks=hooks):
                output = loaded.model.generate(
                    tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False
                )
            response_text = loaded.model.to_string(output[0, tokens.shape[1] :])

            records.append(
                {
                    "feature_id": feature_id,
                    "identifiability_tertile": str(feature_row["identifiability_tertile"]),
                    "identifiability_score": float(feature_row["identifiability_score"]),
                    "decoder_norm": float(feature_row["decoder_norm"]),
                    "model_name": model_name,
                    "model_checkpoint_revision": model_checkpoint_revision,
                    "sae_checkpoint_repo": sae_checkpoint_repo,
                    "sae_checkpoint_revision": sae_checkpoint_revision,
                    "sae_checkpoint_sha256": sae_checkpoint_sha256,
                    "layer": layer,
                    "layer_source": layer_source,
                    "strength": float(strength),
                    "prompt": prompt,
                    "temperature": 0,
                    "pilot_feature_seed": pilot_feature_seed,
                    "response_text": response_text,
                    "coherence": pilot_coherence_flag(response_text),
                    "git_commit": git_commit,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    return records


def summarize_pilot(records: "list[dict[str, Any]]") -> str:
    """Render every pilot record as a readable, strength-grouped text block.

    Meant to be printed and read directly -- REQ-5's definition of done
    requires a human judgment on top of ``pilot_coherence_flag()``'s
    automatic signal, and that judgment needs something more legible than
    the raw JSONL log to work from.
    """
    lines: list[str] = []
    strengths = sorted({record["strength"] for record in records})

    for strength in strengths:
        lines.append(f"=== strength {strength} ===")
        for record in records:
            if record["strength"] != strength:
                continue
            flag = "DEGENERATE" if record["coherence"]["likely_degenerate"] else "ok"
            preview = " ".join(record["response_text"].split())[:160]
            lines.append(
                f"  feature {record['feature_id']} ({record['identifiability_tertile']}, "
                f"id_score={record['identifiability_score']:.3f}) [{flag}]: {preview}"
            )
        lines.append("")

    return "\n".join(lines)


def save_pilot_records(records: "list[dict[str, Any]]", output_path: "Path | str") -> "Path":
    """Persist run_calibration_pilot()'s output as JSONL, one record per line.

    Overwritten on every run rather than append-only: unlike ``data/trials/``'s
    ADR-0005 schema for systematic, judged trial records, a calibration
    pilot is qualitative and re-run whenever the candidate strengths
    change, and each run's log should reflect only that run's candidates,
    not every candidate ever tried across every past pilot.
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return output_path


def main() -> None:
    """CLI entry point: python -m prism.inject --config configs/experiment.yaml.

    Runs the full REQ-5 pilot against the real model/SAE pair, prints
    ``summarize_pilot()``'s output for a human to read, and writes the full
    record set to ``data/results/calibration_pilot.jsonl``. Per CLAUDE.md
    §6 ("Config lives in YAML, not command-line flags"), ``--strengths``
    defaults to whatever ``configs/experiment.yaml``'s ``injection.strengths``
    already holds, so the documented no-argument invocation
    (``python -m prism.inject --config configs/experiment.yaml``)
    reproduces the committed pilot rather than silently sweeping a different
    band; passing ``--strengths`` explicitly still overrides it for a
    one-off exploratory sweep.
    """
    import argparse

    import pandas as pd
    import yaml

    from prism.layers import resolve_injection_layer
    from prism.models import load_model_and_sae
    from prism.prompts import detection_prompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--sampled-features", default="data/results/sampled_features.csv")
    parser.add_argument(
        "--strengths",
        default=None,
        help="comma-separated candidate strengths; defaults to the config's injection.strengths",
    )
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--output", default="data/results/calibration_pilot.jsonl")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="passed straight to load_model_and_sae(); defaults to 'cpu' -- a Modal GPU "
        "invocation must pass 'cuda' explicitly (load_model_and_sae() does not "
        "auto-detect it, see ADR-0023)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    loaded = load_model_and_sae(config, device=args.device)
    sampled = pd.read_csv(args.sampled_features)
    seed = config.get("features", {}).get("sample_seed", 0)
    pilot_features = select_pilot_features(sampled, n_features=args.n_features, seed=seed)

    layer, layer_source = resolve_injection_layer(config, loaded.model.cfg.n_layers)
    if args.strengths is not None:
        strengths = [float(s) for s in args.strengths.split(",")]
    else:
        strengths = [float(s) for s in config["injection"]["strengths"]]

    records = run_calibration_pilot(
        loaded,
        pilot_features,
        strengths,
        layer=layer,
        layer_source=layer_source,
        prompt=detection_prompt(),
        config=config,
        pilot_feature_seed=seed,
        max_new_tokens=args.max_new_tokens,
    )

    print(summarize_pilot(records))
    output_path = save_pilot_records(records, args.output)
    print(f"wrote {len(records)} pilot records to {output_path}")


if __name__ == "__main__":
    main()
