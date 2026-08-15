"""Systematic trial runner plus baseline and control batches (REQ-6, REQ-7).

Three loops share the same underlying trial-record schema and the same
resumable JSONL file, ``data/trials/trials.jsonl``, distinguished by
``prompt_type`` (ADR-0005): ``run_systematic_trials()`` (REQ-6) injects each
REQ-2 sampled feature at each REQ-5 calibrated strength and asks
``prompts.detection_prompt()``, following up with
``prompts.naming_subtask_prompt()`` only when the model's answer reads as
affirmative. ``run_baseline_trials()`` (REQ-7) runs the identical two-turn
protocol with ``inject.no_injection()`` in place of ``inject.inject_concept()``,
to measure the false-positive rate. ``run_control_trials()`` (REQ-7) keeps the
injection active but substitutes one of the versioned
``prompts.unrelated_control_prompt()`` questions for the detection question,
to check for a generic yes-bias independent of the injected concept.

REQ-10 has not resolved the primary injection layer yet; every trial here
uses ``layers.get_fallback_layer()`` (ADR-0009's explicit fallback) and
records ``layer_source: "adr-0009-fallback"`` on every row, the same
convention REQ-5's calibration pilot already established.

This module does not implement judge scoring (REQ-8): every record it writes
carries ``judge_scores: null``.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prism.inject import inject_concept, no_injection
from prism.layers import get_fallback_layer
from prism.prompts import (
    DEFAULT_CONTROL_QUESTIONS_PATH,
    detection_prompt,
    naming_subtask_prompt,
    unrelated_control_prompt,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import torch

    from prism.models import LoadedPrismModel

DEFAULT_TRIALS_PATH = "data/trials/trials.jsonl"
DEFAULT_MAX_NEW_TOKENS = 60

_AFFIRMATIVE_RE = re.compile(r"^\W*yes\b", re.IGNORECASE)


def is_affirmative(response_text: str) -> bool:
    """Whether a detection response counts as a "yes".

    ``detection_prompt()`` asks explicitly for a single yes/no word before
    any elaboration (ADR-0015), so this is a direct read of the first word,
    not a judgment call deferred to REQ-8's judge. That distinction matters:
    the runner has to decide *now*, at generation time, whether to ask the
    naming follow-up, well before a judge model ever sees the transcript.
    ``\\b`` after "yes" excludes a response that merely starts with the
    letters ("yesterday...") without being the word itself.
    """
    return bool(_AFFIRMATIVE_RE.match(response_text.strip()))


# --- shared two-turn (detection, then conditional naming) protocol ---------


def run_two_turn_trial(
    loaded: "LoadedPrismModel",
    hooks_fn: "Callable[[int], list]",
    *,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run one detection trial, following up with the naming question only on
    an affirmative answer (REQ-6, shared with REQ-7's baseline).

    ``hooks_fn(token_start_pos)`` builds a fresh TransformerLens hook list for
    one ``generate()`` call -- either ``inject.inject_concept()`` bound to a
    specific feature/strength, or ``inject.no_injection()`` for a baseline
    trial. It is called once per ``generate()`` call, never reused across
    calls: ``inject_concept()``'s hook raises if handed a second call's
    prefill chunk, since a reused hook's internal position counter would
    already be past where that chunk actually starts (see ``inject.py``).

    The naming follow-up, when asked, continues from the detection turn's
    *actual generated token IDs*, not a re-tokenization of the pasted-together
    text. Re-tokenizing risks the continuation's token boundaries drifting
    from the original prompt's, which would silently shift where
    ``token_start_pos`` actually falls in the new sequence. Concatenating
    token ID tensors keeps the injection's start position exactly aligned
    across both calls, so the injected vector persists through the entire
    two-turn exchange (ADR-0003), not just the first turn.

    Returns a structured record, not a flat string: ``{"detection": {"prompt",
    "response"}, "affirmative": bool, "naming": {"prompt", "response"} | None}``.
    REQ-8's judge needs both turns visible to grade whether detection happened
    before the model could have named the concept some other way.
    """
    import torch

    detection_text = detection_prompt()
    detection_tokens = loaded.model.to_tokens(detection_text)
    token_start_pos = detection_tokens.shape[1] - 1

    detection_output = _generate_turn(
        loaded,
        hooks_fn(token_start_pos),
        detection_tokens,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    detection_response = loaded.model.to_string(detection_output[0, detection_tokens.shape[1] :])
    affirmative = is_affirmative(detection_response)

    naming_record = None
    if affirmative:
        naming_text = naming_subtask_prompt()
        suffix_tokens = loaded.model.to_tokens("\n\n" + naming_text, prepend_bos=False)
        combined_tokens = torch.cat([detection_output, suffix_tokens], dim=1)

        naming_output = _generate_turn(
            loaded,
            hooks_fn(token_start_pos),
            combined_tokens,
            temperature=temperature,
            seed=seed + 1,
            max_new_tokens=max_new_tokens,
        )
        naming_response = loaded.model.to_string(naming_output[0, combined_tokens.shape[1] :])
        naming_record = {"prompt": naming_text, "response": naming_response}

    return {
        "detection": {"prompt": detection_text, "response": detection_response},
        "affirmative": affirmative,
        "naming": naming_record,
    }


def run_control_trial(
    loaded: "LoadedPrismModel",
    hooks: list,
    control_question: dict[str, Any],
    *,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run one unrelated-question control trial (REQ-7): a single generation,
    no naming follow-up, since there is no injected concept for the model to
    name here by design.

    Unlike ``run_two_turn_trial()``, ``hooks`` is a single already-built hook
    list rather than a factory -- a control trial only ever makes one
    ``generate()`` call, so there is no second call to build a fresh list for.
    """
    tokens = loaded.model.to_tokens(control_question["question"])
    output = _generate_turn(loaded, hooks, tokens, temperature=temperature, seed=seed, max_new_tokens=max_new_tokens)
    response = loaded.model.to_string(output[0, tokens.shape[1] :])
    return {
        "question_id": control_question["id"],
        "question": control_question["question"],
        "expected_answer": control_question["expected_answer"],
        "response": response,
    }


def _generate_turn(
    loaded: "LoadedPrismModel",
    hooks: list,
    tokens: "torch.Tensor",
    *,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> "torch.Tensor":
    """Run one seeded ``generate()`` call under a fresh hook list.

    Seeding immediately before each call (rather than once per trial) is what
    makes a trial's naming turn reproducible independent of exactly how many
    tokens the detection turn happened to sample -- a later change to
    ``max_new_tokens`` on the detection turn would otherwise silently change
    which random draws the naming turn's own generation consumes.
    """
    import torch

    torch.manual_seed(seed)
    with loaded.model.hooks(fwd_hooks=hooks):
        return loaded.model.generate(
            tokens, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, verbose=False
        )


# --- trial-record schema, IDs, and the resumable JSONL log -----------------


def _trial_id(
    prompt_type: str,
    *,
    feature_id: int,
    layer: int,
    seed: int,
    strength: float | None = None,
    question_id: str | None = None,
) -> str:
    """Build a trial's identity from its own condition, not a counter or a
    random value, so the same condition always names the same trial and a
    resumed run can tell it apart from one still missing.

    ``layer`` is part of the ID on purpose. REQ-10 has not resolved the
    primary layer yet; when it does, trials at the new layer must get their
    own IDs rather than being silently treated as already-run duplicates of
    trials logged under the ADR-0009 fallback.
    """
    parts = [prompt_type, f"feature{feature_id}", f"layer{layer}"]
    if strength is not None:
        parts.append(f"strength{float(strength):g}")
    parts.append(f"seed{seed}")
    if question_id is not None:
        parts.append(f"q-{question_id}")
    return "::".join(parts)


def _build_record(
    *,
    trial_id: str,
    feature_id: int,
    layer: int,
    layer_source: str,
    strength: float | None,
    prompt_type: str,
    seed: int,
    temperature: float,
    model_response: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one ADR-0005 trial record, extended with the model/SAE
    checkpoint identity CLAUDE.md's reproducibility rule requires alongside
    layer, strength, seed, temperature, and the git commit -- the same
    provenance fields REQ-1/REQ-5's logged results already carry.

    ``feature_id`` is static input metadata (REQ-2's audit), not duplicated
    reasoning: ADR-0005 keeps identifiability score, decoder norm, and
    activation frequency in ``data/audit/features.csv``, joined onto this
    record by ``feature_id`` at analysis time, not copied into every row here.
    """
    return {
        "trial_id": trial_id,
        "feature_id": feature_id,
        "layer": layer,
        "layer_source": layer_source,
        "strength": strength,
        "prompt_type": prompt_type,
        "seed": seed,
        "temperature": temperature,
        "model_name": config["model"]["name"],
        "model_checkpoint_revision": config["model"]["checkpoint_revision"],
        "sae_checkpoint_repo": config["sae"]["checkpoint_repo"],
        "sae_checkpoint_revision": config["sae"]["checkpoint_revision"],
        "sae_checkpoint_sha256": config["sae"]["checkpoint_sha256"],
        "model_response": model_response,
        "judge_scores": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "excluded": False,
        "exclusion_reason": None,
    }


@functools.lru_cache(maxsize=1)
def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_existing_trial_ids(path: "str | Path") -> set[str]:
    """Read every ``trial_id`` already logged at ``path``, so a run can skip
    recomputing them. Missing file reads as "nothing logged yet", not an
    error -- the very first invocation against a fresh checkout has to work
    the same way a resumed one does.
    """
    path = Path(path)
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["trial_id"])
    return ids


def _append_record(path: "str | Path", record: dict[str, Any]) -> None:
    """Append one trial record as its own JSONL line.

    Strictly additive: never reads the file back to rewrite it, so an
    interrupted run loses at most the one trial in flight, and every trial
    logged before that stays exactly as written (ADR-0005's append-only
    convention for ``data/trials/``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# --- REQ-6: systematic trials ------------------------------------------------


def run_systematic_trials(
    config: dict[str, Any],
    loaded: "LoadedPrismModel",
    sampled_features: "pd.DataFrame",
    *,
    trials_path: "str | Path" = DEFAULT_TRIALS_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, int]:
    """Inject every sampled feature at every calibrated strength, at every
    configured seed, and log a detection/naming trial for each (REQ-6).

    Iterates ``sampled_features`` (REQ-2's stratified sample) x
    ``config["injection"]["strengths"]`` (REQ-5's calibrated band) x
    ``config["sampling"]["seeds"]``, at ``config["generation"]
    ["temperature_systematic"]`` (ADR-0008: temperature 1, the systematic
    trials that feed REQ-9's statistics, not a qualitative example). Returns
    ``{"run": n, "skipped": n}`` so a caller can see how much of a resumed
    run was already done.
    """
    layer = get_fallback_layer(loaded.model.cfg.n_layers)
    layer_source = "adr-0009-fallback"
    strengths = [float(s) for s in config["injection"]["strengths"]]
    seeds = [int(s) for s in config["sampling"]["seeds"]]
    temperature = float(config["generation"]["temperature_systematic"])

    existing_ids = _load_existing_trial_ids(trials_path)
    n_run = 0
    n_skipped = 0

    for _, feature_row in sampled_features.iterrows():
        feature_id = int(feature_row["feature_id"])
        decoder_atom = loaded.sae.W_dec[feature_id]

        for strength in strengths:
            for seed in seeds:
                trial_id = _trial_id("detection", feature_id=feature_id, layer=layer, strength=strength, seed=seed)
                if trial_id in existing_ids:
                    n_skipped += 1
                    continue

                def hooks_fn(pos: int, _atom: Any = decoder_atom, _strength: float = strength) -> list:
                    return inject_concept(loaded.model, _atom, layer=layer, strength=_strength, token_start_pos=pos)

                model_response = run_two_turn_trial(
                    loaded, hooks_fn, seed=seed, temperature=temperature, max_new_tokens=max_new_tokens
                )
                record = _build_record(
                    trial_id=trial_id,
                    feature_id=feature_id,
                    layer=layer,
                    layer_source=layer_source,
                    strength=strength,
                    prompt_type="detection",
                    seed=seed,
                    temperature=temperature,
                    model_response=model_response,
                    config=config,
                )
                _append_record(trials_path, record)
                existing_ids.add(trial_id)
                n_run += 1

    return {"run": n_run, "skipped": n_skipped}


# --- REQ-7: no-injection baseline -------------------------------------------


def run_baseline_trials(
    config: dict[str, Any],
    loaded: "LoadedPrismModel",
    sampled_features: "pd.DataFrame",
    *,
    trials_path: "str | Path" = DEFAULT_TRIALS_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, int]:
    """Run the same two-turn protocol as ``run_systematic_trials()`` with
    ``inject.no_injection()`` in place of an actual injection, to measure the
    false-positive rate (REQ-7).

    No strength dimension: a no-op hook behaves identically regardless of
    what strength it would have carried, so sweeping strength here would only
    duplicate the same no-injection generation under a misleading label.
    Iterates ``sampled_features`` x ``config["sampling"]["seeds"]`` instead.
    ``feature_id`` is still recorded on each row -- it names which feature a
    systematic trial at that same condition would have injected, useful for
    matching baseline and systematic rows one-to-one -- but carries no causal
    weight here, since nothing was actually injected.
    """
    layer = get_fallback_layer(loaded.model.cfg.n_layers)
    layer_source = "adr-0009-fallback"
    seeds = [int(s) for s in config["sampling"]["seeds"]]
    temperature = float(config["generation"]["temperature_systematic"])

    existing_ids = _load_existing_trial_ids(trials_path)
    n_run = 0
    n_skipped = 0

    for _, feature_row in sampled_features.iterrows():
        feature_id = int(feature_row["feature_id"])

        for seed in seeds:
            trial_id = _trial_id("baseline", feature_id=feature_id, layer=layer, seed=seed)
            if trial_id in existing_ids:
                n_skipped += 1
                continue

            def hooks_fn(pos: int) -> list:
                return no_injection(loaded.model, layer=layer, token_start_pos=pos)

            model_response = run_two_turn_trial(
                loaded, hooks_fn, seed=seed, temperature=temperature, max_new_tokens=max_new_tokens
            )
            record = _build_record(
                trial_id=trial_id,
                feature_id=feature_id,
                layer=layer,
                layer_source=layer_source,
                strength=None,
                prompt_type="baseline",
                seed=seed,
                temperature=temperature,
                model_response=model_response,
                config=config,
            )
            _append_record(trials_path, record)
            existing_ids.add(trial_id)
            n_run += 1

    return {"run": n_run, "skipped": n_skipped}


# --- REQ-7: unrelated-question control --------------------------------------


def run_control_trials(
    config: dict[str, Any],
    loaded: "LoadedPrismModel",
    sampled_features: "pd.DataFrame",
    *,
    control_questions_path: "str | Path" = DEFAULT_CONTROL_QUESTIONS_PATH,
    trials_path: "str | Path" = DEFAULT_TRIALS_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, int]:
    """Inject the same feature/strength grid ``run_systematic_trials()``
    uses, asking one of ``prompts.unrelated_control_prompt()``'s versioned
    questions instead of the detection question, to check for a generic
    yes-bias under injection independent of the concept itself (REQ-7).

    Single seed, ``config["sampling"]["seeds"][0]``: this control targets
    whether injection alone shifts the model toward "yes" on unrelated
    questions across the strength range, not the trial-to-trial variance a
    multi-seed sweep would add. Every question in the configured set gets
    used somewhere across the run: conditions are assigned questions by
    rotating through the set in a fixed, deterministic order (feature, then
    strength), so a resumed run assigns the same question to the same
    condition it would have on a first pass.

    No naming follow-up: control questions have no injected concept to name,
    so ``run_control_trial()`` makes one generation per trial, not two.
    """
    layer = get_fallback_layer(loaded.model.cfg.n_layers)
    layer_source = "adr-0009-fallback"
    strengths = [float(s) for s in config["injection"]["strengths"]]
    seed = int(config["sampling"]["seeds"][0])
    temperature = float(config["generation"]["temperature_systematic"])

    control_set = unrelated_control_prompt(control_questions_path)
    questions = control_set["questions"]

    existing_ids = _load_existing_trial_ids(trials_path)
    n_run = 0
    n_skipped = 0
    condition_index = 0

    for _, feature_row in sampled_features.iterrows():
        feature_id = int(feature_row["feature_id"])
        decoder_atom = loaded.sae.W_dec[feature_id]

        for strength in strengths:
            question = questions[condition_index % len(questions)]
            condition_index += 1

            trial_id = _trial_id(
                "control", feature_id=feature_id, layer=layer, strength=strength, seed=seed, question_id=question["id"]
            )
            if trial_id in existing_ids:
                n_skipped += 1
                continue

            tokens = loaded.model.to_tokens(question["question"])
            token_start_pos = tokens.shape[1] - 1
            hooks = inject_concept(loaded.model, decoder_atom, layer=layer, strength=strength, token_start_pos=token_start_pos)

            model_response = run_control_trial(
                loaded, hooks, question, seed=seed, temperature=temperature, max_new_tokens=max_new_tokens
            )
            record = _build_record(
                trial_id=trial_id,
                feature_id=feature_id,
                layer=layer,
                layer_source=layer_source,
                strength=strength,
                prompt_type="control",
                seed=seed,
                temperature=temperature,
                model_response=model_response,
                config=config,
            )
            _append_record(trials_path, record)
            existing_ids.add(trial_id)
            n_run += 1

    return {"run": n_run, "skipped": n_skipped}


def main() -> None:
    """CLI entry point: ``python -m prism.runner --config configs/experiment.yaml``.

    Runs all three trial types by default; ``--trial-types`` narrows to a
    comma-separated subset (useful for resuming just one type after a partial
    run, or for a smoke test that skips the two more expensive loops).
    """
    import pandas as pd
    import yaml

    from prism.models import load_model_and_sae

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--sampled-features", default="data/results/sampled_features.csv")
    parser.add_argument("--control-questions", default=DEFAULT_CONTROL_QUESTIONS_PATH)
    parser.add_argument("--trials-path", default=DEFAULT_TRIALS_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--trial-types",
        default="systematic,baseline,control",
        help="comma-separated subset of: systematic, baseline, control",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))

    loaded = load_model_and_sae(config)
    sampled_features = pd.read_csv(args.sampled_features)
    trial_types = {t.strip() for t in args.trial_types.split(",") if t.strip()}

    if "systematic" in trial_types:
        result = run_systematic_trials(
            config, loaded, sampled_features, trials_path=args.trials_path, max_new_tokens=max_new_tokens
        )
        print(f"systematic: {result}")
    if "baseline" in trial_types:
        result = run_baseline_trials(
            config, loaded, sampled_features, trials_path=args.trials_path, max_new_tokens=max_new_tokens
        )
        print(f"baseline: {result}")
    if "control" in trial_types:
        result = run_control_trials(
            config,
            loaded,
            sampled_features,
            control_questions_path=args.control_questions,
            trials_path=args.trials_path,
            max_new_tokens=max_new_tokens,
        )
        print(f"control: {result}")


if __name__ == "__main__":
    main()
