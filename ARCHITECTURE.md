# ARCHITECTURE.md — Prism: Decisions & Implementation Strategy

Each entry below follows the project's usual ADR format: Context, Decision, Alternatives Considered, Consequences, Status. Entries are append-only and numbered sequentially — if a decision changes, add a new ADR that supersedes the old one rather than editing it in place.

---

## ADR-0001: Activation hooking framework

**Context:** The experiment requires reading and writing residual-stream activations at a specific layer and token position during generation, on a GPT-NeoX-family model (Pythia).

**Decision:** Use **TransformerLens**. It has first-class support for the Pythia model family, exposes named hook points at every residual-stream location out of the box, and its `run_with_hooks` pattern maps directly onto the "add a scaled vector at a chosen layer during generation" intervention this project needs — no custom hook-registration code required.

**Alternatives considered:** Raw PyTorch forward hooks (more control, but reinvents functionality TransformerLens already provides correctly, and raises the risk of a hooking bug that silently corrupts every downstream result). `nnsight` (also viable, but the project's existing tooling and prior familiarity favor TransformerLens; not worth the switching cost for a 3-day sprint).

**Consequences:** Adds a dependency, but a well-tested one for exactly this use case.

**Status:** Accepted.

---

## ADR-0002: SAE source and training fallback

**Context:** The core independent variable (identifiability) is only meaningful if the injected vector is an actual SAE decoder atom from a dictionary matching the model being steered. REQ-1 (blocking, Phase 0) checks whether a Pythia-70m SAE compatible with the existing audit already exists.

**Decision:** If REQ-1 resolves "yes" — load via **SAELens** if the SAE is published in a SAELens-compatible format, otherwise write a minimal loader (a trained SAE is just a linear encoder/decoder pair; loading one that isn't in a standard registry is a small amount of code, not a research problem).

If REQ-1 resolves "no" — train a small SAE from scratch: collect residual-stream activations over a modest text corpus (a few million tokens is enough for a 70M-parameter model's residual stream), train a standard L1-sparsity SAE (single hidden layer, ReLU, sparsity penalty), and validate it reconstructs activations reasonably before treating its decoder atoms as meaningful. Budget 1–3 hours of GPU time for this path (see the SPEC's Compute Requirements, §6).

**Alternatives considered:** Skipping Pythia-70m entirely and using only Gemma Scope 2B (already has a published SAE). Rejected as the default because Pythia-70m's iteration speed is valuable during calibration (ADR-0008); Gemma Scope 2B remains the stretch-goal cross-check regardless (REQ-11).

**Consequences:** This decision is genuinely contingent — do not proceed past Phase 0 without resolving it, since every downstream module (feature selection, injection, identifiability scoring) assumes a working SAE.

**Status:** Accepted, pending REQ-1 resolution.

---

## ADR-0003: Injection implementation pattern

**Context:** Need a concrete, reproducible way to "inject" a concept vector that mirrors the source literature's method closely enough to be comparable, while working within TransformerLens's hook API.

**Decision:** Implement injection as an additive intervention via `run_with_hooks`: at the chosen layer, add `strength * normalized_decoder_atom` to the residual stream, starting at the token position immediately preceding the model's response and continuing through every generated token (matching the source protocol's persistent-injection design, not a one-shot single-token nudge).

**Alternatives considered:** Injecting only at the first generated token (rejected — the source literature's design persists the injection throughout the response, and a single-token nudge is a materially different intervention that would break comparability). Injecting via direct weight patching rather than activation hooking (rejected — unnecessarily complex for an additive intervention that hooking handles natively).

**Consequences:** Requires normalizing decoder atoms to a consistent scale before applying the strength multiplier, since raw decoder-atom norms vary across the dictionary — this normalization step must happen in the calibration pilot (REQ-5), not be assumed.

**Status:** Accepted.

---

## ADR-0004: LLM-judge model choice

**Context:** ~500–1,000 trial transcripts need scoring against a four-criterion rubric (detection, correct identification, pre-verbalization timing, coherence).

**Decision:** Use the **Anthropic API** (a capable current model) as the primary judge, given strong instruction-following on structured rubric-grading tasks. Validate on a hand-checked subsample of 10–15 transcripts before trusting it at scale (REQ-8). If API budget or rate limits become a constraint mid-sprint, fall back to a capable open-weight model as judge — but re-run the human-validation subsample against the new judge before switching, since judge behavior isn't assumed transferable.

**Alternatives considered:** A rule-based/keyword scorer (rejected — the rubric requires judging semantic correctness of concept naming and coherence, which keyword matching can't do reliably). Skipping human validation entirely to save time (rejected — an unvalidated judge risks silently corrupting every downstream statistic; the validation step is cheap relative to that risk).

**Consequences:** Introduces a token-budget cost (§6 of the SPEC estimates a few dollars total) — not a compute bottleneck, but worth tracking so scoring doesn't stall mid-Phase-1 on a rate limit.

**Status:** Accepted.

---

## ADR-0005: Data storage format

**Context:** Need a format for trial-level records that supports append-only logging, easy filtering/aggregation for stats, and doesn't require a database server for a 3-day sprint.

**Decision:** **JSONL**, one record per trial, in `data/trials/`, with a fixed schema: `{trial_id, feature_id, layer, strength, prompt_type, seed, temperature, model_response, judge_scores, timestamp, git_commit, excluded, exclusion_reason}`. Static feature metadata (identifiability score, decoder norm, activation frequency, tertile) lives separately in `data/audit/features.csv`, joined onto trial records at analysis time by `feature_id`, not duplicated into every trial row.

**Alternatives considered:** SQLite (more query power, but adds setup overhead and a schema-migration surface area not worth it for this scale and timeline). CSV for trial records (rejected — model responses are long, unstructured text; JSONL handles this far more cleanly than CSV's escaping rules).

**Consequences:** Analysis scripts need a join step (trials ⋈ features on `feature_id`) before regression — make this an explicit, single function in `stats.py`, not inlined ad hoc in multiple places.

**Status:** Accepted.

---

## ADR-0006: Statistical tooling

**Context:** Primary analysis is a logistic regression with covariates; secondary analysis is an AUC comparison between two classifiers.

**Decision:** **statsmodels** for the logistic regression (need proper standard errors and confidence intervals on the identifiability coefficient, not just a point estimate or a bare scikit-learn `.predict`). **scikit-learn** for the AUC comparison between the identifiability-based and norm-only-baseline classifiers, since that's a prediction-quality comparison rather than an inference question.

**Alternatives considered:** A single scikit-learn pipeline for everything (rejected — scikit-learn's logistic regression doesn't surface coefficient CIs as directly as statsmodels, and the report needs to show the identifiability coefficient with a confidence interval, not just a fitted model).

**Consequences:** Two libraries doing logistic regression for two different purposes is intentional here, not redundant — keep the two clearly separated in `stats.py` (`fit_inference_model()` vs. `compare_classifiers()`).

**Status:** Accepted.

---

## ADR-0007: Repository structure

```
prism/
├── CLAUDE.md
├── ARCHITECTURE.md
├── digital-minds-sprint-plan.md
├── prism-explainer.md
├── pyproject.toml
├── configs/
│   └── experiment.yaml          # model, SAE, layer, strengths, seeds — single source of truth per run
├── src/
│   └── prism/
│       ├── __init__.py
│       ├── models.py            # model + SAE loading                        (REQ-1, REQ-2)
│       ├── features.py          # stratified feature sampling from audit     (REQ-2)
│       ├── inject.py            # injection hooks, strength calibration      (REQ-3, REQ-5)
│       ├── prompts.py           # detection / control / naming templates     (REQ-4)
│       ├── runner.py            # trial execution, baseline & control runs   (REQ-6, REQ-7)
│       ├── judge.py             # LLM-judge scoring + validation harness     (REQ-8)
│       ├── stats.py             # regression + AUC comparison                (REQ-9)
│       └── layers.py            # UCARE compression-boundary layer lookup    (REQ-10, stretch)
├── data/
│   ├── audit/                   # read-only: identifiability scores, feature metadata
│   ├── trials/                  # append-only JSONL trial logs
│   └── results/                 # regenerable aggregate tables + figures
├── notebooks/                   # exploratory analysis only — no pipeline logic lives here
└── tests/
    ├── test_inject.py
    ├── test_features.py
    └── test_stats.py
```

**Status:** Accepted.

---

## ADR-0008: Reproducibility protocol

**Context:** Results need to be defensible in a research report with a genuine time crunch; sampling protocol affects both scientific validity and report credibility.

**Decision:** Mirror the source literature's sampling split: **temperature 0** for any qualitative example transcripts shown in the report (deterministic, reproducible, illustrative), **temperature 1** for every systematic trial that feeds the statistics, with multiple trials per condition to estimate variance (report standard error of the mean where relevant, matching how the source literature reports its own layer-sweep results). Every run logs its full config and a git commit hash alongside its output, per the CLAUDE.md non-negotiables.

**Alternatives considered:** Temperature 0 throughout, for maximum determinism (rejected — collapses the variance estimate the statistics need; a single deterministic sample per condition can't distinguish a real effect from noise).

**Status:** Accepted.

---

## ADR-0009: Layer selection mechanism

**Context:** The primary injection layer should be principled rather than an arbitrary fractional depth (the SPEC's stated preference over just copying the source literature's "~2/3 through the model" heuristic, which was tuned for a different model family).

**Decision:** Compute the compression-phase boundary from the existing UCARE intrinsic-dimension trajectory for Pythia-70m, and use that layer as the primary injection site. Implement this as a small standalone lookup (`layers.py`) that takes a precomputed intrinsic-dimension trajectory as input and returns a layer index — do not re-derive the trajectory from scratch inside this project; treat it as an input, same as the identifiability audit.

**Alternatives considered:** Just using a fixed fraction (e.g., 2/3 depth) as in the source paper. Kept as an explicit fallback, not the default: if the UCARE trajectory isn't readily available for the exact checkpoint in use, fall back to the fractional heuristic and note this substitution plainly in the report rather than silently treating it as the geometry-grounded choice.

**Status:** Accepted.

---

## ADR-0010: REQ-1 resolution — existing SAE, corrected base model

**Context:** ADR-0002 left the SAE dependency "accepted, pending REQ-1 resolution." REQ-1's investigation checked `data/audit/` (empty in this repo) and the separate project that actually produces the identifiability audit (`sae-bounding`), against SAELens's registry.

**Decision:** A residual-stream SAE for Pythia-70m-deduped exists and is the exact dictionary `sae-bounding` already scored — Hugging Face `ghidav/pythia-70m-deduped-sae`, path `test/blocks.4.hook_resid_pre/`, source revision `473774a054588503a90844f1afdb7b8fbf5f32a0`, SHA-256 `fdcb4553f5c4b44ddf04e5bfc98b0eddf71ee64c7de657af6eaa3d5e0c95b90f`. Identity confirmed by matching checksum between the locally cached file, `sae-bounding`'s own manifest, and the Hugging Face LFS object. ADR-0002's first branch applies: load via SAELens where the format allows, minimal loader otherwise. No training fallback.

This resolves against `EleutherAI/pythia-70m-deduped`, not the plain `EleutherAI/pythia-70m` `configs/experiment.yaml` named prior to this ADR — the SAE's own training config pins the deduped checkpoint explicitly. `configs/experiment.yaml` is corrected as part of REQ-1 to match.

The checkpoint's training config also records `sae_lens_version: 2.1.3`, four major versions behind this project's pinned `sae-lens==6.49.1`, and `normalize_sae_decoder: false` — the decoder atoms are not unit-normalized in the saved weights, which is why ADR-0003's injection-time normalization step is necessary rather than redundant.

Code review on REQ-1's pull request caught that only the SAE half of the pairing was pinned to a specific commit; `configs/experiment.yaml`'s `model:` block loaded `EleutherAI/pythia-70m-deduped` from whatever `main` currently resolves to. `HookedTransformer.from_pretrained` forwards a `revision` kwarg straight through to `AutoConfig.from_pretrained` and `AutoModelForCausalLM.from_pretrained`, so the same pinning mechanism already used for the SAE applies here too. `configs/experiment.yaml` now records `model.checkpoint_revision: e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (the repository's current commit, unchanged since 2023-07-09), and `load_model_and_sae()` passes it as `revision=` to `from_pretrained`.

**Alternatives considered:** None — this is a resolution of an already-accepted contingent decision, not a new choice between options.

**Status:** Accepted.

---

## ADR-0011: Audit data provenance — features.csv has two sources

**Context:** REQ-1's investigation surfaced a gap ADR-0005 didn't anticipate. `sae-bounding`'s identifiability audit computes frame-theoretic statistics (mutual coherence, Welch bound, coherence gap, cumulative coherence, RIP) per *dictionary* — one row per SAE checkpoint. REQ-2's stratified sampling and ADR-0005's `data/audit/features.csv` schema both need a score per *feature*. That doesn't exist yet, and neither does per-feature activation frequency, which `sae-bounding` has no way to produce since it never loads a base model.

**Decision:** `data/audit/features.csv` is assembled from two sources, not pulled wholesale from one existing artifact as ADR-0005 implied:
1. **Per-feature identifiability score** — produced by extending `sae-bounding` with a per-atom coherence function (the row-wise max of the Gram matrix `mutual_coherence()` already computes, retained instead of discarded after the global-max reduction). Stays in `sae-bounding`; this project still treats it as read-only input, consistent with ADR-0005's intent.
2. **Per-feature activation frequency** — measured inside this project by running Pythia-70m-deduped and the SAE loaded via REQ-1's `load_model_and_sae()` over a text corpus, since that requires a loaded model and `sae-bounding` is decoder-matrix-only by design. This also resolves `sae-bounding`'s own `pending_measurement` gap on this checkpoint's operating L0, as a byproduct of counting per-feature firing rates.

Decoder norm (the third covariate ADR-0005 lists) is a direct property of `W_dec` and needs no new measurement from either project.

This is tracked as a REQ-2 blocking dependency (issue #7), not implemented as part of REQ-1.

**Alternatives considered:** Computing the per-feature identifiability score inside this project instead of `sae-bounding` (rejected — duplicates geometry logic that already exists and is tested there; keeping the audit canonical in one place matches ADR-0005's separation of concerns). Treating the existing dictionary-level coherence number as a stand-in for every feature in that dictionary (rejected outright — that would erase the exact variation H1 is testing for).

**Status:** Accepted.

---

## ADR-0012: Dependency pins corrected against real execution, not just resolution

**Context:** `pyproject.toml`'s original pins (`torch==2.13.0`, plus whatever `transformer-lens==3.7.1`'s unbounded `transformers>=5.9.0` constraint happened to resolve to) were verified only via `uv pip install --dry-run` during scaffolding, a dependency-resolution consistency check, not an actual run. Executing REQ-1 for real surfaced three genuine breaks, caught by code review on REQ-1's pull request rather than documented at the time.

**Decision:** Pin to versions confirmed to load and run correctly, not merely to resolve:

- `torch==2.13.0` fails `WinError 1114` (native DLL initialization failure) loading `c10.dll` on the machine this project is developed on, reproduced identically across two independent Python interpreters and several environment-variable workarounds. `torch==2.6.0` loads cleanly and was used for every real execution in this project so far.
- `sentencepiece==0.2.2` segfaults on a bare `import sentencepiece`, independent of torch or transformer-lens entirely; a clean reinstall didn't change the outcome, ruling out a corrupted download. `sentencepiece==0.2.0` doesn't segfault, and is now pinned explicitly rather than left as an unpinned transitive dependency.
- `transformer-lens==3.7.1` declares `transformers>=5.9.0` with no upper bound, which resolved to `5.15.0`. That version renamed GPT-NeoX's output head (`embed_out` to `lm_head`), which transformer-lens's `convert_neox_weights` doesn't handle, breaking `HookedTransformer.from_pretrained` for Pythia entirely. Pinned to `5.9.0`, transformer-lens's own declared floor, which still has the old attribute name.

None of these are portability guarantees for every machine and accelerator; they are the versions verified to work on the machine this project's real runs have used so far. If a future run on different hardware needs a different pin, that's a new decision to document, not a silent swap.

**Alternatives considered:** Leaving the original pins and working around the failures per-machine (rejected: the whole point of pinning versions in `pyproject.toml` is a run any collaborator can reproduce, and an unpinned or broken-pin dependency defeats that). Pinning to the newest version that happens to work rather than transformer-lens's declared floor for `transformers` (rejected for that specific pin: the declared floor is the version the library's own maintainers verified against, a narrower and more defensible choice than picking an arbitrary newer point in an unbounded range).

**Status:** Accepted.

---

## ADR-0013: REQ-2 resolution — per-feature audit table assembled, corpus pinned

**Context:** ADR-0011 left the per-feature identifiability score and activation-frequency measurement as a confirmed-but-unimplemented plan, tracked in issue #7. REQ-2 needed both before `stratified_sample()` had anything real to stratify on.

**Decision:** Both pieces are implemented as ADR-0011 described, plus one addition ADR-0011 didn't specify: which text corpus activation frequency gets measured against.

Per-feature identifiability now exists in `sae-bounding`, not this repo. `feature_coherence()` (`src/saeframe/frame/coherence.py`) exposes the row-wise max of the Gram matrix `mutual_coherence()` already builds, and `scripts/05_compute_feature_identifiability.py` runs it against one registered cohort at a time, loading only that cohort's checkpoint record rather than the whole registry (the registry has since grown Gemma Scope entries whose files aren't all cached locally, and a per-cohort script has no reason to require them). Run against `pythia-70m-deduped-residual-layer-4` (16,384 features, checksum `fdcb4553f5c4b44ddf04e5bfc98b0eddf71ee64c7de657af6eaa3d5e0c95b90f`, the same checkpoint REQ-1 resolved to), it produced `results/per_feature/pythia-70m-deduped-residual-layer-4.csv`: exact scores (not sampled, unlike the existing dictionary-level `mutual_coherence` column in `audit_table.csv`, which was computed from a 1,024-column random subsample and is expected to disagree slightly with this exact computation's maximum). Committed on `sae-bounding` branch `req-2/per-feature-identifiability` at `105223f`, PR #6, not yet merged at the time this ADR was written.

Activation frequency is measured inside this project, per ADR-0011, using two new `models.py` functions: `measure_activation_frequencies()` (runs pre-tokenized text through the loaded model and SAE, counts each feature's nonzero-encoding rate) and `decoder_norms()` (reads `W_dec`'s row-wise norm directly, no measurement needed, confirming ADR-0011's note that this covariate was always just sitting in the loaded weights).

The corpus for that measurement is `NeelNanda/pile-10k` (Hugging Face dataset, revision `127bfedcd5047750df5ccf3a12979a47bfa0bafa`), a 10,000-document sample of the Pile. Pythia was trained on the Pile, so this keeps the frequency measurement on-distribution for the checkpoint being measured, and it's an established choice in the wider mechanistic-interpretability community for exactly this kind of small-model activation statistic, not a one-off pick. The actual run used the first 500 documents, each truncated to at most 256 tokens, for 111,233 tokens total, sized for a same-session turnaround (a few minutes on CPU) rather than exhaustive coverage — consistent with SPRINT-PLAN.md §3.2's "recomputed quickly if not already logged." 77 of 16,384 features never fired across that corpus; that's a real result of a finite sample against a large, long-tailed dictionary, not an error, and is left as a legitimate zero rather than smoothed.

`src/prism/audit_build.py` is a new module ADR-0007's repository tree didn't list. It's a one-time data-preparation script, invoked as `python -m prism.audit_build --identifiability-csv <path> --identifiability-source-commit <sha>`, that joins the three sources above by feature index and writes `data/audit/features.csv` plus a provenance record (`data/results/req2_feature_audit_provenance.json`) with the model, SAE, and corpus identity, and a git commit hash. It is not part of the per-run experiment pipeline `features.py` exposes and is not expected to run again unless the upstream audit or the corpus choice changes.

The provenance record identifies the identifiability CSV by `identifiability_source_repo`, `identifiability_source_commit` (the sae-bounding commit that produced it, required with no default since it changes every regeneration and shouldn't be guessed), and `identifiability_source_sha256` (the file's own checksum, computed at build time). It does not record the local filesystem path the CSV was read from, which was machine-specific and not itself part of the file's identity. `build_feature_audit_table()` checks `identifiability_source_commit` is a well-formed 40-character git SHA before doing anything else, but that's a format check, not a verification — it still isn't checked against the commit that actually produced `identifiability_csv`, because that requires a pinned expected value to check against, and `sae-bounding` PR #6 hasn't merged yet (tracked in issue #17). Blocking the build entirely until that verification exists was considered and rejected for now: REQ-2 needed `data/audit/features.csv` to exist well before #6 could realistically merge, and a hard block would have made this ADR's own resolution impossible to land. The gap is documented, not hidden, and #17 stays open until it's closed for real.

**Alternatives considered:** A corpus native to this repo (e.g., reusing `RECONSTRUCTION_VALIDATION_PROMPTS`, REQ-1's 8-sentence, 120-token set) — rejected as too small to produce a meaningful per-feature rate across 16,384 features. Streaming the full Pile or a larger slice — rejected as unnecessary for a covariate used to balance a sample, not to estimate the primary effect, and disproportionate to a 3-day sprint's time budget.

**Status:** Accepted.

---

## Implementation Strategy: Build Order

Maps directly onto the SPEC's phases (`digital-minds-sprint-plan.md` §4). Build in this order — later modules depend on earlier ones, and building out of order risks writing against an interface that hasn't stabilized yet.

**Phase 0 (today):**
1. `configs/experiment.yaml` skeleton + `models.py` — resolve REQ-1 (the SAE dependency check) before writing anything else; this is the one decision that changes everything downstream if it goes the "train from scratch" way (ADR-0002).
2. `features.py` — stratified sampler, once the audit table's feature-metadata format is confirmed.
3. `inject.py` — hook implementation (no calibration yet, just the mechanism).
4. `prompts.py` — the three prompt templates.

**Phase 1 (Aug 15):**
5. `inject.py` calibration routine (REQ-5) — small pilot, find the working strength band.
6. `runner.py` — full trial runner plus baseline/control batches (REQ-6, REQ-7).
7. `judge.py` — scoring pipeline, human-validated before trusting it at scale (REQ-8).
8. `stats.py` — primary regression + AUC comparison (REQ-9).

**Phase 2 (Aug 16):**
9. `layers.py` (stretch, REQ-10) and a Gemma Scope adapter in `models.py` (stretch, REQ-11), only if Phase 1 finished with slack.
10. Figure generation (matplotlib scripts reading from `data/results/`) feeding directly into the report.
11. Report assembly — by this point every number in the report should trace back to a `data/results/` file, not be transcribed by hand from a notebook.

If Phase 1 runs long, cut in this order: REQ-11 (Gemma Scope) first, then REQ-10 (layer sweep) — both are explicitly marked stretch in the SPEC, and the core H1 result (identifiability vs. detection on Pythia-70m at one principled layer) is a complete, submittable result on its own.
