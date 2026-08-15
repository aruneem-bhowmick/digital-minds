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

## ADR-0014: REQ-2 resolution — cross-repo identifiability checksum verified (closes #17)

**Context:** ADR-0013 recorded the identifiability CSV's checksum but never checked it against anything, since `sae-bounding` PR #6 hadn't merged and there was no stable value to pin. That PR has since merged.

**Decision:** `configs/experiment.yaml` gains an `identifiability_source:` block (`repo`, `commit`, `checksum`), matching the reproducibility pattern the `sae:` block already uses for the SAE checkpoint. `build_feature_audit_table()` now verifies `identifiability_csv`'s SHA-256 against `identifiability_source.checksum` before reading anything else from it, raising a clear error naming both the actual and expected checksum on a mismatch, instead of only recording whatever the file happened to contain. `identifiability_source_commit` and `identifiability_source_repo` moved from required CLI arguments to config fields, since they're no longer values that change on every invocation — they're a pinned identity, the same way the SAE checkpoint's revision is.

Pinned values: repo `aruneem-bhowmick/sae-bounding`, commit `09cc4a4` (the PR #6 merge commit), checksum `b6ded2c9…03567a` (full value in config).

The merge itself included one change beyond what ADR-0013 described: `feature_coherence()`'s reduction changed from `np.max(off_diagonal, axis=0)` to `axis=1`, with the accompanying unit test changed to a non-symmetric input to make the two axes produce different results. Checked before trusting it: a Gram matrix (`G = D^T @ D`) is symmetric by construction, and for a symmetric matrix `axis=0` and `axis=1` reductions of the same array are identical — proven directly (`np.array_equal`, real SAE decoder, zero difference) and confirmed by regenerating `results/per_feature/pythia-70m-deduped-residual-layer-4.csv` from the merged code: byte-identical to the pre-merge file. The change doesn't affect this project's data. The test case that "proves" a difference does so with a matrix that couldn't arise as a real Gram matrix (it isn't symmetric), which is worth knowing if that test is ever used as a template elsewhere, but isn't this project's concern to fix in a repo it doesn't own.

**Alternatives considered:** Keeping `identifiability_source_commit` as a required CLI argument even after the merge — rejected once a real pinned value existed to check against; matching the SAE checkpoint's own config-driven pattern is more consistent and makes the CLI simpler, not more complex, for the common case.

**Status:** Accepted.

---

## ADR-0015: REQ-4 resolution — prompt templates, control-set storage, and criterion mapping

**Context:** `SPRINT-PLAN.md` §3.4 and the REQ-4 build-order entry call for three prompt templates and a versioned unrelated-question control set, but neither the SPEC nor ADR-0007's repository tree says where that control set should live or how "versioned" should be implemented. That gap needed a decision before `unrelated_control_prompt()` could be written, not an assumption baked silently into the code.

**Decision:** `src/prism/prompts.py` implements three functions, each targeting one of the criteria named in the REQ-4 prompt sequence (detection, naming/accuracy, internality):

- `detection_prompt()` returns a fixed string asking the model to check its own current processing for anything unusual and to answer with a plain yes or no before elaborating. The wording never mentions injection or steering, so the identical prompt serves REQ-6's injected trials and REQ-7's no-injection baseline without a second variant. Asking for a yes/no verdict before any description is a deliberate ordering, not a style choice: it is what makes "detection before verbalization" (Morris & Plunkett's causal-bypassing criterion, `SPRINT-PLAN.md` line 33) checkable at all from the transcript alone.
- `naming_subtask_prompt()` returns a fixed follow-up string, meaningful only after an affirmative response to `detection_prompt()`, asking the model to identify what it noticed.
- `unrelated_control_prompt()` returns the full contents of a small, versioned control-question set: at least 8 to 10 yes/no questions spanning unrelated domains, each with an expected answer of "no," matching `SPRINT-PLAN.md` §3.4's "default-negative expected answer" description of Lindsey's own bias control. This targets internality specifically: an affirmative answer to `detection_prompt()` only supports a genuine internal signal if the same model, under the same injection, does not also default to "yes" on questions with nothing to do with the injected concept.

The control-question set lives in `configs/control_questions.yaml`, not `data/audit/`, `data/trials/`, or a Python constant inside `prompts.py`. `configs/` already holds `experiment.yaml`, the project's other externally editable, non-code experiment input (CLAUDE.md §6); the control questions are the same kind of artifact; a researcher should be able to add or reword a question without touching Python. `data/audit/` is reserved for the read-only identifiability audit (ADR-0005, ADR-0011) and `data/trials/` and `data/results/` for run output, neither of which describes a fixed set of prompt text. The file carries a top-level `version` string and, per question, a stable `id`, the `question` text, and `expected_answer`. `unrelated_control_prompt()` validates the loaded file before returning it (`version` present, at least 8 questions, unique ids, unique question text, every `expected_answer` equal to `"no"`) and raises a specific error naming which rule failed, the same defensive pattern `load_feature_audit()` already uses on `data/audit/features.csv`.

None of the three functions reproduce Lindsey (2025)'s own wording. `SPRINT-PLAN.md`'s four-criterion framing (line 12: accuracy, grounding, internality, metacognitive representation) and the REQ-4 build sequence's own framing (detection, naming/accuracy, internality, coherence) name the criteria slightly differently; this project's code and docstrings use the REQ-4 framing consistently, since that is the wording the build sequence itself docstrings were asked to cite. Coherence has no dedicated template: it is judged by REQ-8 from whatever text a trial produces, regardless of which of the three prompts elicited it, so no function here claims to probe it.

**Alternatives considered:** A Python constants list inside `prompts.py`, matching `models.py`'s `RECONSTRUCTION_VALIDATION_PROMPTS` pattern (rejected: that list is a fixed, one-off validation corpus not meant for routine editing, while the control-question set is explicitly meant to be auditable and extendable without a code change, per the REQ-4 build sequence). Storing the control set under `data/` (rejected: `data/` is reserved for audit input and run output per ADR-0005 and ADR-0011, and this file is neither).

**Status:** Accepted.

---

## ADR-0016: REQ-5 resolution — pilot feature source, fallback layer, and the calibrated strength band

**Context:** `SPRINT-PLAN.md` §3.3 specifies the pilot's shape (~5 features x 3-4 strengths, temperature 0) but leaves two implementation choices open, and REQ-5 needed both settled before the pilot could run at all: where the pilot's features come from, and what layer to inject into before REQ-10 has picked one.

**Decision:** `select_pilot_features()` draws its ~5 features from REQ-2's already-sampled, tertile-labeled population (`data/results/sampled_features.csv`), not a fresh draw from the full audit table. The pilot exists to calibrate a strength band for REQ-6's systematic trials, and REQ-6 injects exactly this sampled population; drawing the pilot from a separately randomized set risked calibrating against features the systematic trials would never touch.

The primary injection layer is REQ-10's decision, and REQ-10 hasn't run. Per ADR-0009's explicit fallback, `layers.get_fallback_layer(n_layers, fraction=2/3)` is implemented now, as the one piece of `layers.py` ADR-0009 already fully specifies independent of the UCARE trajectory `get_compression_boundary_layer()` will need. On Pythia-70m-deduped's six transformer blocks this resolves to layer 4, `blocks.4.hook_resid_pre`, the same hook point the SAE was already trained against. Every pilot record carries `layer_source: "adr-0009-fallback"` explicitly, and `configs/experiment.yaml`'s `injection.layer` field stays `TODO`: this is a stopgap value used to run the pilot, not the resolved REQ-10 choice, and the shared config should not imply otherwise.

The pilot ran against the real model and SAE at candidate strengths 1, 2, 4, 6, 8, 16, 32, and 64, all committed in full to `data/results/calibration_pilot.jsonl` rather than discarding the strengths outside the final band: CLAUDE.md's rule against curating out incoherent trial output applies to the raw generations behind this decision, not only to the systematic trial data collected later. Strength 0 already loops into a repeated short phrase by 60 tokens on this raw, non-instruction-tuned checkpoint, which set the bar for "too weak": a strength that reproduces that same baseline pattern isn't showing an effect. Strength 1 was borderline (a literal no-op for one of five pilot features). Strength 2 produced the pilot's clearest coherent, on-topic response. Strengths 4-6 held together as (repetitive) English: one feature's output uses ordinary Unicode curly quotation marks that render oddly in some terminal fonts, not corrupted text, which an earlier draft of the pilot notes mischaracterized before being corrected against the raw record. Strength 8 was consistently degenerate across features, and the wider sweep (16-64) confirmed that failure mode gets uniformly worse with no exceptions, matching `SPRINT-PLAN.md`'s documented "brain damage" pattern. `configs/experiment.yaml`'s `injection.strengths` is set to `[1, 2, 4, 8]`: the full observed transition from weak/borderline through the confirmed too-strong boundary, not narrowed to only the strengths that looked cleanest, since REQ-9's regression already treats strength as a covariate and REQ-6 needs real coverage of the failure mode, not just the middle of the band. Full pilot records and per-strength notes live in `data/results/calibration_pilot.jsonl` and `data/results/calibration_notes.md`.

**Alternatives considered:** Drawing pilot features fresh from the full audit table rather than REQ-2's sample (rejected: would let the pilot calibrate against features REQ-6 never actually injects). Narrowing the final strength band to only the strengths that produced legible text, e.g. dropping 8 (rejected: CLAUDE.md's rule against curating out incoherent trials applies to the calibration decision itself, not just to trial data collected later; a band that never samples the too-strong boundary would leave REQ-9's strength covariate untested at the one condition the source literature specifically flags).

**Status:** Accepted.

---

## ADR-0017: REQ-6/REQ-7 resolution — two-turn trial protocol, record shape, and trial volume

**Context:** `SPRINT-PLAN.md` §3.3–§3.4 describes the systematic-trial, baseline, and control conditions, and ADR-0005 fixes the JSONL field list, but neither settles four implementation questions REQ-6/REQ-7 needed answered before the runner could write a single record: how a two-turn detection-then-naming exchange fits into ADR-0005's single `model_response` field, how the runner decides at generation time whether to ask the naming follow-up at all, how the injected concept stays active through that second turn without reusing a hook list `inject.py` already refuses to reuse, and how many trials each of the three batches should actually run given `SPRINT-PLAN.md` §6's ~500–1,000 trial budget.

**Decision:** `model_response` is a structured object, not a flat string, and its shape depends on `prompt_type`. Systematic (`"detection"`) and baseline (`"baseline"`) trials record `{"detection": {"prompt", "response"}, "affirmative": bool, "naming": {"prompt", "response"} | None}`; control trials record `{"question_id", "question", "expected_answer", "response"}`. REQ-8's judge already has to branch its grading logic on `prompt_type` (a control trial has no naming turn to grade for timing or accuracy), so branching on the same field to read `model_response` is not new coupling.

Whether to ask the naming follow-up is decided by `runner.is_affirmative()`, a direct regex read of the first word of the detection response (`^\W*yes\b`, case-insensitive) against `prompts.detection_prompt()`'s own request for a yes/no answer before any elaboration. This is a fast, low-level parse, fully separate from REQ-8's judge, which grades the complete transcript after the fact. The two are not meant to agree in every case: an LLM judge might reasonably read a hedged or indirect response as an affirmative detection where a literal first-word regex does not, but the runner has to make some decision immediately, before a follow-up can be asked or skipped, and a judge model is not in that loop yet.

The naming follow-up's continuation is built from the detection turn's own generated token IDs (`torch.cat([detection_output, suffix_tokens], dim=1)`), never from re-tokenizing the detection prompt and response pasted together as a string. Re-tokenizing risks the join point landing on different token boundaries than the original generation used, which would silently shift `token_start_pos` in the new sequence and either miss part of the intended injection window or inject into prompt tokens that must stay clean. Token-ID concatenation keeps `token_start_pos` exactly aligned across both `generate()` calls, so a fresh hook built for the second call injects starting from the same real position the first call used, letting the concept stay active through the whole two-turn exchange (ADR-0003) rather than only the detection turn.

Trial volume, against `configs/experiment.yaml`'s REQ-2/REQ-5 grid (40 sampled features, strengths `[1, 2, 4, 8]`, seeds `[0, 1, 2]`), all at the single ADR-0009 fallback layer since REQ-10 has not run:
- Systematic: features × strengths × seeds = 40 × 4 × 3 = 480 trials.
- Baseline: features × seeds, no strength dimension, since a no-injection hook behaves identically regardless of what strength it would have carried = 40 × 3 = 120 trials.
- Control: features × strengths, at the single canonical seed (`sampling.seeds[0]`), one control question per condition rotated deterministically through `configs/control_questions.yaml`'s full set so every question gets used across the run = 40 × 4 = 160 trials.

Total 760 trials, inside `SPRINT-PLAN.md` §6's ~500–1,000 estimate. Control uses one seed rather than all three: the question this batch answers is whether injection itself shifts the model toward "yes" on unrelated questions across the strength range, not the trial-to-trial variance a multi-seed sweep would add, and three seeds here would have pushed total volume well past budget for a comparatively smaller gain than covering the strength range and the full question set does. All three batches share one file, `data/trials/trials.jsonl`, with `trial_id` built from each trial's own condition (prompt type, feature, layer, strength where relevant, seed, control question ID where relevant) rather than a counter, so a resumed run can tell an already-logged trial apart from a missing one without re-deriving it from row order. `layer` is part of every `trial_id` on purpose: when REQ-10 resolves the primary layer for real, trials at the new layer need their own IDs, not to be silently skipped as duplicates of the fallback-layer trials logged here.

**Alternatives considered:** Sweeping all three seeds for the control batch, matching systematic exactly (rejected: would have pushed total trial volume to roughly 1,120, past the SPEC's stated budget, for a check whose main value is breadth across strengths and questions rather than seed-level variance). Deciding affirmative/negative by asking REQ-8's judge inline during the trial run rather than a first-word regex (rejected: REQ-8 is explicitly out of scope for this module, and the runner needs an answer before the naming follow-up can be asked or skipped, not after a batch judge pass days later). Re-tokenizing the concatenated prompt-plus-response text for the naming turn instead of concatenating token IDs (rejected outright once the token-boundary drift risk was identified: it would have made `token_start_pos` wrong in a way that could silently corrupt the second turn's injection window).

**Status:** Accepted.

---

## ADR-0018: REQ-8 resolution — judge model, concept grounding via top-activating context, and the detection/control grading split

**Context:** ADR-0004 accepted "the Anthropic API" as the judge without pinning a model id, deferred to REQ-8 (flagged in `.claude/flagged-decisions.md` #1). Separately, `SPRINT-PLAN.md` §3.5's four-criterion rubric ("affirmative detection, correct concept identification, ... detection prior to verbalizing the concept, and output coherence") assumes a known concept exists per feature to grade naming accuracy against. `data/audit/features.csv` (ADR-0011/ADR-0013) carries no such label — only `identifiability_score`, `decoder_norm`, `activation_frequency` — and neither `sae-bounding` nor this project's own provenance records one anywhere. No earlier ADR anticipated this gap; it surfaced only once `judge.py` needed a real value to compare naming responses against.

**Decision:**

Judge model: `claude-opus-4-8`. Every judge call omits `temperature`/`top_p`/`top_k` (rejected outright on this model) and does not request extended thinking — a rubric-grading task over short transcripts doesn't need it, and skipping it keeps the roughly 760-trial scoring pass inside the token budget `SPRINT-PLAN.md` §6 estimates. `configs/experiment.yaml`'s `judge.model` field, previously `TODO`, is set to this value.

Concept grounding: rather than inventing a per-feature label, `judge.py` derives grounding evidence from the same corpus REQ-2 already measured activation frequency against (`NeelNanda/pile-10k`, ADR-0013's pinned dataset revision). For each feature actually sampled and injected in REQ-6's systematic trials, `models.top_activating_snippets()` finds the token positions where that feature's SAE-encoded value is highest across the corpus and returns the surrounding text window for each. This mirrors how SAE decoder atoms are conventionally interpreted in the auto-interpretability literature — max-activating dataset examples as a feature's de facto identity — and avoids fabricating a label, which CLAUDE.md's non-negotiables rule out outright. The judge receives these snippets as reference evidence and is asked whether the model's naming-turn response plausibly names the same concept, rather than doing an exact-string match against a fixed label. Output: `data/results/feature_concept_grounding.json`, keyed by `feature_id`, carrying the same corpus/model/SAE provenance fields `req2_feature_audit_provenance.json` already uses. A feature that never fires in the corpus — a real, previously-documented outcome (ADR-0013 records 77 of 16,384 features never firing over the full corpus) — gets an empty snippet list; `score_trial()` tells the judge explicitly that no grounding evidence was found, rather than handing over an empty list silently, so a "cannot confirm" grade reads differently from "confirmed wrong."

Grading schema splits by `prompt_type`, following ADR-0017's precedent for the same split in `model_response`'s own shape:
- `detection`/`baseline` trials (two-turn): `{detected, concept_identified, concept_confidence, identified_before_verbalizing, coherent, reasoning}`. `concept_identified` and `identified_before_verbalizing` are `null` when `detected` is false, since there is no naming turn to grade in that case — the same convention `runner.py` already uses for `model_response["naming"]`.
- `control` trials (single turn, no concept to name): `{affirmative, coherent, reasoning}`.

Both schemas are enforced via `output_config.format` (structured JSON output), not free-text parsing, so a malformed judge response surfaces as a hard error rather than a silently mis-parsed score.

**Alternatives considered:** A hand-authored concept label per sampled feature (rejected: with 40 sampled features and no existing interpretability artifact for this specific SAE checkpoint, authoring labels by inspection is either slow enough to blow the sprint's remaining budget or shortcut enough to risk being an uninformed guess dressed up as ground truth — worse than grounding in the model's own real firing pattern). A single shared JSON schema across every `prompt_type` with unused fields left `null` (rejected: control trials have no naming turn at all, and Lindsey's four criteria don't apply to them the same way a shared schema implies — leaving fields structurally present but meaningless would require every caller to already know which ones to ignore).

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
