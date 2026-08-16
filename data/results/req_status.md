# REQ status report

QA pass before this codebase is used to write the research report (`.claude/prompts.md` Prompt 12). Everything below was checked directly in this session, not assumed from an earlier ADR or a script that merely exited without an error.

## QA findings

**Test suite.** `pytest -q` from the repo root: 258 passed, 5 skipped, 0 failed. The 5 skips are all explicit `# N/A: <reason>` stubs or environment-gated cases (a negative-identifiability case ruled out upstream, a judge test that needs a real `ANTHROPIC_API_KEY`, a wording-reproduction check that would have to quote the source text it's checking against, and two stats.py cases already ruled out by an upstream guarantee), not silently disabled coverage. One side effect worth recording: `tests/test_models.py::test_load_model_and_sae_returns_a_working_pair` is a real integration test (loads the actual model and SAE, no mocks) that calls `save_reconstruction_result()` with its default output path, so every full test-suite run rewrites `data/results/req1_sae_validation.json` with a fresh `git_commit`/`timestamp`. The measured value itself (`fraction_variance_explained = 0.9808...`) reproduced identically across the runs in this session, so this is a real re-validation, not drift. The committed file was restored to its last-committed state after each run in this session so this branch's diff stays about figures and QA, not an incidental timestamp bump.

**Trial-record completeness.** Every record in `data/trials/trials.jsonl` (760 total) has either a non-null `judge_scores` or a non-null `exclusion_reason`: 752 scored, 8 excluded, 0 with neither. The 8 excluded all carry `exclusion_reason` values starting `judge refused to grade trial ... (category='bio')`, matching ADR-0019 and `judge_scoring_provenance.json`'s `"refused": 8`.

**Judge validation.** `data/results/judge_validated.flag` exists and its `reviewer_note` is a substantive, specific record of a human review, not a placeholder: it states the 752/8 scored/excluded split, names the refusal category and gives two concrete example trigger phrases ("tabacum", "wear-time responses"), confirms the 15-trial stratified sample in `judge_validation_sample.md` was read in full, separately describes the one `detected: true` trial and the runner/judge divergence behind it, and states the known limitation that zero trials in the dataset have a real naming turn. Read in full for this pass, not just checked for existence.

**Placeholder/TODO grep.** `grep -rniE "TODO|FIXME|XXX|HACK|placeholder|dummy|mock|fake|synthetic"` across `src/`, `tests/`, `configs/`, and `data/results/` (excluding `__pycache__`) turned up:
- `configs/experiment.yaml`: `injection.layer: TODO`. Intentional and documented (ADR-0009, ADR-0016). The primary layer is REQ-10's decision, REQ-10 was not attempted this session, and the config says so plainly rather than guessing a value. Not a leftover.
- `tests/test_models.py`: asserts the config's `model`/`sae` fields are *not* `"TODO"`, a real regression test, not a stub.
- `src/prism/figures.py`: the phrase "placeholder chart" appears once, inside a docstring describing what the layer-comparison function must *not* do. Not a placeholder itself.
- `src/prism/layers.py`: "is not implemented here yet" describes `get_compression_boundary_layer()`, which genuinely does not exist because REQ-10 was not attempted. Documented, not silent.
- Instances of "fake" and "mock" inside `data/trials/trials.jsonl` and `data/results/analysis_table.csv` are substrings of real, degenerate model output text (a generated `http://fakelike.org/...` string, or judge reasoning describing a "fake journal citation" the model hallucinated). Real experimental data, not fabricated records.

No other placeholder, TODO, or hardcoded example value was found.

**Provenance.** Every number-bearing file in `data/results/` traces to a real, existing git commit and a logged config:
- `regression_results.json`, `auc_comparison.csv`: `git_commit: bb70525...`, both fitted from `data/trials/trials.jsonl` per the config `configs/experiment.yaml` pins.
- `req1_sae_validation.json`: `git_commit: 254cf0d...` (current HEAD as of this session; see the test-suite side effect above), reconstruction validated against the pinned SAE checkpoint.
- `req2_feature_audit_provenance.json`, `sampled_features.csv`: `git_commit: ba890d0...`; `sampled_features.csv` has no embedded provenance fields of its own but is traceable through `git log` (`ba890d0`, `8c75570`, `c54e9db`).
- `feature_concept_grounding.json`: `git_commit: 6aa5482...`, includes the corpus identity (`NeelNanda/pile-10k`, pinned revision) it was measured against.
- `judge_scoring_provenance.json`: `git_commit: 668d66b...`, judge model `claude-opus-4-8`, scored/skipped/refused counts.
- `calibration_pilot.jsonl`: every row carries its own `git_commit` (`933eeb7...`); `calibration_notes.md` cites the same commit.
- `judge_validated.flag`, `judge_validation_sample.md`: no embedded `git_commit` field, but both are version-controlled and traceable via `git log` (`85da775`).
- `data/results/figures/*`: generated this session by `python -m prism.figures` against the committed `regression_results.json`, `analysis_table.csv`, and `calibration_pilot.jsonl` above. Inherits their provenance rather than carrying its own, since a figure is a rendering of already-provenanced numbers, not a new measurement.

All six commit hashes above were checked directly against this repository's object store (`git cat-file -t <hash>`) and confirmed to exist as real commits, not invented placeholders.

## REQ-1 through REQ-14

| REQ | Status | Note |
|---|---|---|
| REQ-1 | Done | SAE dependency resolved to `ghidav/pythia-70m-deduped-sae` (ADR-0010); `load_model_and_sae()` implemented; reconstruction validated at `fraction_variance_explained = 0.9808` against real tokens. |
| REQ-2 | Done | `features.py` implements the audit loader, stratified sampler, and balance check; `data/audit/features.csv` (16,384 features) and `data/results/sampled_features.csv` (40 sampled, tertile-labeled) exist with real provenance. |
| REQ-3 | Done | `inject.py` implements the persistent injection hook; zero-vector no-op and hook-cleanup are both tested explicitly. |
| REQ-4 | Done | `prompts.py` implements the three templates; the control-question set is externally editable in `configs/control_questions.yaml`. |
| REQ-5 | Done | Real pilot (`data/results/calibration_pilot.jsonl`, 40 trials) and `calibration_notes.md` document the too-weak/usable/too-strong transition; `configs/experiment.yaml`'s `injection.strengths` reflects it. |
| REQ-6 | Done | `runner.py`'s `run_systematic_trials()` ran for real; systematic trials are present in `data/trials/trials.jsonl` with a resumable, deduplicated writer. |
| REQ-7 | Done | `run_baseline_trials()`/`run_control_trials()` ran for real: 120 baseline and 160 control trials in the same trial log. |
| REQ-8 | Done | `judge.py`'s pipeline scored 752/760 trials for real (8 excluded on judge refusal, ADR-0019); `judge_validated.flag` reflects an actual, specific human review, confirmed by reading it in this session. |
| REQ-9 | Done | `stats.py` refuses to run without `judge_validated.flag`; `fit_inference_model()`/`compare_classifiers()` ran against the real 474-trial detection subset. Result is a real null: 1/474 graded detected, identifiability coefficient CI `[-1.19, 2.20]`, decoder_norm AUC (0.76) exceeds identifiability AUC (0.63). |
| REQ-10 | Not Attempted | Dropped from this session's scope on instruction, matching `SPRINT-PLAN.md`'s own cut-order guidance. `layers.py` implements only the ADR-0009 fallback (`get_fallback_layer()`); `get_compression_boundary_layer()` does not exist. Every trial in `data/trials/trials.jsonl` is at the single fallback layer (layer 4); `configs/experiment.yaml`'s `injection.layer` is still `TODO`. |
| REQ-11 | Not Attempted | Dropped from this session's scope on instruction, matching `SPRINT-PLAN.md`'s risk register (REQ-11 is the first cut under time pressure). No Gemma Scope 2B loading path exists anywhere in `models.py`. |
| REQ-12 | Partial | This session implemented `src/prism/figures.py` and committed the two figures that have real data behind them (`identifiability_vs_detection.{svg,png}`, `strength_calibration.{svg,png}`) to `data/results/figures/`. The layer-comparison figure is correctly absent (REQ-10 not attempted). The report document itself (the 4-8 page PDF per `SPRINT-PLAN.md`'s deliverables checklist) has not been drafted in this codebase; that is a writing task, out of scope for this session. |
| REQ-13 | Not Attempted | The Limitations & Dual-Use appendix is report-writing, not code in this repo. Source material for it already exists here (ADR-0019's refusal-handling note, the calibration pilot's documented "brain damage" failure mode, ADR-0020's honest null-result framing), but the appendix text itself has not been written. |
| REQ-14 | Not Attempted | Abstract, author/affiliation block, and submission are writing and submission tasks outside this codebase. |
