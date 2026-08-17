# REQ status report

QA pass before this codebase is used to write the research report (`.claude/prompts.md` Prompt 12). Everything below was checked directly in this session, not assumed from an earlier ADR or a script that merely exited without an error.

## QA findings

**Test suite.** `pytest -m "not integration" -q` from the repo root: 281 passed, 4 skipped, 14 deselected, 0 failed. The skips are explicit `# N/A: <reason>` stubs or environment-gated cases, not silently disabled coverage. The REQ-11 additions cover model-scoped judge scoring, default preservation of refusal exclusions, explicit refusal retry, and per-model analysis from the shared ledger.

*Update from a later REQ-11 session:* the `0.9808` figure above reflected a bug ADR-0022 later found and fixed: the metric didn't exclude the BOS token, whose activation norm is a documented outlier at this hook point. Re-measured under the fix, Pythia's real reconstruction quality is `0.9782` (112 tokens, not 120). The `req1_sae_validation.json` committed alongside this note carries the current, correct number.

**Trial-record completeness.** The shared ledger now has 970 records: Pythia has 760 (752 scored, 8 explicitly excluded refusals, 0 pending) and Gemma has 210 (210 scored, 0 refusals, 0 pending). The 8 Pythia exclusions retain their original `judge refused to grade trial ... (category='bio')` reasons, matching the preserved Pythia provenance record. The Gemma pass is separately recorded in `judge_scoring_provenance_gemma.json` and cannot revisit Pythia rows without an explicit refusal-retry flag.

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

All nine distinct commit hashes above were checked directly against this repository's object store (`git cat-file -t <hash>`) and confirmed to exist as real commits, not invented placeholders.

## REQ-1 through REQ-14

| REQ | Status | Note |
|---|---|---|
| REQ-1 | Done | SAE dependency resolved to `ghidav/pythia-70m-deduped-sae` (ADR-0010); `load_model_and_sae()` implemented; reconstruction validated at `fraction_variance_explained = 0.9782` against real tokens (re-measured under ADR-0022's BOS-exclusion fix; was `0.9808` before that fix). |
| REQ-2 | Done | `features.py` implements the audit loader, stratified sampler, and balance check; `data/audit/features.csv` (16,384 features) and `data/results/sampled_features.csv` (40 sampled, tertile-labeled) exist with real provenance. |
| REQ-3 | Done | `inject.py` implements the persistent injection hook; zero-vector no-op and hook-cleanup are both tested explicitly. |
| REQ-4 | Done | `prompts.py` implements the three templates; the control-question set is externally editable in `configs/control_questions.yaml`. |
| REQ-5 | Done | Real pilot (`data/results/calibration_pilot.jsonl`, 40 trials) and `calibration_notes.md` document the too-weak/usable/too-strong transition; `configs/experiment.yaml`'s `injection.strengths` reflects it. |
| REQ-6 | Done | `runner.py`'s `run_systematic_trials()` ran for real; systematic trials are present in `data/trials/trials.jsonl` with a resumable, deduplicated writer. |
| REQ-7 | Done | `run_baseline_trials()`/`run_control_trials()` ran for real: 120 baseline and 160 control trials in the same trial log. |
| REQ-8 | Done | `judge.py`'s pipeline scored 752/760 trials for real (8 excluded on judge refusal, ADR-0019); `judge_validated.flag` reflects an actual, specific human review, confirmed by reading it in this session. |
| REQ-9 | Done | `stats.py` refuses to run without `judge_validated.flag`; `fit_inference_model()`/`compare_classifiers()` ran against the real 474-trial detection subset. Result is a real null: 1/474 graded detected, identifiability coefficient CI `[-1.19, 2.20]`, decoder_norm AUC (0.76) exceeds identifiability AUC (0.63). |
| REQ-10 | Not Attempted | Blocked on missing external data, not a time-pressure cut: the UCARE intrinsic-dimension trajectory ADR-0009 names as the intended input does not exist anywhere in this repo or `sae-bounding` (ADR-0021). `layers.py` implements only the ADR-0009 fallback (`get_fallback_layer()`); `get_compression_boundary_layer()` is unimplemented (ADR-0021), matching `layers.py`'s own docstring. Every trial in `data/trials/trials.jsonl` is at the single fallback layer (layer 4); `configs/experiment.yaml`'s `injection.layer` is still `TODO`. |
| REQ-11 | Done | Gemma Scope's 210 real trials were re-scored under model-scoped judge isolation: 210 scored, 0 refused, and Pythia remains 752 scored / 8 excluded. Gemma has 9/120 judge-detected systematic trials; its identifiability coefficient is 0.29 (95% CI [-0.34, 0.92]) and its identifiability AUC is 0.60. The pooled 594-trial fit is also null (coefficient -0.59, 95% CI [-1.39, 0.21]) but is descriptive only because model designs differ. `--score-model-name`, `--retry-refusals`, separate scoring provenance, and per-model stats prevent the cross-model bookkeeping bugs found before analysis. |
| REQ-12 | Done | The primary Pythia figure and calibration figure have real data behind them, and the per-model detection-rate figure is explicitly descriptive because the intervention regimes differ. The layer-comparison figure remains correctly absent because REQ-10 is blocked. |
| REQ-13 | Not Attempted | Limitations and dual-use documentation is a report-writing task and is not committed to this codebase. |
| REQ-14 | Not Attempted | Abstract, author/affiliation block, and external submission are report-writing or submission tasks outside this codebase. |
