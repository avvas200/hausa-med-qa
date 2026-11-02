# Hausa Medical QA: does medical reasoning in small LLMs survive translation into Hausa?

A pre-registered pilot measuring how much accuracy and answer stability small open LLMs lose when medical multiple-choice questions are asked in Hausa rather than English, where that loss comes from, and whether a small DPO intervention recovers any of it.

The hypotheses (H1–H4), decision rules, and analysis plan are fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) before any model is evaluated.

**Frozen at commit `66d3f1d` ("Freeze pre-registration"), 2026-09-24 16:37 +0100.** That commit sets the `FROZEN` status in `PREREGISTRATION.md`; the hash was written back into this README by the next commit, `9fac87b`, because a commit cannot contain its own hash. An earlier version of this line cited `6bb50a1` in error — that commit is "changes for translate.py file", made five hours before the freeze.

Supporting provenance, all checkable with `git log`:

- [`src/stats.py`](src/stats.py), which contains every decision rule and statistical test, is byte-identical to its first commit (`68b1640`, 2026-09-23) and has not been modified since — before any translation or model run.
- Baseline evaluation started 2026-09-25 15:21 +0100 (`results/eval_run_log.json`) with [`src/evaluate.py`](src/evaluate.py) already committed at `cd35807` (2026-09-24 18:01).
- Four deviations after freezing are logged in `PREREGISTRATION.md` § 13, each with its date and the reason it was made.


## Status

- [x] 01 · Data preparation (sampling, repair, translation)
- [x] 01b · Translation quality check, review of flagged segments, rater sheets
- [x] Human validation of translations (75 items, 2 raters)
- [x] Pre-registration frozen
- [x] 02 · Baseline evaluation (H1–H3)
- [x] 02b · E1 error analysis (exploratory) — coded; coding under revision, see below
- [ ] 03 · DPO intervention (H4)
- [ ] 04 · Analysis and figures
- [ ] Report

## Results so far (H1–H3)

Three models, n = 500 paired items, zero-shot, greedy decoding, answer-letter parsing, unparseable scored as incorrect. Full numbers in [`results/baseline_results.json`](results/baseline_results.json); run provenance in [`results/eval_run_log.json`](results/eval_run_log.json).

**Accuracy in the canonical option order (`p0`).** The pre-registered "always pick the most common letter" baseline for this order is **0.318**.

| Model | EN | HA | HA→EN | EN − HA | HA vs. letter baseline |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 0.618 | 0.276 | 0.450 | +34.2 pp | −4.2 pp |
| Qwen2.5-3B-Instruct | 0.526 | 0.122 | 0.392 | +40.4 pp | −19.6 pp |
| MedGemma-4B-IT | 0.550 | 0.356 | 0.402 | +19.4 pp | +3.8 pp |

Asked in Hausa, none of the three models is meaningfully better than always answering the most common letter, and two are worse than it. Qwen's figure is depressed by format failure: only 47% of its Hausa outputs could be parsed, and unparseable counts as incorrect, as pre-registered. Under robust accuracy (correct in all three option orderings) the Hausa figures are 0.088, 0.034 and 0.188.

- **H1 — language gap: supported for all three models.** Holm-adjusted exact McNemar p < 1e−12 throughout; every paired-bootstrap CI excludes zero.
- **H2 — translate-test recovers ≥ half the gap: supported for Llama (0.509) and Qwen (0.668), not supported for MedGemma** (0.237, p = 0.073).
- **H3 — robustness gap: supported for all three.** Semantic consistency EN → HA: Llama 0.670 → 0.210, Qwen 0.676 → 0.084, MedGemma 0.578 → 0.408.

**Where these conclusions are fragile.** Both of the following are pre-registered analyses that disagree with the primary one, and are recorded here rather than left to be found:

- Llama's H2 clears the threshold by 0.009 (recovery 0.509 against a 0.50 rule) and **flips to not-supported** under sensitivity analysis (b), which drops items with any option kept in English: 0.436.
- Qwen's H2 **flips to not-supported** under the secondary letter-probability scoring: 0.489.

**Gap averaged over all three option orderings** rather than the canonical order alone. H1 is pre-registered on `p0`, and `p0` happens to be the worst Hausa ordering for Llama (0.276, against 0.342 and 0.312), so the single-order gap overstates that model's loss:

| Model | EN mean | HA mean | gap |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 0.608 | 0.310 | 29.8 pp |
| Qwen2.5-3B-Instruct | 0.521 | 0.126 | 39.5 pp |
| MedGemma-4B-IT | 0.554 | 0.373 | 18.1 pp |

### E1 error analysis — provisional, do not cite

The coded sheet ([`results/e1_coding_done.csv`](results/e1_coding_done.csv), 50 rows) and its summary are committed for transparency, but **the category proportions are being revised and should not be cited yet.** Two problems found on review: 10 of the 29 rows coded as a translation error or medical-term loss point at a segment whose Hausa text is byte-identical to the English (`Placenta`, `CMV`, `Warfarin`, `Atopy`, `AFP`, ...), where no translation error is possible; and the `notes` column is empty on all 50 rows, so no judgement can be audited. E1 is exploratory with no hypothesis attached (`PREREGISTRATION.md` § 3 and deviation 4).

## Layout

```
PREREGISTRATION.md          hypotheses, decision rules, analysis plan
src/config.py               all fixed settings (seed, sizes, filters, models)
src/data.py                 filtering, stratified sampling, disjointness, records
src/translate.py            resumable NLLB-200 translation with passthrough/retry/fallback
src/qe.py                   round-trip quality check and review-sheet handling
src/evaluate.py             prompts, generation, answer parsing, letter-probability scoring
src/stats.py                McNemar, paired bootstrap, Holm, H1–H4 decision rules
src/e1.py                   E1 sampling, coding sheet, validation, summary
notebooks/01_data_preparation.ipynb
notebooks/01b_translation_qe.ipynb
notebooks/02_baseline_evaluation.ipynb
notebooks/02b_error_analysis.ipynb
results/                    small committed outputs (manifests, metrics, figures)
```

Large artefacts (datasets, translations, model outputs) live on Google Drive under `hausa-med-qa/` and are not committed. `results/data_manifest.json` records their SHA-256 hashes.

## Running

Open each notebook in Colab with a GPU runtime and run top to bottom. Every stage caches to Drive and resumes after a disconnect.

## Data and scope

Questions come from MedMCQA; check its license before redistributing any derived data. This is a capability evaluation, not a clinical validation; no model here should be used for medical advice.
