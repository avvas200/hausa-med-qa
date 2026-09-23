# Hausa Medical QA: does medical reasoning in small LLMs survive translation into Hausa?

A pre-registered pilot measuring how much accuracy and answer stability small open LLMs lose when medical multiple-choice questions are asked in Hausa rather than English, where that loss comes from, and whether a small DPO intervention recovers any of it.

The hypotheses (H1–H4), decision rules, and analysis plan are fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) before any model is evaluated. Frozen at commit: `________`.

## Status

- [x] 01 · Data preparation (sampling, repair, translation)
- [ ] 01b · Translation quality check, review of flagged segments, rater sheets
- [ ] Human validation of translations (75 items, 2 raters)
- [ ] Pre-registration frozen
- [ ] 02 · Baseline evaluation (H1–H3)
- [ ] 03 · DPO intervention (H4)
- [ ] 04 · Analysis and figures
- [ ] Report

## Layout

```
PREREGISTRATION.md          hypotheses, decision rules, analysis plan
src/config.py               all fixed settings (seed, sizes, filters, models)
src/data.py                 filtering, stratified sampling, disjointness, records
src/translate.py            resumable NLLB-200 translation with passthrough/retry/fallback
src/qe.py                   round-trip quality check and review-sheet handling
src/stats.py                McNemar, paired bootstrap, Holm, H1–H4 decision rules
notebooks/01_data_preparation.ipynb
notebooks/01b_translation_qe.ipynb
results/                    small committed outputs (manifests, metrics, figures)
```

Large artefacts (datasets, translations, model outputs) live on Google Drive under `hausa-med-qa/` and are not committed. `results/data_manifest.json` records their SHA-256 hashes.

## Running

Open each notebook in Colab with a GPU runtime and run top to bottom. Every stage caches to Drive and resumes after a disconnect.

## Data and scope

Questions come from MedMCQA; check its license before redistributing any derived data. This is a capability evaluation, not a clinical validation; no model here should be used for medical advice.
