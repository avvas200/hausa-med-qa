# Pre-Registration: Does Medical Reasoning in Small LLMs Survive Translation into Hausa?

**A paired benchmark, robustness audit, and preference-tuning pilot**

Author: Abbas Sani · Status: FROZEN. Frozen before any model was evaluated on the evaluation set.
Frozen version: the commit that set this status. Its hash and date are recorded in `README.md`.

---

## 1. Motivation

Medical LLM benchmarks are overwhelmingly English. Hausa is spoken by an estimated 80+ million people, many in regions with limited clinician access, yet it is rarely evaluated in medical NLP. If small, locally deployable models lose most of their medical competence when a question is asked in Hausa, then accuracy figures reported on English benchmarks overstate their usefulness for these communities.

This pilot asks a narrow, testable question: for the same medical questions, how much accuracy and answer stability do small open LLMs lose when the questions are presented in Hausa, where does that loss come from, and can a small preference-tuning intervention recover any of it?

## 2. Positioning against related work

> Note: the references below should be verified against the original papers before the report is written.

**AfriMed-QA.** A benchmark of medical questions drawn from African clinical and educational contexts. Its language coverage needs to be confirmed. If it contains little or no Hausa, this pilot is complementary: it covers the language axis, not the content axis.

**"Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA" (RMS-RSP).** This paper evaluates English medical QA (including AfriMed-QA) with MedGemma-4B-IT. It measures robustness as accuracy and semantic consistency under three answer-option reorderings. Its central finding is that gains appear more in invariance than in raw accuracy. This pilot adopts the same reordering protocol and metrics, and extends them to a language perturbation, English → Hausa. That makes the two sets of results directly comparable.

**Zhen Chen et al.**
- **IPDS / MAP.** A benchmark and multi-agent framework for inpatient pathways.
- **M3LLM / PMC-MIBench.** Converts an existing resource (compound figures from the biomedical literature) into training and benchmark data for medical multimodal reasoning.

This pilot follows the same methodology at small scale. It converts an existing resource (English medical MCQs) into a new paired benchmark, then evaluates a post-training method on it. The natural extensions are Hausa inpatient triage (following MAP) and multimodal Hausa medical QA (following M3LLM). Both are proposed as doctoral directions, not attempted here.

## 3. Research questions and hypotheses

All hypotheses, thresholds, and tests below are fixed before evaluation-set results are seen.

**H1 — Language gap (confirmatory).** For each model, accuracy on Hausa (HA) items is lower than on the paired English (EN) items.
- *Supported for a model if:* the EN − HA difference is at least 5 percentage points (pp), the exact McNemar test gives p < α after Holm correction across the three models (family α = 0.05), and the 95% paired-bootstrap CI excludes zero.

**H2 — Source of the gap (confirmatory).** Translate-test (Hausa question machine-translated back to English, HA→EN) recovers at least half of the EN − HA gap.
- *Supported for a model if:* (HA→EN − HA) / (EN − HA) ≥ 0.5, the HA→EN vs HA McNemar p < α after Holm correction, and the recovery ratio's bootstrap CI lower bound is > 0.
- *Interpretation:* strong recovery points to a language-understanding bottleneck. Weak recovery points to translation damage or knowledge loss that the pipeline cannot undo. H2 is tested only for models where H1 is supported.

**H3 — Robustness gap (confirmatory).** Semantic consistency under three option reorderings is lower in HA than in EN.
- *Semantic consistency* is the fraction of items for which the model selects the same underlying option in all three orderings. This definition follows RMS-RSP; confirm the exact definition against that paper before freezing.
- *Robust accuracy* is the fraction correct in all three orderings.
- *Supported for a model if:* the EN − HA consistency difference is at least 5 pp and its 95% paired-bootstrap CI excludes zero.

**H4 — Intervention (confirmatory, with a null result explicitly anticipated).** DPO on self-generated Hausa chain-of-thought preference pairs (correct vs. incorrect final answer) improves HA accuracy for Llama-3.2-3B-Instruct.
- *Decision rule (identical to the author's prior DPO study):* mean Δ ≥ +3 pp across 3 seeds, pooled McNemar p < 0.025, and per-seed CIs excluding zero.
- *Secondary outcome:* the change in HA semantic consistency. It is reported whatever the accuracy result, following RMS-RSP's finding that invariance can improve without accuracy gains.

**E1 — Exploratory, only if time permits.** Error analysis on 50 discordant items (correct in EN, wrong in HA). Each item is coded as one of: translation error, medical-term loss, option-format failure, or reasoning failure. No hypothesis test is attached.

## 4. Data

**Source.** MedMCQA validation split (license: Apache-2.0, per dataset card), sampled stratified by subject with a fixed seed (`seed=20260923`). The test split is not used because its labels are not public.

**Text repair.** MedMCQA contains a scraping artifact in which the letter pair "rt" was deleted from many words (e.g. "aery" for artery, "impoant" for important). Before filtering, question and option text is repaired with the fixed rule list in `src/data.py`. The number of replacements is recorded in the data manifest, and a sample of repaired questions is checked by hand.

**Exclusions.** The following items are removed before sampling, with counts recorded in the manifest:
- *Dental* subject items. Dental makes up about 30% of the filtered validation split and falls outside this study's general-health motivation. This scope decision was made after inspecting subject counts but before any model was run.
- Multi-answer items.
- Items with duplicate options.
- Items whose options depend on position or on other options ("All of the above", "Both A and B"), because these change meaning under reordering (H3).
- Items that refer to an image.

**Answer-position imbalance.** Gold answers in the canonical order are not uniformly distributed across A–D. An "always pick the most common letter" baseline is therefore reported alongside every accuracy figure. H1 and H2 are unaffected, since all their conditions share the same canonical order. If the AfriMed-QA MCQ subset's license permits it, a second source is added: up to 150 of its items, reported as a separate stratum.

**Evaluation set.** n = 500 items, 4 options each.

**DPO training pool.** 500 items from the MedMCQA *train* split. This pool is strictly disjoint from the evaluation set, and disjointness is checked by exact and near-duplicate matching on question text.

**Translation.**
- *Model and decoding:* NLLB-200 3.3B (bf16), beam search (4 beams, `no_repeat_ngram_size=4`), with `max_new_tokens = 1.6 × input length + 10`. Question stems and each option are translated separately.
- *Why these settings:* a first pass with the distilled 1.3B model and greedy decoding produced looping, repetitive output on about 10% of segments, mostly short option texts such as anatomical terms. The settings were changed in response. No model had been evaluated at that point.
- *Passthrough rule:* segments with no letters, or whose letters are all codes (ALLCAPS acronyms, mixed-case codes such as IgA, tokens of at most 2 letters), are copied verbatim.
- *Degeneration safeguard:* an output counts as degenerate if the same word appears 3 or more times in a row, if any single word makes up more than 40% of an output of 6+ words, or if the output is longer than 4 × the source length + 20 characters. A degenerate output is retried once with stricter decoding (5 beams, `no_repeat_ngram_size=2`, `repetition_penalty=1.3`). If the retry is also degenerate, the English source is kept (status `fallback_en`), mirroring common use of English loanwords for medical terms in Hausa.
- *Status recording:* every segment's status (`passthrough` / `translated` / `retried` / `fallback_en`) is recorded, and counts go into the manifest.
- *Back-translation (for HA→EN):* the same model and settings. Segments kept in English are copied back verbatim.

**Quality check and review of flagged segments.** The safeguards above catch looping output but not fluent mistranslation. Inspection of the first full run found fluent but wrong outputs on short options, for example "Bladder" rendered as an unrelated phrase, and drug names with invented additions. So:
- *Automatic flagging:* every Hausa segment is back-translated with a *different* model (NLLB-200 distilled 1.3B). Using a different model avoids selecting segments that the translate-test model happens to round-trip well, which would bias H2. A segment is flagged if its round-trip chrF1 against the source is below 60 (options) or 40 (questions), if an option's back-translation is more than 2 × the source word count + 1, or if the Hausa adds a sentence.
- *Review of evaluation items:* the author, a native Hausa speaker, reviews each flagged evaluation segment and records one decision: `keep`, `english` (use the English term, as is common for medical vocabulary in Hausa), or `edit` (write a corrected Hausa version). This happens before any model is run, without access to model outputs, and without changing which option is correct. A blank decision defaults to `english` for options and `keep` for questions, and defaults are counted. The completed sheet is committed to the repository.
- *Training pool:* flagged segments fall back to English automatically, with no human review. Training-pool items whose *question stem* fell back to English are excluded from DPO pair generation (section 6), because they are not Hausa questions.
- *Role of the thresholds:* they only determine the review workload. They may be adjusted once, before the review sheet is exported, and the final values are recorded in the manifest.
- *Threshold adjustment (made before export, before any model was run):* the first QE run showed that round-trip chrF misses fluent errors, including calques that both NLLB models reverse the same way (e.g. "surface tension" rendered with the Hausa word for emotional tension). A check of 20 random evaluation options scoring ≥ 80 found 2 errors, while samples in the 60–80 range contained several. The option threshold for the *evaluation review* was therefore raised from 60 to 80 (about 1,200 of 1,729 translated options flagged). The *training pool* keeps the original threshold of 60, so that its options stay mostly in Hausa; unreviewed errors in the pool are a limitation for H4.

- *Outcome of the review (recorded before freezing):* 1,276 evaluation segments were flagged (1,215 options, 61 question stems). Decisions: 1,016 `keep`, 232 `english`, 28 `edit`; no decision was left blank. Seven question stems first marked `english` were changed to `keep` or `edit`, because question stems stay in Hausa by design. After review, 232 of 2,000 evaluation options (11.6%) are in English, and 95 of 500 items have at least one English option; in addition, 271 segments are passthrough codes, numbers or acronyms. In the training pool, 937 options and 70 question stems fell back to English automatically.

The evaluation set is therefore *Hausa question stems with Hausa or English answer options*. This reflects realistic Hausa–English code-switching in medical contexts, and the proportion of English options is reported.
- *Prompt template:* translated once, then hand-corrected by the author.

**Human validation.**
- *Sample:* 75 randomly selected items (15%), rated on the *final* Hausa text (after review) independently by the author and at least one other native Hausa speaker who did not take part in the review. This random sample estimates the residual error rate, including errors the automatic flags missed.
- *Adequacy:* rated 1–5, where 1 means meaning lost and 5 means meaning fully preserved.
- *Medical-term flag:* whether a medical term was mistranslated. An English loanword kept in Hausa is not counted as an error.
- *Reporting:* mean adequacy, the proportion of items rated ≥ 4, and inter-rater agreement (quadratic-weighted Cohen's κ). Because ratings turned out to be concentrated at 4–5, where κ is unstable, the report also gives the full rater-by-rater table, exact agreement, agreement within one point, and agreement on "acceptable" (≥ 4). κ remains the pre-registered agreement statistic and is reported as computed.
- *Rater flag (defined before freezing, used in sensitivity analysis a):* an item is flagged if either rater gave adequacy ≤ 3 or marked a medical-term error.
- *Results (obtained before freezing; no model had been run):*
  - Mean adequacy 4.52 (author) and 4.36 (second rater); 100% and 98.7% of items rated ≥ 4.
  - Medical-term errors marked: 0 by both raters.
  - Quadratic-weighted κ = −0.30; exact agreement 34.7%; agreement within one point 98.7%; agreement on ≥ 4: 98.7%. The raters agree that almost all translations are acceptable but not on which are a 4 and which a 5 (the second rater is stricter). κ is therefore not informative here.
  - Items flagged by either rater: 1 of 75. 61 of the 75 rated items also contained segments the author had reviewed.
  - *Coverage limitation:* both raters reported checking mainly the question stems, and the answer options less closely. The validation therefore estimates stem quality well but does not give a reliable residual error rate for options. The only estimate for unreviewed options (chrF1 ≥ 80) is the author's pre-export check, which found 2 errors in 20 (about 10%, with a wide interval).

**Sensitivity analysis.** H1–H3 are repeated twice: (a) excluding items flagged by any rater (1 item; this analysis is therefore expected to match the primary one closely), and (b) excluding items with any option kept in English (`fallback_en` or `fallback_en_reviewed`). The primary analysis uses all 500 items. Because the full set is machine-translated, part of the measured gap may reflect translation error rather than model failure. The validation subset bounds how large that share is likely to be.

## 5. Models and inference

| Model | Role | Precision on T4 |
|---|---|---|
| Llama-3.2-3B-Instruct | Evaluation + H4 intervention | fp16 / 4-bit |
| Qwen2.5-3B-Instruct | Evaluation | fp16 / 4-bit |
| MedGemma-4B-IT | Evaluation (medical-domain model; matches RMS-RSP) | 4-bit |

If MedGemma access is unavailable, it is replaced by Gemma-3-4B-IT, and the substitution is recorded.

**Prompting.**
- Zero-shot, with one fixed template per language, and the instruction language matching the question language.
- For HA→EN, the English template is used.

**Decoding and scoring.**
- Greedy decoding, `max_new_tokens=8`, answer-letter output.
- The letter is parsed with a fixed regex. An unparseable output is scored as incorrect.
- Parse rates are reported per condition, since formatting failure is itself a finding in low-resource settings.

**Conditions per model.**
- EN, HA, and HA→EN in the canonical option order.
- Two further fixed option permutations for EN and HA, used for H3.
- That makes 7 passes × 500 items = 3,500 generations per model.

## 6. Intervention protocol (H4)

1. **Sample reasoning.** For each training-pool item whose question stem is in Hausa (430 of 500; the 70 items whose stem fell back to English are excluded), sample k = 4 chain-of-thought completions (temperature 0.8) from the base model.
2. **Build preference pairs.** Keep items that have at least one correct and at least one incorrect completion. From each, form one (chosen = correct, rejected = incorrect) pair. The final pair count is reported. If fewer than 200 pairs result, k is raised to 8 once, and this contingency is pre-registered here.
3. **Train.** DPO with LoRA, using the same configuration and hyperparameters as the author's prior study (`dpo-reasoning-transfer`), with 3 seeds.
4. **Evaluate.** HA accuracy and HA semantic consistency on the evaluation set, plus EN accuracy to check for regression.

## 7. Statistical analysis

- **Paired tests.** Exact McNemar for every accuracy comparison, since the same items appear in each condition.
- **Confidence intervals.** 95% paired bootstrap with 10,000 resamples over items.
- **Multiple comparisons.** Holm correction across the three models within each of H1 and H2. For H4, the prior decision rule is used as specified.
- **Power.** With n = 500 and about 20% discordant pairs, a 5 pp gap is detected with roughly 70% power. Gaps of 10 pp or more are detected with high power. Published cross-lingual results suggest low-resource gaps are often large, but the pilot is underpowered for small effects, and this is stated as a limitation.
- **Code.** All tests are implemented in `src/stats.py`, reused from the prior study, and committed before results are generated.

## 8. Compute budget (Colab free-tier T4)

A 20-item timing run was done in notebook 01; the figures below are planning estimates.

| Stage | Rough estimate |
|---|---|
| Translation (500 eval + 500 train + back-translation; measured ~17 min for 1.3B greedy on A100, expect several times longer for 3.3B with beam search) | ~1–1.5 h |
| Baseline evaluation (3 models × 3,500 short generations) | 3–5 h |
| DPO pair generation (500 × k = 4 CoT) | 3–4 h |
| DPO training + evaluation (3 seeds) | 4–6 h |
| **Total** | **~11–17 h** |

All stages are written to be resumable from cached artefacts, as in the prior study. Runs use Colab Pro (A100 40 GB where available).

## 9. Timeline

The HKPFS deadline is typically around 1 December; confirm the exact date on the RGC and PolyU sites.

| Week | Dates (2026) | Work |
|---|---|---|
| 1 | 28 Sep – 4 Oct | License checks, sampling, translation, timing run |
| 2 | 5 – 11 Oct | Human validation; **freeze and commit this pre-registration** |
| 3 | 12 – 18 Oct | Baseline evaluation (H1–H3) |
| 4 | 19 – 25 Oct | DPO pair generation and training (H4) |
| 5 | 26 Oct – 1 Nov | Statistical analysis, error analysis (E1), figures |
| 6 | 2 – 8 Nov | Report, README, outreach email to Prof. Chen |
| Buffer | 9 – 22 Nov | Slippage; HKPFS application materials |

## 10. Threats to validity

- **Translation confound.** Machine-translation errors inflate the apparent gap. This is mitigated by the human-validated subset, the flagged-item sensitivity analysis, and the E1 error coding.
- **Author review.** The author reviews flagged segments and is also one of the raters, so author-rated quality may be optimistic. Mitigations: a second rater who did not take part in the review, reporting inter-rater agreement, and reporting how many rated items were also reviewed.
- **Unvalidated option quality.** Human validation covered mainly question stems (section 4). Unreviewed options with high round-trip scores have an estimated error rate of about 10% from a 20-item check. Option errors would inflate the measured gap; E1 error coding records how often a discordant item traces back to an option error.
- **English fallback.** Segments kept in English make Hausa items partly English. This biases the measured gap toward zero, a conservative direction for H1, and sensitivity analysis (b) quantifies the effect.
- **Back-translation artefacts.** Round-tripping can restore English phrasing that the model memorised, which would overstate H2 recovery. This is noted as a caveat in interpreting H2.
- **Contamination.** English MedMCQA items may appear in pretraining data, which inflates EN accuracy relative to HA. The EN − HA gap therefore mixes language effects with memorisation effects. This is acknowledged explicitly and not resolved in the pilot.
- **Scale.** The pilot uses 3–4B models, n = 500, and one intervention model. Results should not be generalised to larger models or other languages.

## 11. Ethics and scope

- **Data.** No patient data is used. All items come from public educational datasets.
- **Scope of claims.** This is a capability evaluation, not a clinical validation. No claim is made that any model is safe for medical use in Hausa or any other language, and the report states this prominently.
- **Human raters.** Raters are volunteers, credited in the report with their consent.

## 12. Deliverables

- Public GitHub repository with code, cached artefacts, and this frozen pre-registration.
- The paired EN/HA evaluation set, released only if the source license permits redistribution; otherwise, release the scripts that regenerate it.
- A technical report of about 6–8 pages, with all hypotheses reported as supported, not supported, or inconclusive under the rules above.

## 13. Deviations after freezing
 
**Deviation 1: answer-turn prefill (decided before any evaluation item was run).**
- *What changed:* the model's answer turn now begins with the cue word `Answer:` (English template) or `Amsa:` (Hausa template), so the first generated tokens are the answer letter. The templates, greedy decoding, `max_new_tokens=8`, the parser, and the rule that unparseable outputs count as incorrect are unchanged. The change applies to all models, languages and conditions.
- *Why:* the pre-evaluation smoke test on 10 **training-pool** items showed that, with the Hausa template, Llama-3.2-3B parsed 7/10 and Qwen2.5-3B parsed 1/10: the models repeated the Hausa instruction instead of answering. English parsed 10/10 for all models, and MedGemma parsed 10/10 in both languages. Without the change, the Hausa accuracy of two models would mostly measure instruction echoing, not medical knowledge.
- *What is kept:* the original set-up without prefill is still run on the evaluation set in the canonical order for English and Hausa (`en_p0_noprefill`, `ha_p0_noprefill`). These runs are exploratory, are not used for H1–H3, and are reported with their parse rates, since format-following failure is itself a finding.

**Deviation 2: secondary letter-probability scoring (decided before any evaluation item was run).**
- *What was added:* from the same generation pass, the log-probability of each answer letter (A–D) as the first generated token is recorded, and the highest-scoring letter is taken as a second answer. Accuracy, semantic consistency, robust accuracy and the H1–H3 statistics are also computed on these answers and reported as a **secondary, non-confirmatory** analysis. The confirmatory H1–H3 decisions still use the generated and parsed answers.
- *Why:* with the prefill of deviation 1, the repeated smoke test (10 training-pool items) parsed 10/10 in both languages for Llama-3.2-3B and MedGemma-4B, but only 3/10 in Hausa for Qwen2.5-3B, which continued with Hausa or Swahili text instead of a letter. A model-specific prompt fix was rejected, because tuning prompts per model until they parse would make the conditions non-comparable. Letter-probability scoring separates two causes of Hausa failure, not following the answer format and not knowing the answer, without changing the primary measure.
- *Stopping rule:* no further changes are made to prompts, decoding or parsing. Qwen2.5-3B's Hausa parse failures count as incorrect in the confirmatory analysis, as pre-registered, and its parse rate is reported next to its accuracy.

**Deviation 3: hardware and precision (recorded after the evaluation run; not a choice made in response to results).**
- *What differs:* the section 5 table plans fp16 or 4-bit on a T4, and 4-bit for MedGemma-4B. All three models were instead run in bfloat16 without quantisation, on a Tesla T4 (bf16 is emulated in software on this GPU), as recorded in `results/eval_models_resolved.json` and `results/eval_run_log.json`. Prompts, decoding, parsing and scoring are unchanged.
- *Why it happened:* the loading code selects bf16 whenever PyTorch reports bf16 support, and current PyTorch reports it on a T4 through emulation, so the planned fp16/4-bit set-up was never used.
- *Consequence:* higher numerical precision than planned. It is not expected to change any conclusion. For H4, the DPO-tuned models are evaluated with the same bf16 set-up, so base and tuned models remain comparable.