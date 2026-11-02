# E1 coding guide

Generated from `CODEBOOK` in `src/e1.py` -- do not edit by hand.
Fixed by deviation 5 in `PREREGISTRATION.md` before any row was coded.

Fill three columns on every row: **code**, **segment**, **notes**. Leave everything
else alone. Work down the sheet; do not skip rows.

## The four categories

Check them **in this order** and use the first that applies.

### 1. `format`

Option-format failure. The Hausa rendering broke an option's surface form so the choice set stopped functioning: numbers, units, ranges or doses mangled; two options collapsed to the same string; an option truncated or merged into another. This does NOT mean the model failed to emit a letter -- that is recorded automatically in the 'unparseable' column, not coded by hand.

### 2. `term`

Medical-term loss. For the coded segment the Hausa differs from the English AND a medical term's referent is lost or changed: wrong drug, wrong anatomical structure, wrong condition, wrong procedure. A term carried over unchanged in English or Latin is NOT term loss -- the model is shown the same string in both conditions, so that segment cannot be the source of the difference.

### 3. `translation`

Translation error. For the coded segment the Hausa differs from the English AND the non-terminological meaning is wrong: negation dropped or added, a quantifier or comparative flipped, wrong clause attachment, or syntax that changes what is being asked.

### 4. `reasoning`

Reasoning failure. Residual category. The note must affirm that the question stem, the gold option and the option the model chose are all adequate Hausa, so the item was answerable as presented and the model simply got it wrong.

## Rules the sheet will enforce

A sheet that breaks any of these is rejected with the offending rows named,
so it is worth checking as you go.

| | Rule |
|---|---|
| R1 | `code` is one of `format`, `term`, `translation`, `reasoning` |
| R2 | `notes` is filled on **every** row, at least 10 characters |
| R3 | `term` and `translation` need a `segment`; `reasoning` must leave it empty; `format` may use one |
| R4 | A coded `segment` must differ between English and Hausa. The `identical_options` column lists the options where it does not |
| R5 | A `term` or `translation` error must sit in `stem`, the gold option, or the option the model chose. The `codeable_segments` column lists exactly which |

## Columns that help you

| Column | Meaning |
|---|---|
| `gold` | The correct answer |
| `model_answer_ha` | What the model answered in Hausa (always wrong here, by construction) |
| `identical_options` | Options whose Hausa is identical to the English. **R4 rejects these** |
| `codeable_segments` | The only segments R5 allows for `term`/`translation` |
| `unparseable` | Set automatically when no letter could be read. Do not code `format` for this |
| `*_status` | How the Hausa was produced: `auto`, `reviewed_kept`, `reviewed_edited`, `english` (kept in English), `code_or_number` (passed through) |

## Writing the note

Say what is wrong and how it could have changed the answer. For `reasoning`,
confirm the stem, the gold option and the chosen option are all adequate Hausa.

- Good: "Hausa renders 'bladder' as the word for 'stomach', so the gold option names the wrong organ."
- Bad: "wrong translation"
