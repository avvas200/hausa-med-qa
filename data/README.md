# Released data: provenance, licence, attribution

This folder is otherwise gitignored. The files listed below are released deliberately,
because `PREREGISTRATION.md` § 4 commits to publishing the review sheet and because the
coding sheets are the evidence behind the E1 figures. Everything else here — caches,
intermediate JSONL, model outputs — stays local.

## What is released

| file | rows | what it is |
|---|---|---|
| `adjudication_eval_done.csv` | 1276 | the § 4 review sheet: every flagged evaluation segment with the author's `keep` / `english` / `edit` decision and any corrected Hausa |
| `validation_rater1_done.csv` | 75 | human validation, rater 1 (the author): adequacy 1–5, medical-term error flag, notes |
| `validation_rater2_done.csv` | 75 | human validation, rater 2 (independent of the review) |
| `e1_coding_coder1.csv` | 50 | E1 blank coding sheet as exported |
| `e1_coding_coder2.csv` | 50 | the same blank sheet, retained unfilled — see below |
| `e1_coding_coder1_done.csv` | 50 | E1 coding pass, completed |
| `e1_coding_coder2_done.csv` | 50 | E1 verification pass over the same 50 rows |
| `e1_codebook.md` | — | the written E1 category definitions, generated from `src/e1.CODEBOOK` |

`e1_coding_coder2.csv` is kept blank and released so that an independent second coder can
still be recruited without re-exporting anything; `e1_coding_coder2_done.csv` is a
verification pass, not independent coding (see `PREREGISTRATION.md` § 13, deviation 5).

## Source and licence

The English question stems and answer options in these sheets are derived from
**MedMCQA** (validation split), via the Hugging Face distribution
[`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa).
Text was repaired for a known scraping artifact before use (`src/data.py`), so it is not
byte-identical to the source.

Two licences are attached to MedMCQA by its own publishers, and they disagree:

- The source project, [github.com/MedMCQA/MedMCQA](https://github.com/MedMCQA/MedMCQA),
  is released under the **MIT License**.
- The Hugging Face dataset card carries `license: apache-2.0` in its YAML metadata, and
  the loading script `medmcqa.py` has an Apache-2.0 header. The card's own "Licensing
  Information" section is an unfilled placeholder.

`results/data_manifest.json` records `apache-2.0` because that is what the dataset card
reported when the data was prepared on 2026-09-23; that field is a provenance record and
is left as it was. Both licences are permissive and both permit redistribution of derived
text with attribution and a copy of the licence notice, so the release is clean under
either reading. This note is the attribution.

The Hausa text is machine translation produced by this study with NLLB-200-3.3B
(CC-BY-NC-4.0), post-edited in places by the author. **Non-commercial use only**, which
is the binding constraint on these files — it is stricter than either MedMCQA licence.

Model outputs (`model_raw_output` in the E1 sheets) come from Llama-3.2-3B-Instruct and
MedGemma-4B-IT and are subject to those models' own licences.

## Citation

If you use these files, cite MedMCQA as its authors ask:

```bibtex
@inproceedings{pal2022medmcqa,
  title     = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for
               Medical domain Question Answering},
  author    = {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {174},
  pages     = {248--260},
  year      = {2022},
  publisher = {PMLR}
}
```
