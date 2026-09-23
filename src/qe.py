"""Round-trip quality estimation (QE) and human adjudication of flagged segments.

QE back-translates HA -> EN with a *different* model (C.QE_MODEL) from the one
used for the translate-test condition, so that filtering on QE does not select
segments the translate-test model happens to round-trip well (which would bias H2).

Flags only decide what a human reviews. Unflagged errors are estimated by the
independent random rater sample.
"""
import re

import pandas as pd

from . import config as C

_sent_break = re.compile(r"[.!?]\s+\S")


def chrf1(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF(beta=1, lowercase=True).sentence_score(hyp, [ref]).score


def flag_reasons(src: str, ha: str, back: str, field: str):
    """Return (chrF1 of round trip vs source, list of reasons)."""
    score = chrf1(back, src)
    reasons = []
    thr = C.QE_MIN_CHRF_OPTION if field == "option" else C.QE_MIN_CHRF_QUESTION
    if score < thr:
        reasons.append("low_roundtrip")
    if field == "option" and len(back.split()) > 2 * len(src.split()) + 1:
        reasons.append("expansion")
    if len(_sent_break.findall(ha)) > len(_sent_break.findall(src)):
        reasons.append("added_sentence")
    return score, reasons


DECISIONS_OPTION = {"keep", "english", "edit"}
DECISIONS_QUESTION = {"keep", "edit"}


def adjudication_sheet(flag_df: pd.DataFrame, eval_by_id: dict) -> pd.DataFrame:
    """flag_df columns: key, field, en, ha, back_qe, chrf1, reasons.
    Adds context (question stem, gold option) and empty decision columns."""
    rows = []
    for r in flag_df.itertuples(index=False):
        item_id = r.key.split("|")[0]
        rec = eval_by_id[item_id]
        is_gold = r.field == "option" and int(r.key.split("|o")[1]) == rec["answer_idx"]
        rows.append({
            "key": r.key, "field": r.field, "question_en_context": rec["question"],
            "is_gold_option": is_gold, "en": r.en, "ha": r.ha, "back_translation": r.back_qe,
            "chrf1": round(r.chrf1, 1), "reasons": ",".join(r.reasons),
            "suggested": "english" if r.field == "option" else "keep",
            "decision": "", "edited_ha": "",
        })
    return pd.DataFrame(rows)


def read_decisions(path) -> pd.DataFrame:
    """Validate a completed sheet. A blank decision takes the suggested value
    and is logged as a default."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["decision"] = df["decision"].str.strip().str.lower()
    df["defaulted"] = df["decision"] == ""
    df.loc[df.defaulted, "decision"] = df.loc[df.defaulted, "suggested"]
    errors = []
    for r in df.itertuples(index=False):
        allowed = DECISIONS_OPTION if r.field == "option" else DECISIONS_QUESTION
        if r.decision not in allowed:
            errors.append(f"{r.key}: decision '{r.decision}' not in {sorted(allowed)}")
        if r.decision == "edit" and not r.edited_ha.strip():
            errors.append(f"{r.key}: decision 'edit' but edited_ha is empty")
    if errors:
        raise ValueError("Fix these rows in the sheet:\n" + "\n".join(errors[:30]))
    return df
