"""E1 exploratory error analysis (PREREGISTRATION.md section 3; sample fixed in
deviation 4, re-coded under deviation 5). No hypothesis test is attached.

Discordant item = correct in en_p0 and wrong in ha_p0 (primary, parsed scoring).
25 items are drawn per model (C.E1_MODELS, in that order, one seeded RNG). An
item may be drawn for both models; each (model, item) row is coded separately.

Only draw_sample() reads the cached model outputs. Everything after it works
from the coding sheet alone -- the sheet carries the paired EN/HA text, the
segment statuses, the gold letter and the model's answer -- so exporting,
validation, agreement and the summary all run without Drive or a GPU.
"""
import difflib
import math
import random
import re

import pandas as pd

from . import config as C
from . import data as D

# Pre-registered categories, checked in this order (first that applies wins).
CODES = ["format", "term", "translation", "reasoning"]
NEEDS_SEGMENT = {"term", "translation"}
NO_SEGMENT = {"reasoning"}
SEGMENTS = ["stem"] + C.LETTERS
MIN_NOTE_CHARS = 10

# Written definitions, fixed by deviation 5 before re-coding. The first pass
# left these implicit, which is what made its proportions uninterpretable.
CODEBOOK = {
    "format": (
        "Option-format failure. The Hausa rendering broke an option's surface form so "
        "the choice set stopped functioning: numbers, units, ranges or doses mangled; "
        "two options collapsed to the same string; an option truncated or merged into "
        "another. This does NOT mean the model failed to emit a letter -- that is "
        "recorded automatically in the 'unparseable' column, not coded by hand."
    ),
    "term": (
        "Medical-term loss. For the coded segment the Hausa differs from the English "
        "AND a medical term's referent is lost or changed: wrong drug, wrong anatomical "
        "structure, wrong condition, wrong procedure. A term carried over unchanged in "
        "English or Latin is NOT term loss -- the model is shown the same string in "
        "both conditions, so that segment cannot be the source of the difference."
    ),
    "translation": (
        "Translation error. For the coded segment the Hausa differs from the English "
        "AND the non-terminological meaning is wrong: negation dropped or added, a "
        "quantifier or comparative flipped, wrong clause attachment, or syntax that "
        "changes what is being asked."
    ),
    "reasoning": (
        "Reasoning failure. Residual category. The note must affirm that the question "
        "stem, the gold option and the option the model chose are all adequate Hausa, "
        "so the item was answerable as presented and the model simply got it wrong."
    ),
}

# Segment-status labels shown on the sheet, and the groups the summary reports.
STATUS_LABEL = {
    "translated": "auto", "retried": "auto",
    "kept_after_review": "reviewed_kept", "post_edited": "reviewed_edited",
    "fallback_en_reviewed": "english", "fallback_en": "english",
    "passthrough": "code_or_number",
}
LABEL_GROUP = {
    "auto": "unreviewed", "reviewed_kept": "reviewed", "reviewed_edited": "reviewed",
    "english": "english", "code_or_number": "code_or_number",
}

CONTEXT_COLS = (
    ["row", "subject", "question_en", "question_ha", "question_status"]
    + [f"option_{L}_{f}" for L in C.LETTERS for f in ("en", "ha", "status")]
    + ["gold", "model_answer_ha", "model_raw_output"]
)
HELPER_COLS = ["unparseable", "identical_options", "n_identical_options", "codeable_segments"]
CODING_COLS = ["code", "segment", "notes"]
SHEET_COLS = CONTEXT_COLS + HELPER_COLS + CODING_COLS

_ws = re.compile(r"\s+")


def _norm(text) -> str:
    return _ws.sub(" ", str(text)).strip()


def _outputs(model, cond):
    return {r["id"]: r for r in D.read_jsonl(C.OUTPUTS / model / f"{cond}.jsonl")}


def draw_sample(ids):
    """ids: evaluation item ids in file order. Returns (rows, pool sizes).
    rows: list of dicts (row, model, id), already shuffled. Needs cached outputs."""
    rng = random.Random(C.E1_SEED)
    picked, pools = [], {}
    for m in C.E1_MODELS:
        en, ha = _outputs(m, "en_p0"), _outputs(m, "ha_p0")
        assert set(en) == set(ids) == set(ha), f"{m}: outputs incomplete"
        pool = [i for i in ids if en[i]["correct"] and not ha[i]["correct"]]
        pools[m] = len(pool)
        picked += [(m, i) for i in rng.sample(pool, C.E1_N_PER_MODEL)]
    rng.shuffle(picked)
    rows = [{"row": f"e1_{k + 1:02d}", "model": m, "id": i} for k, (m, i) in enumerate(picked)]
    return rows, pools


def coding_sheet(rows, eval_en, eval_ha, seg_status):
    """One row per sampled (model, item), model hidden. Canonical option order."""
    en_by, ha_by = {r["id"]: r for r in eval_en}, {r["id"]: r for r in eval_ha}
    ha_out = {m: _outputs(m, "ha_p0") for m in C.E1_MODELS}
    out = []
    for r in rows:
        i, en, ha = r["id"], en_by[r["id"]], ha_by[r["id"]]
        assert en["perms"][0] == [0, 1, 2, 3], "ha_p0 must be the canonical order"
        o = ha_out[r["model"]][i]
        row = {"row": r["row"], "subject": en["subject"],
               "question_en": en["question"], "question_ha": ha["question"],
               "question_status": STATUS_LABEL[seg_status[f"{i}|q"]]}
        for j, L in enumerate(C.LETTERS):
            row[f"option_{L}_en"] = en["options"][j]
            row[f"option_{L}_ha"] = ha["options"][j]
            row[f"option_{L}_status"] = STATUS_LABEL[seg_status[f"{i}|o{j}"]]
        row.update({"gold": C.LETTERS[en["answer_idx"]],
                    "model_answer_ha": C.LETTERS[o["choice"]] if o["choice"] >= 0 else "none",
                    "model_raw_output": o["raw"]})
        out.append(row)
    return blank_sheet(pd.DataFrame(out))


# ---------- sheet preparation ----------

def segment_pair(row, seg):
    """(english, hausa, status label) for one segment of a sheet row."""
    if str(seg).lower() == "stem":
        return row["question_en"], row["question_ha"], row["question_status"]
    return row[f"option_{seg}_en"], row[f"option_{seg}_ha"], row[f"option_{seg}_status"]


def identical_segments(row):
    """Letters whose Hausa option is identical to its English. Such a segment is
    shown to the model unchanged, so it cannot carry a translation or term error."""
    return [L for L in C.LETTERS if _norm(row[f"option_{L}_en"]) == _norm(row[f"option_{L}_ha"])]


def codeable_segments(row):
    """Segments a term/translation error may sit in: the stem, the gold option and
    the option the model chose. An error elsewhere cannot explain a discordance."""
    segs = ["stem", row["gold"]]
    if row["model_answer_ha"] in C.LETTERS and row["model_answer_ha"] not in segs:
        segs.append(row["model_answer_ha"])
    return segs


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add the automatic helper columns. These are computed, never hand-entered."""
    df = df.copy()
    ident = df.apply(identical_segments, axis=1)
    df["unparseable"] = (df["model_answer_ha"] == "none").map({True: "yes", False: ""})
    df["identical_options"] = ident.map(",".join)
    df["n_identical_options"] = ident.map(len)
    df["codeable_segments"] = df.apply(lambda r: ",".join(codeable_segments(r)), axis=1)
    return df


def blank_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Context + helper columns with code/segment/notes blank, ready for a coder."""
    df = annotate(df)
    for c in CODING_COLS:
        df[c] = ""
    return df[SHEET_COLS]


def read_sheet(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


# ---------- validation ----------

def check_codes(df: pd.DataFrame, expected_rows=None):
    """Return the list of rule violations in a completed sheet (empty if clean).

    R1 code in CODES                R4 a coded segment's HA must differ from its EN
    R2 note on every row            R5 term/translation must sit in stem/gold/chosen
    R3 segment present iff needed   R6 row set matches the exported sheet
    """
    errors = []
    if expected_rows is not None and set(df["row"]) != set(expected_rows):
        errors.append("R6: rows in the completed sheet do not match the exported sheet")
    for r in df.to_dict("records"):
        tag, code, seg = r["row"], r["code"], str(r["segment"]).strip()
        if code not in CODES:
            errors.append(f"{tag}: R1 code {code!r} not in {CODES}")
            continue
        if len(str(r["notes"]).strip()) < MIN_NOTE_CHARS:
            errors.append(f"{tag}: R2 needs a note of at least {MIN_NOTE_CHARS} characters "
                          f"saying what the error is and how it changed the answer")
        if code in NEEDS_SEGMENT and seg not in SEGMENTS:
            errors.append(f"{tag}: R3 code {code!r} needs segment = stem, A, B, C or D")
            continue
        if code in NO_SEGMENT and seg:
            errors.append(f"{tag}: R3 code {code!r} must leave segment empty")
            continue
        if not seg:
            continue
        if seg not in SEGMENTS:
            errors.append(f"{tag}: R3 segment {seg!r} not in {SEGMENTS}")
            continue
        en, ha, _ = segment_pair(r, seg)
        if _norm(en) == _norm(ha):
            errors.append(f"{tag}: R4 segment {seg} is identical in English and Hausa "
                          f"({_norm(en)[:40]!r}), so it cannot carry a {code} error")
        if code in NEEDS_SEGMENT and seg != "stem" and seg not in codeable_segments(r):
            errors.append(f"{tag}: R5 segment {seg} is neither the gold answer ({r['gold']}) "
                          f"nor the model's choice ({r['model_answer_ha']}), so an error there "
                          f"cannot explain the discordance")
    return errors


def read_codes(path, key_path=None, expected_rows=None) -> pd.DataFrame:
    """Validate a completed sheet, raising on any violation. Joins the key if given."""
    df = read_sheet(path)
    df["code"] = df["code"].str.strip().str.lower()
    df["segment"] = df["segment"].str.strip().map(
        lambda s: "stem" if s.lower() == "stem" else s.upper())
    errors = check_codes(df, expected_rows)
    if errors:
        raise ValueError(f"{len(errors)} problem(s) in {path}:\n" + "\n".join(errors[:40]))
    if key_path is not None:
        df = df.merge(read_sheet(key_path), on="row")
    return df


# ---------- inter-coder agreement ----------

def cohen_kappa(a, b, labels=CODES) -> float:
    """Unweighted Cohen's kappa for two equal-length label sequences."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(L) / n) * (b.count(L) / n) for L in labels)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def agreement(a, b) -> dict:
    """Raw agreement and kappa. Over 50 rows and four unevenly used categories
    kappa is unstable, so raw agreement is reported beside it -- the same caveat
    PREREGISTRATION.md section 4 applies to the translation raters."""
    a, b = list(a), list(b)
    raw = sum(x == y for x, y in zip(a, b)) / len(a) if a else float("nan")
    return {"n": len(a), "raw_agreement": round(raw, 3),
            "cohen_kappa": round(cohen_kappa(a, b), 3),
            "confusion": pd.crosstab(pd.Series(a, name="coder1"),
                                     pd.Series(b, name="coder2")).to_dict()}


def note_similarity(merged: pd.DataFrame) -> dict:
    """How textually alike the two coders' free-text notes are.

    Independent coders describe the same error in their own words, so their notes
    overlap in content but not in wording. Near-identical notes mean the second
    sheet was written with the first in view, in which case the codes are not
    independent and the agreement figures are not a reliability estimate. The
    check is reported either way so that independence is evidenced, not asserted.
    """
    def norm(s):
        s = str(s).replace("‘", "'").replace("’", "'").lower()
        return _ws.sub(" ", s).strip()

    sims = pd.Series([difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
                      for a, b in zip(merged["notes_1"], merged["notes_2"])])
    return {"n": int(len(sims)), "median": round(float(sims.median()), 3),
            "mean": round(float(sims.mean()), 3), "min": round(float(sims.min()), 3),
            "max": round(float(sims.max()), 3),
            "n_at_least_0.95": int((sims >= 0.95).sum()),
            "n_at_least_0.80": int((sims >= 0.80).sum()),
            "independent": bool(sims.median() < 0.5)}


def merge_coders(path1, path2):
    """Validate both sheets and compare. Returns (merged, agreement, disagreements).
    `disagreements` is a sheet for adjudication: both coders' judgements are shown
    and code/segment/notes are blank for the adjudicator to fill."""
    c1, c2 = read_codes(path1), read_codes(path2)
    if set(c1["row"]) != set(c2["row"]):
        raise ValueError("the two coder sheets cover different rows")
    merged = c1.merge(c2[["row"] + CODING_COLS], on="row", suffixes=("_1", "_2"))
    stats = agreement(merged["code_1"], merged["code_2"])
    stats["note_similarity"] = note_similarity(merged)
    dis = merged[merged["code_1"] != merged["code_2"]].copy()
    for c in CODING_COLS:
        dis[c] = ""
    both = [f"{c}_{i}" for i in (1, 2) for c in CODING_COLS]
    return merged, stats, dis[CONTEXT_COLS + HELPER_COLS + both + CODING_COLS]


def finalise(merged: pd.DataFrame, adjudicated_path=None) -> pd.DataFrame:
    """One final code per row: the coders' agreed code, or the adjudicated one."""
    out = merged.copy()
    agreed = out["code_1"] == out["code_2"]
    for c in CODING_COLS:
        out[c] = out[f"{c}_1"].where(agreed, "")
    out["adjudicated"] = ~agreed
    if (~agreed).any():
        if adjudicated_path is None:
            raise ValueError(f"{int((~agreed).sum())} rows disagree and need adjudication")
        adj = read_sheet(adjudicated_path).set_index("row")
        for c in CODING_COLS:
            out.loc[~agreed, c] = out.loc[~agreed, "row"].map(adj[c]).values
    errors = check_codes(out)
    if errors:
        raise ValueError("adjudicated codes are invalid:\n" + "\n".join(errors[:40]))
    return out


# ---------- summary ----------

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def _code_table(sub: pd.DataFrame) -> dict:
    n = len(sub)
    return {c: {"n": int((sub.code == c).sum()),
                "prop": round(float((sub.code == c).mean()), 3),
                "ci95": wilson(int((sub.code == c).sum()), n)} for c in CODES}


def _locations(sub: pd.DataFrame) -> dict:
    """Where the text errors sit. R5 means an option locus is gold or chosen."""
    roles, statuses = [], []
    for r in sub[sub.code.isin(NEEDS_SEGMENT)].to_dict("records"):
        seg = r["segment"]
        _, _, label = segment_pair(r, seg)
        roles.append("stem" if seg == "stem" else "gold" if seg == r["gold"] else "chosen")
        statuses.append(LABEL_GROUP[label])
    return {"text_error_location": pd.Series(roles).value_counts().to_dict(),
            "text_error_segment_status": pd.Series(statuses).value_counts().to_dict()}


def summarise(coded: pd.DataFrame, agreement_stats=None, pools=None,
              coding_procedure=None) -> dict:
    """Pooled proportions are primary. Per-model figures are descriptive only,
    because the author re-coded unblinded (deviation 5)."""
    coded = annotate(coded)
    out = {
        "n_rows": len(coded),
        "codebook": "deviation-5",
        "primary_pooled": {"n": len(coded), "codes": _code_table(coded), **_locations(coded)},
        "descriptive_per_model": {
            m: {"n": len(sub), "codes": _code_table(sub), **_locations(sub)}
            for m, sub in coded.groupby("model")
        },
        "untranslated_options": {
            "rows_with_any_identical_option": int((coded.n_identical_options > 0).sum()),
            "mean_identical_options_per_row": round(float(coded.n_identical_options.mean()), 3),
        },
        "unparseable_rows": int((coded.unparseable == "yes").sum()),
        "items_drawn_for_both_models": int(coded.groupby("id").model.nunique().eq(2).sum()),
        "sample": {"models": C.E1_MODELS, "n_per_model": C.E1_N_PER_MODEL,
                   "seed": C.E1_SEED, "discordant_pool_sizes": pools},
    }
    if "adjudicated" in coded:
        out["adjudicated_rows"] = int(coded["adjudicated"].sum())
    if agreement_stats is not None:
        out["inter_coder_agreement"] = agreement_stats
        sim = agreement_stats.get("note_similarity")
        if sim is not None and not sim["independent"]:
            out["inter_coder_agreement"]["not_a_reliability_estimate"] = (
                f"The two coders' notes have median textual similarity "
                f"{sim['median']} and {sim['n_at_least_0.95']} of {sim['n']} are "
                f"near-identical, so the second sheet was not written independently "
                f"of the first. Raw agreement and kappa above describe how far the "
                f"second pass confirmed the first; they do not estimate inter-coder "
                f"reliability and must not be reported as such."
            )
    if coding_procedure is not None:
        out["coding_procedure"] = coding_procedure
    return out
