"""E1 exploratory error analysis (PREREGISTRATION.md section 3; sample fixed in
deviation 4). No hypothesis test is attached.

Discordant item = correct in en_p0 and wrong in ha_p0 (primary, parsed scoring).
25 items are drawn per model (C.E1_MODELS, in that order, one seeded RNG). An
item may be drawn for both models; each (model, item) row is coded separately.

The coding sheet does not show which model produced a row, and rows are
shuffled. The key mapping rows to models is written to a separate file that
should not be opened until coding is finished.
"""
import math
import random

import pandas as pd

from . import config as C
from . import data as D

# Pre-registered categories, checked in this order (first that applies wins)
CODES = ["format", "term", "translation", "reasoning"]
NEEDS_SEGMENT = {"term", "translation"}
SEGMENTS = ["stem"] + C.LETTERS

STATUS_LABEL = {  # sheet shows these; the summary uses the raw status
    "translated": "auto", "retried": "auto",
    "kept_after_review": "reviewed_kept", "post_edited": "reviewed_edited",
    "fallback_en_reviewed": "english", "fallback_en": "english",
    "passthrough": "code_or_number",
}
STATUS_GROUP = {
    "translated": "unreviewed", "retried": "unreviewed",
    "kept_after_review": "reviewed", "post_edited": "reviewed",
    "fallback_en_reviewed": "english", "fallback_en": "english",
    "passthrough": "code_or_number",
}


def _outputs(model, cond):
    return {r["id"]: r for r in D.read_jsonl(C.OUTPUTS / model / f"{cond}.jsonl")}


def draw_sample(ids):
    """ids: evaluation item ids in file order. Returns (rows, pool sizes).
    rows: list of dicts (row, model, id), already shuffled."""
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
                    "model_raw_output": o["raw"],
                    "code": "", "segment": "", "notes": ""})
        out.append(row)
    return pd.DataFrame(out)


def read_codes(path, key_path):
    """Validate the completed sheet and join the hidden key back on."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["code"] = df["code"].str.strip().str.lower()
    df["segment"] = df["segment"].str.strip().map(lambda s: "stem" if s.lower() == "stem" else s.upper())
    errors = []
    if set(df.row) != set(key.row):
        errors.append("rows in the completed sheet don't match the exported sheet")
    for r in df.itertuples(index=False):
        if r.code not in CODES:
            errors.append(f"{r.row}: code '{r.code}' not in {CODES}")
        elif r.code in NEEDS_SEGMENT and r.segment not in SEGMENTS:
            errors.append(f"{r.row}: code '{r.code}' needs segment = stem, A, B, C or D")
        elif r.code not in NEEDS_SEGMENT and r.segment:
            errors.append(f"{r.row}: code '{r.code}' should have an empty segment")
    if errors:
        raise ValueError("Fix these rows in the sheet:\n" + "\n".join(errors[:30]))
    return df.merge(key, on="row")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def summarise(coded, eval_en, seg_status):
    en_by = {r["id"]: r for r in eval_en}
    ha_out = {m: _outputs(m, "ha_p0") for m in C.E1_MODELS}

    def where(r):
        """(role, status group) of the segment that carries the error."""
        if r.segment == "stem":
            return "stem", STATUS_GROUP[seg_status[f"{r.id}|q"]]
        j = C.LETTERS.index(r.segment)
        role = ("gold" if j == en_by[r.id]["answer_idx"]
                else "chosen" if j == ha_out[r.model][r.id]["choice"] else "other_option")
        return role, STATUS_GROUP[seg_status[f"{r.id}|o{j}"]]

    out = {"n_rows": len(coded), "per_model": {},
           "items_drawn_for_both_models": int(coded.groupby("id").model.nunique().eq(2).sum())}
    for m, sub in list(coded.groupby("model")) + [("all", coded)]:
        n = len(sub)
        codes = {c: {"n": int((sub.code == c).sum()), "prop": round(float((sub.code == c).mean()), 3),
                     "ci95": wilson(int((sub.code == c).sum()), n)} for c in CODES}
        err = sub[sub.code.isin(NEEDS_SEGMENT)]
        locs = pd.DataFrame([where(r) for r in err.itertuples(index=False)],
                            columns=["role", "status"]) if len(err) else pd.DataFrame(columns=["role", "status"])
        out["per_model"][m] = {
            "n": n, "codes": codes,
            "text_error_location": locs.role.value_counts().to_dict(),
            "text_error_segment_status": locs.status.value_counts().to_dict(),
        }
    return out
