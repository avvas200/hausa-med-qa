"""Sampling, filtering, disjointness checks and record assembly."""
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

from . import config as C

OPTS = ["opa", "opb", "opc", "opd"]
_punct = re.compile(r"[^\w\s]")
_ws = re.compile(r"\s+")


def normalize(text) -> str:
    return _ws.sub(" ", _punct.sub(" ", str(text).lower())).strip()


# MedMCQA scraping artifact: the letter pair "rt" was dropped from many words
# ("aery" = artery, "impoant" = important). Best-effort repair list; every
# replacement is counted and a sample is shown in notebook 01 for checking.
REPAIRS = [
    (r"\baer(y|ies|ial|iole|ioles|iolar|itis)\b", r"arter\1"),
    (r"\baerio", "arterio"),
    (r"\baoa\b", "aorta"), (r"\baoic\b", "aortic"),
    (r"hypeension", "hypertension"), (r"hypeensive", "hypertensive"),
    (r"hyperoph", "hypertroph"), (r"hypeon", "hyperton"),
    (r"hypehyro", "hyperthyro"), (r"hypeherm", "hypertherm"),
    (r"\bimpoan(t|ce|tly)\b", r"importan\1"),
    (r"\bpoion(s?)\b", r"portion\1"), (r"\bpoal\b", "portal"),
    (r"\bcoex\b", "cortex"), (r"\bcoical\b", "cortical"),
    (r"\bcoico", "cortico"), (r"\bcoisol\b", "cortisol"),
    (r"\b(osteo|poly|mono|peri|haem|hem)?ahr(?=[aeio])", r"\1arthr"),
    (r"\bhea\b", "heart"), (r"\bheaburn\b", "heartburn"), (r"\bheabeat", "heartbeat"),
    (r"\bpaial(ly)?\b", r"partial\1"), (r"\bpaicula", "particula"),
    (r"\bpaicle(s?)\b", r"particle\1"), (r"\bpauri", "parturi"),
    (r"\bmoality\b", "mortality"), (r"\binseion(s?)\b", r"insertion\1"),
    (r"aicular\b", "articular"), (r"\baificial", "artificial"),
    (r"\bceain(ly)?\b", r"certain\1"), (r"\bcailag", "cartilag"),
    (r"\bsho(er|est|ness|ening)?\b", r"short\1"),
    (r"\bsuppo(s|ed|ing|ive)?\b", r"support\1"),
    (r"\bfouh\b", "fourth"), (r"veebra", "vertebra"),
    (r"\b(?-i:sta(ed|ing)?)\b", r"start\1"),  # lowercase only, keeps STA (artery)
]
_REPAIRS = [(re.compile(p, re.I), r) for p, r in REPAIRS]


def repair_text(text: str, counter: dict | None = None) -> str:
    for pat, repl in _REPAIRS:
        def sub(m, repl=repl, name=pat.pattern):
            if counter is not None:
                counter[name] = counter.get(name, 0) + 1
            out = m.expand(repl)
            w = m.group(0)
            if len(w) > 1 and w.isupper():
                return out.upper()
            return out[0].upper() + out[1:] if w[0].isupper() else out
        text = pat.sub(sub, text)
    return text


def repair_df(df: pd.DataFrame):
    """Repair question and option text. Returns (df, counts, changed_row_mask)."""
    counts = {}
    df = df.copy()
    before = df[["question"] + OPTS].astype(str).agg("||".join, axis=1)
    for col in ["question"] + OPTS:
        df[col] = df[col].map(lambda x: repair_text(x, counts) if isinstance(x, str) else x)
    after = df[["question"] + OPTS].astype(str).agg("||".join, axis=1)
    return df, counts, (before != after).values


def load_split(split: str) -> pd.DataFrame:
    from datasets import load_dataset
    return load_dataset(C.MEDMCQA, split=split).to_pandas()


def apply_filters(df: pd.DataFrame):
    """Return (filtered_df, log of how many rows each rule removed)."""
    log = {"start": len(df)}
    opt_pat = re.compile("|".join(C.EXCLUDE_OPTION_PATTERNS), re.I)
    q_pat = re.compile("|".join(C.EXCLUDE_QUESTION_PATTERNS), re.I)

    rules = [
        ("excluded_subject", lambda d: ~d["subject_name"].isin(C.EXCLUDE_SUBJECTS)),
        ("multi_answer", lambda d: d["choice_type"] == "single"),
        ("bad_label", lambda d: d["cop"].isin([0, 1, 2, 3])),
        ("empty_option", lambda d: d[OPTS].apply(
            lambda r: all(isinstance(x, str) and x.strip() for x in r), axis=1)),
        ("duplicate_options", lambda d: d[OPTS].apply(
            lambda r: len({normalize(x) for x in r}) == 4, axis=1)),
        ("position_dependent_option", lambda d: ~d[OPTS].apply(
            lambda r: any(opt_pat.search(x) for x in r), axis=1)),
        ("needs_image", lambda d: ~d["question"].str.contains(q_pat, na=False)),
        ("too_long", lambda d: d["question"].str.split().str.len() <= C.MAX_QUESTION_WORDS),
    ]
    for name, rule in rules:
        mask = rule(df).astype(bool)
        log[name] = int((~mask).sum())
        df = df[mask]
    log["remaining"] = len(df)
    return df.reset_index(drop=True), log


def stratified_sample(df: pd.DataFrame, n: int, seed: int, by: str = "subject_name"):
    """Proportional allocation with largest-remainder rounding."""
    counts = df[by].value_counts().sort_index()
    quotas = counts / counts.sum() * n
    base = quotas.astype(int)
    remainder = (quotas - base).sort_values(ascending=False, kind="mergesort")
    base[remainder.index[: n - int(base.sum())]] += 1
    parts = [df[df[by] == s].sample(n=int(k), random_state=seed)
             for s, k in base.items() if k > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def drop_overlaps(pool: pd.DataFrame, ref: pd.DataFrame, thresh: float = C.NEAR_DUP_JACCARD):
    """Remove pool rows that exactly or nearly duplicate any ref question."""
    ref_norm = [normalize(q) for q in ref["question"]]
    ref_exact = set(ref_norm)
    ref_sets = [set(t.split()) for t in ref_norm]
    keep, dropped = [], 0
    for q in pool["question"]:
        nq = normalize(q)
        s = set(nq.split())
        bad = nq in ref_exact or any(_jaccard(s, r) >= thresh for r in ref_sets)
        keep.append(not bad)
        dropped += bad
    return pool[keep].reset_index(drop=True), dropped


def make_permutations(rng: random.Random, k: int = C.N_EXTRA_PERMUTATIONS):
    """perm[j] = index of the original option shown at position j.
    First entry is always the canonical order; the rest are distinct."""
    perms = [[0, 1, 2, 3]]
    while len(perms) < k + 1:
        p = [0, 1, 2, 3]
        rng.shuffle(p)
        if p not in perms:
            perms.append(p)
    return perms


def to_records(df: pd.DataFrame, seed: int, source: str = "medmcqa"):
    rng = random.Random(seed)
    records = []
    for row in df.itertuples(index=False):
        records.append({
            "id": row.id,
            "source": source,
            "subject": row.subject_name,
            "topic": row.topic_name,
            "question": row.question.strip(),
            "options": [getattr(row, o).strip() for o in OPTS],
            "answer_idx": int(row.cop),
            "perms": make_permutations(rng),
            "lang": "en",
        })
    return records


def segments(records):
    """Flatten records into translatable segments keyed by id|field."""
    out = {}
    for r in records:
        out[f"{r['id']}|q"] = r["question"]
        for j, o in enumerate(r["options"]):
            out[f"{r['id']}|o{j}"] = o
    return out


def apply_translations(records, translated: dict, lang: str):
    out = []
    for r in records:
        t = dict(r)
        t["question"] = translated[f"{r['id']}|q"]
        t["options"] = [translated[f"{r['id']}|o{j}"] for j in range(4)]
        t["lang"] = lang
        out.append(t)
    return out


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
