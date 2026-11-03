"""H4 DPO intervention: chain-of-thought sampling and preference-pair
construction (PREREGISTRATION.md section 6).

Ported from the author's prior study `dpo-reasoning-transfer` (src/prefs.py),
which section 6 requires this study to follow. Two things are adapted and
nothing else: the task is 4-option multiple choice rather than GSM8K, so the
verifier compares answer letters instead of numbers; and generation is made
resumable through a JSONL cache, matching evaluate.run_condition, because
Colab sessions drop.

The pair-construction design is deliberately unchanged from the prior study:
`chosen` and `rejected` are drawn uniformly at random from the correct and
incorrect chains, with no preference for length, so no anti-length signal is
introduced. Per-chain lengths are recorded so any residual length effect can
be discussed.

Spurious correctness is the known weakness of a final-answer verifier: a chain
can land on the right letter through broken reasoning. On a 4-option task with
a base model near chance this is not a footnote but the central risk, which is
why notebook 03a measures it before any training is run.
"""
import json
import random
import re
from pathlib import Path

import torch

from . import config as C
from . import data as D

# Final-answer extraction. Unlike evaluate.parse_letter, which reads the FIRST
# letter because generation is prefilled with "Amsa:" and capped at 8 tokens,
# a chain of thought ends with its answer, so these search from the END.
_FINAL = re.compile(
    r"(?i:amsa|answer)\W{0,5}(?:(?i:ita ce|shine|ne|is)\W{1,4})?([ABCDabcd])(?![A-Za-z])")
# The bare-letter fallback is anchored to the very END of the chain, not merely
# to its closing lines. Hausa sentences open with "A" constantly ("A cikin...",
# "A nan...", "A karshe..."), so a rule that accepted a letter anywhere in the
# tail reads that preposition as an answer: it would both inflate the parse rate
# and bias every cue-less chain towards A, on a task whose headline finding is
# about an answer-letter baseline.
_TAIL = re.compile(r"(?<![A-Za-z])\(?([ABCDabcd])[.)\]:,]*\s*$")


def final_letter(text: str) -> int:
    """Position 0-3 of the chain's final answer letter, or -1 if none is found.

    Tried in order: the last explicit "Amsa:/Answer: X" cue anywhere in the
    chain, then a bare letter ending the chain. The second rule is anchored to
    the end so that a letter mentioned while reasoning -- or a Hausa sentence
    opening with the preposition "A" -- is not mistaken for the conclusion.
    Both rules accept a lower-case letter; the model is asked for "Amsa: C" but
    a 3B model at temperature 0.8 does not reliably capitalise.
    """
    matches = list(_FINAL.finditer(text))
    if matches:
        return C.LETTERS.index(matches[-1].group(1).upper())
    tail = "\n".join([ln for ln in text.strip().splitlines() if ln.strip()][-2:])
    matches = list(_TAIL.finditer(tail))
    if matches:
        return C.LETTERS.index(matches[-1].group(1).upper())
    return -1


def cot_prompt(rec: dict) -> str:
    """Canonical option order: pair generation never reorders options."""
    opts = "\n".join(f"{L}. {rec['options'][j]}" for j, L in enumerate(C.LETTERS))
    return C.DPO_COT_PROMPT.format(question=rec["question"], options=opts)


def hausa_stem_items(records, seg_status):
    """Training-pool items whose question stem is actually in Hausa. Section 6
    excludes the stems that fell back to English (70 of 500)."""
    return [r for r in records if seg_status[f"{r['id']}|q"] != "fallback_en"]


def pilot_subset(records, n=C.DPO_PILOT_N, seed=C.DPO_SEED):
    """A seeded subset for the gate in notebook 03a. The full run covers every
    item anyway, so the pilot's cached completions are reused, not discarded."""
    rng = random.Random(seed)
    return sorted(rng.sample(list(records), min(n, len(records))), key=lambda r: r["id"])


@torch.inference_mode()
def sample_completions(tok, model, rec, k=C.DPO_K, seed=None):
    """k stochastic chains for one item, decoded in a single call via
    num_return_sequences, as the prior study does."""
    if seed is not None:
        torch.manual_seed(seed)
    text = tok.apply_chat_template([{"role": "user", "content": cot_prompt(rec)}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=C.DPO_MAX_NEW_TOKENS, do_sample=True,
                         temperature=C.DPO_TEMPERATURE, top_p=C.DPO_TOP_P,
                         num_return_sequences=k, pad_token_id=tok.pad_token_id)
    prompt_len = enc["input_ids"].shape[1]
    return [tok.decode(s[prompt_len:], skip_special_tokens=True).strip() for s in out]


def generate_pool(tok, model, records, cache_path, k=C.DPO_K):
    """Resumable: items already cached are skipped. Returns rows in record order."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
    # Section 6 raises k to 8 if the pilot yields fewer than 200 pairs, and the
    # obvious way to do that is to pass k=8 here. Without this check the cached
    # k=4 rows would simply be returned, and gate_report would quietly describe
    # a k=4 run as the k=8 one. Caches are keyed by k in their filename.
    stale = sorted({r["k"] for r in done.values()} - {k})
    if stale:
        raise ValueError(
            f"{cache_path} holds rows generated with k={stale} but k={k} was asked "
            f"for. Use a cache path keyed to k (dpo_cot_k{k}.jsonl) so the two runs "
            f"stay separate, or delete the file to regenerate.")
    todo = [r for r in records if r["id"] not in done]
    for rec in todo:
        chains = sample_completions(tok, model, rec, k=k, seed=C.DPO_SEED)
        picks = [final_letter(c) for c in chains]
        row = {"id": rec["id"], "gold": rec["answer_idx"], "k": k, "chains": chains,
               "picks": picks, "correct": [p == rec["answer_idx"] for p in picks]}
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[rec["id"]] = row
    return [done[r["id"]] for r in records]


# ---------- preference pairs ----------

def build_pair(row, rec, rng):
    """One (chosen, rejected) pair, or None if every chain agrees. Mirrors
    build_preference_pair in dpo-reasoning-transfer/src/prefs.py."""
    correct = [(i, c) for i, c in enumerate(row["chains"]) if row["correct"][i]]
    incorrect = [(i, c) for i, c in enumerate(row["chains"]) if not row["correct"][i]]
    if not correct or not incorrect:
        return None
    ci, chosen = rng.choice(correct)
    ri, rejected = rng.choice(incorrect)
    return {"id": row["id"], "question": rec["question"], "options": rec["options"],
            "gold": C.LETTERS[row["gold"]], "chosen": chosen, "rejected": rejected,
            "chosen_letter": C.LETTERS[row["picks"][ci]] if row["picks"][ci] >= 0 else "none",
            "rejected_letter": C.LETTERS[row["picks"][ri]] if row["picks"][ri] >= 0 else "none",
            "chosen_len_chars": len(chosen), "rejected_len_chars": len(rejected),
            "chosen_idx": ci, "rejected_idx": ri,
            "n_correct": len(correct), "n_incorrect": len(incorrect),
            "all_picks": [C.LETTERS[p] if p >= 0 else "none" for p in row["picks"]]}


def build_pairs(rows, records, seed=C.DPO_SEED):
    rng = random.Random(seed)
    by_id = {r["id"]: r for r in records}
    pairs = [build_pair(row, by_id[row["id"]], rng) for row in rows]
    return [p for p in pairs if p is not None]


def format_for_trl(pair, tok):
    """TRL DPOTrainer wants prompt / chosen / rejected as plain strings."""
    rec = {"question": pair["question"], "options": pair["options"]}
    prompt = tok.apply_chat_template([{"role": "user", "content": cot_prompt(rec)}],
                                     tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "chosen": pair["chosen"], "rejected": pair["rejected"]}


# ---------- the pilot gate ----------

def gate_report(rows, pairs, n_full=None) -> dict:
    """Does pair generation carry enough signal to be worth training on?

    Reports three things the decision turns on: whether enough pairs will be
    produced, whether the chains can be parsed at all, and how much of the
    'correct' label is skill rather than luck. On a 4-option task a base model
    at chance produces pairs whose chosen side is mostly a lucky guess, and
    DPO on those teaches a preference for guessing, not for reasoning.
    """
    k = rows[0]["k"] if rows else 0
    n_items = len(rows)
    picks = [p for r in rows for p in r["picks"]]
    corr = [c for r in rows for c in r["correct"]]
    n_samples = len(picks)
    parse_rate = sum(p >= 0 for p in picks) / n_samples if n_samples else float("nan")
    acc = sum(corr) / n_samples if n_samples else float("nan")
    parsed = [c for p, c in zip(picks, corr) if p >= 0]
    acc_parsed = sum(parsed) / len(parsed) if parsed else float("nan")
    yield_rate = len(pairs) / n_items if n_items else float("nan")
    # Share of correct chains that are chance rather than knowledge, from
    # acc = skill + (1 - skill) / 4 on a 4-option task.
    skill = max(0.0, (acc - 0.25) / 0.75)
    luck_share = 1 - skill / acc if acc > 0 else float("nan")
    return {
        "n_items": n_items, "k": k, "n_samples": n_samples,
        "parse_rate": round(parse_rate, 3),
        "accuracy_all_samples": round(acc, 3),
        "accuracy_parsed_only": round(acc_parsed, 3),
        "n_correct_per_item": {str(i): sum(sum(r["correct"]) == i for r in rows)
                               for i in range(k + 1)},
        "pairs": len(pairs), "pair_yield_rate": round(yield_rate, 3),
        "projected_pairs_full_pool": (None if n_full is None or not n_items
                                      else int(round(yield_rate * n_full))),
        "min_pairs_threshold": C.DPO_MIN_PAIRS,
        "estimated_skill_rate": round(skill, 3),
        "estimated_luck_share_of_chosen": round(luck_share, 3),
        "reference_points": {"chance": 0.25, "majority_letter_baseline_p0": 0.318,
                             "greedy_letter_only_ha_p0": 0.276},
    }


def inspection_sample(pairs, n=10, seed=C.DPO_SEED):
    """Correct chains to read by hand. The numbers above cannot tell you whether
    a chain reasons correctly or guesses well -- only reading them can."""
    rng = random.Random(seed)
    picked = rng.sample(pairs, min(n, len(pairs)))
    return [{"id": p["id"], "gold": p["gold"], "chosen_letter": p["chosen_letter"],
             "n_correct_of_k": p["n_correct"], "chosen_chain": p["chosen"]}
            for p in picked]


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return D.sha256(path)
