"""Baseline evaluation (notebook 02). Committed before the first evaluation run;
the template, decoding and parsing rules follow PREREGISTRATION.md section 5.

Records carry perms[k][j] = index of the original option shown at position j.
Each output row stores the parsed position, the original option it maps to
(`choice`, -1 if unparseable) and whether it is correct.
"""
import json
import re
from pathlib import Path

import torch

from . import config as C

# Fixed answer-letter parser, tried in this order (case-sensitive letters):
#   1. output starts with the letter ("B", "B.", "**B**", "(B) ...")
#   2. keyword then letter ("Answer: B", "The answer is B", "Amsa: B", "Amsar ita ce B")
#   3. a letter followed by . ) or : anywhere ("... option C) ...")
_START = re.compile(r"^\W*([ABCD])(?![A-Za-z])")
_KEY = re.compile(r"(?i:answer|amsa\w*)\W{0,5}(?:(?i:is|ita ce|shine|ne)\W{1,4})?([ABCD])(?![A-Za-z])")
_MARK = re.compile(r"(?<![A-Za-z])\(?([ABCD])[.):](?![A-Za-z])")


def parse_letter(text: str) -> int:
    """Position 0-3 of the answer letter, or -1 if none can be parsed."""
    for pat in (_START, _KEY, _MARK):
        m = pat.search(text)
        if m:
            return C.LETTERS.index(m.group(1))
    return -1


def prompt_text(rec: dict, perm, lang: str) -> str:
    opts = "\n".join(f"{L}. {rec['options'][perm[j]]}" for j, L in enumerate(C.LETTERS))
    return C.EVAL_PROMPTS[lang].format(question=rec["question"], options=opts)


def chat_prompt(tok, rec, perm, lang):
    return tok.apply_chat_template([{"role": "user", "content": prompt_text(rec, perm, lang)}],
                                   tokenize=False, add_generation_prompt=True)


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda")
    except (ValueError, KeyError):  # multimodal checkpoints such as MedGemma
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype, device_map="cuda")
    return tok, model.eval(), str(dtype).replace("torch.", "")


@torch.inference_mode()
def generate(tok, model, prompts):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=C.EVAL_MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return [t.strip() for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)]


def run_condition(tok, model, records, lang, perm_idx, cache_path, batch_size=C.EVAL_BATCH):
    """Resumable: items already in cache_path are skipped. Returns rows in record order."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
    todo = [r for r in records if r["id"] not in done]
    for s in range(0, len(todo), batch_size):
        batch = todo[s:s + batch_size]
        texts = generate(tok, model, [chat_prompt(tok, r, r["perms"][perm_idx], lang) for r in batch])
        with open(cache_path, "a", encoding="utf-8") as f:
            for r, t in zip(batch, texts):
                perm = r["perms"][perm_idx]
                pos = parse_letter(t)
                choice = perm[pos] if pos >= 0 else -1
                row = {"id": r["id"], "raw": t, "pos": pos, "choice": choice,
                       "gold": r["answer_idx"], "correct": choice == r["answer_idx"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[r["id"]] = row
    return [done[r["id"]] for r in records]
