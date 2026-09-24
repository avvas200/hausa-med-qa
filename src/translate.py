"""Resumable NLLB-200 translation with pre-registered safeguards.

Each segment gets exactly one status:
  passthrough  numbers, units-only strings, acronyms/codes -> copied verbatim
  copied       caller asked to copy (e.g. back-translating a passthrough)
  translated   main decoding produced a non-degenerate output
  retried      main output degenerate; retry decoding succeeded
  fallback_en  both attempts degenerate -> English source kept (loanword rule)
"""
import json
import re
from collections import Counter
from pathlib import Path

import torch

from . import config as C

_alpha = re.compile(r"[A-Za-z]+")


def is_passthrough(text: str) -> bool:
    """True for strings that should not be translated: no letters (numbers,
    symbols), or every alphabetic token is a code (ALLCAPS, mixed case like
    IgA/mRNA, or at most 2 letters)."""
    toks = _alpha.findall(text)
    if not toks:
        return True

    def is_code(t):
        return (t.isupper() or len(t) <= 2
                or (t != t.lower() and t != t.capitalize()))
    return all(is_code(t) for t in toks)


def is_degenerate(src: str, hyp: str) -> bool:
    words = hyp.split()
    if not words:
        return True
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 3:
            return True
    if len(words) >= 6 and max(Counter(words).values()) / len(words) > 0.4:
        return True
    return len(hyp) > 4 * len(src) + 20


class Translator:
    def __init__(self, model_name: str = C.NLLB_MODEL, device: str = "cuda",
                 dtype=torch.bfloat16):
        from transformers import AutoModelForSeq2SeqLM
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=dtype).to(device).eval()
        self._toks = {}
        self.truncated = 0

    def _tok(self, src_lang):
        from transformers import AutoTokenizer
        if src_lang not in self._toks:
            self._toks[src_lang] = AutoTokenizer.from_pretrained(self.model_name, src_lang=src_lang)
        return self._toks[src_lang]

    @torch.inference_mode()
    def translate(self, texts, src_lang, tgt_lang, gen_kwargs,
                  batch_size=C.TRANSLATION_BATCH, max_length=C.TRANSLATION_MAX_LEN):
        tok = self._tok(src_lang)
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out = [None] * len(texts)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = [texts[i] for i in idx]
            self.truncated += sum(len(x) > max_length for x in tok(batch)["input_ids"])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(self.device)
            max_new = int(C.MAX_NEW_FACTOR * enc["input_ids"].shape[1]) + C.MAX_NEW_BIAS
            gen = self.model.generate(**enc, forced_bos_token_id=tgt_id,
                                      max_new_tokens=max_new, **gen_kwargs)
            for i, d in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = d.strip()
        return out


def _load(path):
    cache = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[(r["key"], r["src"])] = (r["hyp"], r["status"])
    return cache


def translate_segments(tr, segs: dict, cache_path, src_lang, tgt_lang,
                       copy_keys=(), chunk=256, progress=True):
    """Translate {key: text}; returns ({key: hyp}, {key: status}).
    Re-running after a disconnect skips everything already cached."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load(cache_path)
    copy_keys = set(copy_keys)

    def write(rows):
        with open(cache_path, "a", encoding="utf-8") as f:
            for k, hyp, status in rows:
                f.write(json.dumps({"key": k, "src": segs[k], "hyp": hyp, "status": status},
                                   ensure_ascii=False) + "\n")
                cache[(k, segs[k])] = (hyp, status)

    # A copy request overrides an earlier cached translation of the same text
    # (e.g. a Hausa segment identical to the English that was later reviewed as
    # "english": it must be copied back verbatim, not machine-translated).
    todo = [k for k, v in segs.items()
            if (k, v) not in cache or (k in copy_keys and cache[(k, v)][1] != "copied")]
    fixed = [(k, segs[k], "copied" if k in copy_keys else "passthrough")
             for k in todo if k in copy_keys or is_passthrough(segs[k])]
    write(fixed)
    todo = [k for k in todo if (k, segs[k]) not in cache]

    for s in range(0, len(todo), chunk):
        keys = todo[s:s + chunk]
        hyps = tr.translate([segs[k] for k in keys], src_lang, tgt_lang, C.GEN_MAIN)
        bad = [k for k, h in zip(keys, hyps) if is_degenerate(segs[k], h)]
        rows = [(k, h, "translated") for k, h in zip(keys, hyps) if k not in set(bad)]
        if bad:
            retry = tr.translate([segs[k] for k in bad], src_lang, tgt_lang, C.GEN_RETRY)
            for k, h in zip(bad, retry):
                rows.append((k, h, "retried") if not is_degenerate(segs[k], h)
                            else (k, segs[k], "fallback_en"))
        write(rows)
        if progress:
            print(f"  {min(s + chunk, len(todo))}/{len(todo)} segments")

    hyps = {k: cache[(k, v)][0] for k, v in segs.items()}
    status = {k: cache[(k, v)][1] for k, v in segs.items()}
    return hyps, status
