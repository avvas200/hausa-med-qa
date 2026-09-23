"""Resumable NLLB-200 translation with a JSONL cache."""
import json
from pathlib import Path

import torch


class Translator:
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.float16):
        from transformers import AutoModelForSeq2SeqLM
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device).eval()
        self._toks = {}
        self.truncated = 0

    def _tok(self, src_lang: str):
        from transformers import AutoTokenizer
        if src_lang not in self._toks:
            self._toks[src_lang] = AutoTokenizer.from_pretrained(self.model_name, src_lang=src_lang)
        return self._toks[src_lang]

    @torch.inference_mode()
    def translate(self, texts, src_lang, tgt_lang, batch_size=16, max_length=512):
        tok = self._tok(src_lang)
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out = [None] * len(texts)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = [texts[i] for i in idx]
            full = tok(batch, add_special_tokens=True)["input_ids"]
            self.truncated += sum(len(x) > max_length for x in full)
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(self.device)
            gen = self.model.generate(
                **enc, forced_bos_token_id=tgt_id, do_sample=False, num_beams=1,
                max_new_tokens=min(1024, 2 * enc["input_ids"].shape[1] + 16))
            for i, d in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = d.strip()
        return out


def load_cache(path) -> dict:
    cache = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[(row["key"], row["src"])] = row["hyp"]
    return cache


def translate_segments(translator, segs: dict, cache_path, src_lang, tgt_lang,
                       chunk=256, batch_size=16, max_length=512, progress=True):
    """Translate {key: text}. Cached (key, text) pairs are skipped, so the call
    can simply be re-run after a Colab disconnect."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    todo = [k for k, v in segs.items() if (k, v) not in cache]
    for s in range(0, len(todo), chunk):
        keys = todo[s:s + chunk]
        hyps = translator.translate([segs[k] for k in keys], src_lang, tgt_lang,
                                    batch_size=batch_size, max_length=max_length)
        with open(cache_path, "a", encoding="utf-8") as f:
            for k, h in zip(keys, hyps):
                f.write(json.dumps({"key": k, "src": segs[k], "hyp": h}, ensure_ascii=False) + "\n")
                cache[(k, segs[k])] = h
        if progress:
            print(f"  {min(s + chunk, len(todo))}/{len(todo)} new segments translated")
    return {k: cache[(k, v)] for k, v in segs.items()}
