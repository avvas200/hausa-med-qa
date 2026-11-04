"""Pre-flight for notebook 03a (H4 pilot gate). Run before booking a GPU.

Exercises every non-GPU path in src/dpo.py against a stub model: answer parsing,
the pilot subset, the resumable cache, pair construction and the gate report,
including the edge cases a near-chance model actually produces. The point is
that nothing here can crash or mislead AFTER a 40-minute generation run.

    PYTHONPATH=. python tests/preflight_h4.py

Three bugs it caught on first run, all fixed in src/dpo.py:
  * final_letter rejected a lower-case 'amsa: c', which a 3B model at
    temperature 0.8 emits often - it would have depressed the parse rate the
    gate exists to measure;
  * the bare-letter fallback matched a letter anywhere in the closing lines, so
    a Hausa sentence opening with the preposition 'A' was read as answer A,
    biasing every cue-less chain towards the very letter the study's baseline
    is about;
  * asking generate_pool for k=8 against a k=4 cache returned the k=4 rows in
    silence - exactly the move section 6 pre-registers when pairs < 200.
"""
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

from src import config as C
from src import dpo as DPO

ok = fail = 0


def check(name, got, want):
    global ok, fail
    if got == want:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL {name}: got {got!r}, want {want!r}")


print("0. the CoT prompt, and whether the parser can read what it asks for")
tmpl = C.DPO_COT_PROMPT
check("template takes a question", "{question}" in tmpl, True)
check("template takes options", "{options}" in tmpl, True)
# Sentence 1 is reused from the author-corrected EVAL_PROMPTS, so the two
# prompts address the model in the same register.
shared = C.EVAL_PROMPTS["ha"].split(". ")[0]
check("opening sentence matches the corrected eval prompt", tmpl.startswith(shared), True)
check("the approved letter parenthetical is reused", "(A, B, C ko D)" in tmpl, True)
# Naming a concrete example letter would bias generation towards it, on a task
# whose baseline is a majority letter.
check("no concrete example letter is shown",
      [L for L in C.LETTERS if f"Amsa: {L}" in tmpl], [])
check("the placeholder is used instead", "Amsa: X" in tmpl, True)

_rec = {"id": "demo", "question": "Wane magani ne ke hana shan taba?",
        "options": ["Bupropion", "Metformin", "Atenolol", "Warfarin"], "answer_idx": 0}
rendered = DPO.cot_prompt(_rec)
check("the question is rendered", _rec["question"] in rendered, True)
check("all four options are rendered",
      [L for L in C.LETTERS if f"{L}. " not in rendered], [])
# The whole point: a chain that obeys the instruction must parse.
for i, L in enumerate(C.LETTERS):
    obedient = f"Bupropion magani ne da ake amfani da shi.\nAmsa: {L}"
    check(f"a chain obeying the instruction parses ({L})", DPO.final_letter(obedient), i)
    check(f"...and in lower case ({L.lower()})",
          DPO.final_letter(obedient.replace(f"Amsa: {L}", f"amsa: {L.lower()}")), i)

print("1. final_letter on realistic chain endings")
L = C.LETTERS
cases = [
    # (chain, expected index, why it is here)
    ("Bayani...\nAmsa: C", 2, "the format the prompt asks for"),
    ("Bayani...\nAmsa: C.", 2, "trailing full stop"),
    ("Bayani...\namsa: c", 2, "lower case"),
    ("Bayani...\nAmsa ita ce B", 1, "copula variant in the regex"),
    ("Bayani...\nAmsa shine D", 3, "second copula variant"),
    ("Reasoning...\nAnswer: A", 0, "model answers in English"),
    ("Zabin A ba daidai ba ne, saboda haka.\nAmsa: C", 2,
     "an early letter mentioned while reasoning must not win"),
    ("Amsa: A\nSai dai a duba kuma.\nAmsa: D", 3, "LAST cue wins, not the first"),
    ("Da farko muna duba A.\nSaboda haka zabi na shi ne (B)", 1,
     "no cue word; bare letter in the tail"),
    ("Tunani mai tsawo...\nC", 2, "bare letter alone on the final line"),
    ("Babu wata amsa da ta dace a nan.", -1, "no letter at all -> unparseable"),
    # Lower-case regressions. Accepting a bare lower-case letter after the cue
    # turned ordinary Hausa into answer A: "a", "da" and "ba" are among the
    # commonest words in the language, so the rule manufactured the very
    # letter bias this study measures.
    ("Babu amsa a nan", -1, "cue + the preposition 'a' is not an answer"),
    ("Ba zan iya ba da amsa a kan wannan tambaya ba", -1, "the same in a full sentence"),
    ("amsa a cikin rubutun", -1, "cue + 'a cikin' is not an answer"),
    ("Bayani...\namsa: c", 2, "lower case IS an answer after a colon"),
    ("Tunani...\nc", 2, "or alone on the closing line"),
    ("Bayani...\nJibu la sahihi ni C.", 2, "Swahili cue, caught by the tail rule"),
    ("Ketamine ne.\nJibu: X = A", 0, "Swahili, trailing letter"),
    ("A cikin wannan yanayin, Aspirin shine magani.", -1,
     "'A' opens a sentence and 'Aspirin' starts with A: must NOT be read as a letter"),
]
for chain, want, why in cases:
    got = DPO.final_letter(chain)
    check(f"final_letter({why})", got, want)

print("2. hausa_stem_items and pilot_subset")
recs = [{"id": f"i{n:03d}", "question": f"Tambaya {n}?",
         "options": [f"zabi {n}{x}" for x in "abcd"], "answer_idx": n % 4}
        for n in range(500)]
# The REAL status vocabulary, as written by translate.translate_segments and by
# notebook 01b cell 17 -- not an invented one. The pool's English stems carry
# "fallback_en_auto", which an exact match on "fallback_en" misses entirely.
VOCAB = ["translated", "retried", "passthrough", "fallback_en",
         "fallback_en_auto", "fallback_en_reviewed"]
seg = {}
english = set()
for n in range(500):
    if n % 7 == 0:
        s = "fallback_en_auto"
    elif n % 11 == 0:
        s = "fallback_en"
    elif n % 23 == 0:
        s = "fallback_en_reviewed"
    elif n % 5 == 0:
        s = "passthrough"
    else:
        s = "translated"
    seg[f"i{n:03d}|q"] = s
    if s.startswith("fallback_en"):
        english.add(f"i{n:03d}")
hausa = DPO.hausa_stem_items(recs, seg)
check("every English-stem status is excluded", len(hausa), 500 - len(english))
check("no excluded item survives",
      [r["id"] for r in hausa if r["id"] in english], [])
check("passthrough stems are kept (a code is not a fallback)",
      any(seg[f"{r['id']}|q"] == "passthrough" for r in hausa), True)
for s in VOCAB:
    one = [{"id": "z", "question": "q", "options": list("abcd"), "answer_idx": 0}]
    kept = len(DPO.hausa_stem_items(one, {"z|q": s}))
    check(f"status {s!r} -> {'dropped' if s.startswith('fallback_en') else 'kept'}",
          kept, 0 if s.startswith("fallback_en") else 1)
try:
    DPO.hausa_stem_items([{"id": "nope", "question": "q", "options": list("abcd"),
                           "answer_idx": 0}], {})
    check("a missing stem status must raise", "no error", "KeyError")
except KeyError as e:
    check("a missing stem status raises rather than silently including it",
          "segment_status.json" in str(e), True)
p1 = DPO.pilot_subset(hausa)
p2 = DPO.pilot_subset(hausa)
check("pilot size", len(p1), C.DPO_PILOT_N)
check("pilot is deterministic", [r["id"] for r in p1], [r["id"] for r in p2])
check("pilot is sorted by id", [r["id"] for r in p1], sorted(r["id"] for r in p1))
check("pilot items come from the hausa subset",
      all(r["id"] in {h["id"] for h in hausa} for r in p1), True)

print("3. generate_pool: caching and resume")


class StubTok:
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return msgs[0]["content"]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return self

    def to(self, device):
        return self

    def __getitem__(self, key):
        return _Shape()

    def decode(self, seq, skip_special_tokens=True):
        return seq


class _Shape:
    shape = (1, 0)


class StubModel:
    """Deterministic 'generation': item n gets n%5 correct chains out of k."""
    device = "cpu"

    def __init__(self, recs):
        self.by_q = {r["question"]: r for r in recs}
        self.calls = 0

    def generate(self, **kw):
        self.calls += 1
        rec = self._rec
        n_right = int(rec["id"][1:]) % 5
        gold = C.LETTERS[rec["answer_idx"]]
        wrong = [x for x in C.LETTERS if x != gold][0]
        out = []
        for j in range(kw["num_return_sequences"]):
            letter = gold if j < n_right else wrong
            out.append(f"Tunani kan {rec['id']}...\nAmsa: {letter}")
        return out


def patched_sample(tok, model, rec, k=C.DPO_K, seed=None):
    model._rec = rec
    return model.generate(num_return_sequences=k)


DPO.sample_completions = patched_sample

tmp = Path(tempfile.mkdtemp())
cache = tmp / "dpo_cot_k4.jsonl"
pilot = p1[:20]
model = StubModel(pilot)
rows = DPO.generate_pool(StubTok(), model, pilot, cache, k=4)
check("rows returned", len(rows), 20)
check("one generate call per item", model.calls, 20)
check("rows are in record order", [r["id"] for r in rows], [r["id"] for r in pilot])

model2 = StubModel(pilot)
rows2 = DPO.generate_pool(StubTok(), model2, pilot, cache, k=4)
check("resume regenerates nothing", model2.calls, 0)
check("resume returns the same rows", [r["id"] for r in rows2], [r["id"] for r in rows])

# half-written cache, as a dropped Colab session leaves it
lines = cache.read_text(encoding="utf-8").splitlines()[:12]
cache.write_text("\n".join(lines) + "\n", encoding="utf-8")
model3 = StubModel(pilot)
DPO.generate_pool(StubTok(), model3, pilot, cache, k=4)
check("resume regenerates only the missing items", model3.calls, 8)

print("4. the k-mismatch trap (section 6 raises k to 8 if pairs < 200)")
model4 = StubModel(pilot)
try:
    DPO.generate_pool(StubTok(), model4, pilot, cache, k=8)
    check("k=8 against a k=4 cache must raise", "no error", "ValueError")
except ValueError as e:
    check("k=8 against a k=4 cache raises", "k=[4]" in str(e).replace(" ", ""), True)
    check("nothing was regenerated before raising", model4.calls, 0)
    print(f"   {e}")
# a k-specific cache path is the documented way out
model5 = StubModel(pilot)
rows5 = DPO.generate_pool(StubTok(), model5, pilot, tmp / "dpo_cot_k8.jsonl", k=8)
check("a separate k=8 cache works", sorted({r["k"] for r in rows5}), [8])
check("and really generates 8 chains", sorted({len(r["chains"]) for r in rows5}), [8])

print("5. build_pairs and gate_report")
pairs = DPO.build_pairs(rows, pilot)
mixed = [r for r in rows if 0 < sum(r["correct"]) < r["k"]]
check("one pair per mixed item", len(pairs), len(mixed))
check("no pair from an all-correct or all-wrong item",
      all(0 < p["n_correct"] < 4 for p in pairs), True)
for p in pairs:
    assert p["chosen"] != p["rejected"], p["id"]
    assert p["chosen_letter"] == p["gold"], p["id"]
    assert p["rejected_letter"] != p["gold"], p["id"]
ok += 1

rep = DPO.gate_report(rows, pairs, n_full=430)
check("parse rate on clean chains", rep["parse_rate"], 1.0)
check("buckets cover 0..k", sorted(rep["n_correct_per_item"]), ["0", "1", "2", "3", "4"])
check("buckets sum to n_items", sum(rep["n_correct_per_item"].values()), len(rows))
check("projection uses the full pool",
      rep["projected_pairs_full_pool"], int(round(rep["pair_yield_rate"] * 430)))

print("6. gate_report edge cases")


def synth(n, k, n_right, parsed=True):
    return [{"id": f"x{i}", "gold": 0, "k": k,
             "chains": ["..."] * k,
             "picks": [0] * n_right + ([1] * (k - n_right) if parsed
                                       else [-1] * (k - n_right)),
             "correct": [True] * n_right + [False] * (k - n_right)}
            for i in range(n)]


for label, rows_e, npairs in [
    ("all chains wrong (acc 0)", synth(10, 4, 0), 0),
    ("all chains right (acc 1)", synth(10, 4, 4), 0),
    ("nothing parseable", synth(10, 4, 0, parsed=False), 0),
    ("at chance (acc 0.25)", synth(10, 4, 1), 10),
    ("empty run", [], 0),
]:
    try:
        r = DPO.gate_report(rows_e, DPO.build_pairs(rows_e, [
            {"id": f"x{i}", "question": "q", "options": list("abcd"), "answer_idx": 0}
            for i in range(len(rows_e))]) if rows_e else [], n_full=430)
        print(f"   {label:26s} acc={r['accuracy_all_samples']} "
              f"skill={r['estimated_skill_rate']} "
              f"luck={r['estimated_luck_share_of_chosen']} pairs={r['pairs']}")
        ok += 1
    except Exception as e:
        fail += 1
        print(f"   FAIL {label}: {type(e).__name__}: {e}")

print("6b. failure-mode classification")
# Fragments taken from results/h4_pilot_chains.csv, so the classifier is tested
# against what the model actually produced rather than invented text.
LOOP = "taka shi ne kwata ya kamata shi, " * 12
check("a loop scores high on repetition", DPO.repetition_ratio(LOOP) > 0.5, True)
check("ordinary prose does not",
      DPO.repetition_ratio("Angiotensin-converting enzyme inhibitors sune "
                           "maganin da ya dace ga hauhawar jini a cikin koda.") < 0.5,
      True)
check("a short string is not scored", DPO.repetition_ratio("Amsa: B"), 0.0)

cases = [
    ("Bayani...\nJibu la sahihi ni C.", "code_switch", "Swahili 'jibu'"),
    ("Ketamine ne.\nhivyo X = A", "code_switch", "Swahili 'hivyo'"),
    (LOOP, "repetition", "a loop outranks every other label"),
    ("Tambaya: Lactogen: Harafin amsarka D", "prompt_echo", "echoes the template"),
    ("Maganin cututtukan myeloid ana amsa A. Imatinib.", "other", "a clean short answer"),
    ("   ", "empty", "whitespace"),
]
for text, want, why in cases:
    check(f"classify ({why})", DPO.classify_chain(text)["primary"], want)

# Words shared by Hausa and Swahili must not trip the marker list, or ordinary
# Hausa counts as code-switching and the headline number is wrong.
for hausa in ("Amsar da ta dace ita ce B", "Zabin ya kamata ya zama C",
              "Wannan shine amsar da yake daidai: D"):
    check(f"plain Hausa is not flagged ({hausa[:22]}...)",
          DPO.classify_chain(hausa)["swahili"], False)

check("truncation is not claimed without a tokenizer",
      DPO.classify_chain("a" * 5000)["truncated"], False)


class LenTok:
    """Stands in for a real tokenizer: one token per whitespace-separated word."""

    def __call__(self, t, add_special_tokens=False):
        return {"input_ids": list(range(len(t.split())))}


check("truncation is detected with one",
      DPO.classify_chain(" ".join(["w"] * (C.DPO_MAX_NEW_TOKENS + 5)),
                         tok=LenTok())["truncated"], True)

fm_rows = [{"id": "a", "gold": 0, "k": 2, "chains": ["Amsa: A", "Jibu ni B"],
            "picks": [0, 1], "correct": [True, False]},
           {"id": "b", "gold": 0, "k": 2, "chains": [LOOP, "   "],
            "picks": [-1, -1], "correct": [False, False]}]
fm = DPO.failure_modes(fm_rows)
check("counts every chain", fm["n_chains"], 4)
check("parsed + unparsed covers them", fm["n_parsed"] + fm["n_unparsed"], 4)
check("primary labels partition the unparsed group",
      sum(v["n"] for v in fm["unparsed_only"]["primary"].values()),
      fm["n_unparsed"])
check("the caveat travels with the numbers", "lower bound" in fm["caveat"], True)
check("an empty pool does not crash", DPO.failure_modes([])["n_chains"], 0)

print("7. inspection_sample and write_json")
check("inspection_sample caps at the pair count", len(DPO.inspection_sample(pairs, n=10)),
      min(10, len(pairs)))
check("inspection_sample survives n > len(pairs)",
      len(DPO.inspection_sample(pairs[:3], n=10)), 3)
h = DPO.write_json(rep, tmp / "sub" / "gate.json")
check("write_json creates parents and hashes", isinstance(h, str) and len(h) == 64, True)
check("write_json round-trips",
      json.load(open(tmp / "sub" / "gate.json", encoding="utf-8"))["pairs"], rep["pairs"])

shutil.rmtree(tmp, ignore_errors=True)
print()
print(f"{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
