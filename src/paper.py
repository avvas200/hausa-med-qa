"""Emit paper/numbers.tex from results/, so the manuscript never holds a
hand-typed number.

Every figure in the paper is a LaTeX macro defined here and expanded at compile
time. If a result changes, regenerate and the paper changes with it; a number
that is not in results/ cannot appear in the text by accident.

`check_macros` closes the other half of that loop: it reads main.tex, collects
every macro the text uses, and fails if one is not defined. A typo in a macro
name otherwise renders as nothing at all, which is the worst failure mode a
results section can have.
"""
import json
import re
from pathlib import Path

from . import config as C
from . import figures as F

MODEL_MACRO = {"llama": "Llama", "qwen": "Qwen", "medgemma": "MedGemma"}


def _f3(x):
    return f"{x:.3f}"


def _f2(x):
    return f"{x:.2f}"


def _pp(x):
    return f"{100 * x:.1f}"


def macros(res) -> dict:
    """{macro name: expansion}. Names are letters only -- TeX forbids digits."""
    b = res["baseline"]
    acc, pr, p = b["accuracy"], b["parse_rate"], b["primary"]
    lp = b["accuracy_letter_prob"]
    m = {}

    m["NEval"] = str(p["n_items"])
    m["Chance"] = "0.25"
    base = b["majority_letter_baseline"]["p0"]
    m["BaselineP"] = _f3(base)

    for k, M in MODEL_MACRO.items():
        m[f"En{M}"] = _f3(acc[k]["en_p0"])
        m[f"Ha{M}"] = _f3(acc[k]["ha_p0"])
        m[f"HaEn{M}"] = _f3(acc[k]["ha2en_p0"])
        m[f"Gap{M}"] = _pp(acc[k]["en_p0"] - acc[k]["ha_p0"])
        ci = p["H1"][k]["ci"]
        m[f"GapLo{M}"] = _pp(ci["ci_low"])
        m[f"GapHi{M}"] = _pp(ci["ci_high"])
        m[f"ParseHa{M}"] = _f2(pr[k]["ha_p0"])
        m[f"ParseEn{M}"] = _f2(pr[k]["en_p0"])
        m[f"LpHa{M}"] = _f3(lp[k]["ha_p0"])
        rec = p["H2"][k]["recovery"]
        m[f"Rec{M}"] = _f2(rec["point"])
        m[f"RecLo{M}"] = _f2(rec["ci_low"])
        m[f"RecHi{M}"] = _f2(rec["ci_high"])
        m[f"RecP{M}"] = f"{p['H2'][k]['p_holm']:.3f}"
        h3 = p["H3"][k]
        m[f"ConsEn{M}"] = _f3(h3["cons_en"])
        m[f"ConsHa{M}"] = _f3(h3["cons_ha"])
        m[f"ConsGap{M}"] = _f3(h3["ci"]["point"])
        # Signed distance from the majority-letter baseline, in points. The
        # sign is what the paper's central claim turns on, so it is computed
        # here rather than asserted in the prose.
        m[f"Margin{M}"] = _pp(acc[k]["ha_p0"] - base)
    m["NBelowBaseline"] = str(sum(acc[k]["ha_p0"] <= base for k in MODEL_MACRO))
    m["NModels"] = str(len(MODEL_MACRO))

    # E1
    e = res["e1"]["primary_pooled"]
    m["ENRows"] = str(e["n"])
    for code, name in [("term", "Term"), ("translation", "Trans"),
                       ("format", "Format"), ("reasoning", "Reason")]:
        v = e["codes"][code]
        m[f"E{name}"] = _f2(v["prop"])
        m[f"E{name}N"] = str(v["n"])
        m[f"E{name}Lo"] = _f2(v["ci95"][0])
        m[f"E{name}Hi"] = _f2(v["ci95"][1])
    loc = e["text_error_location"]
    m["ELocStem"] = str(loc.get("stem", 0))
    m["ELocChosen"] = str(loc.get("chosen", 0))
    m["ELocGold"] = str(loc.get("gold", 0))
    m["ETextTotal"] = str(sum(loc.values()))
    m["EUntranslated"] = str(
        res["e1"]["untranslated_options"]["rows_with_any_identical_option"])

    # Human validation of the translations
    v = res["validation"]
    m["ValN"] = str(C.N_HUMAN_VALIDATION)
    m["ValAdequacyOne"] = _f2(v["rater1"]["mean_adequacy"])
    m["ValAdequacyTwo"] = _f2(v["rater2"]["mean_adequacy"])
    m["ValGeFourOne"] = _f2(v["rater1"]["prop_adequacy_ge4"])
    m["ValGeFourTwo"] = _f2(v["rater2"]["prop_adequacy_ge4"])
    # Percentages read better than proportions in prose ("98.7\%" not "0.99").
    m["ValGeFourOnePct"] = _pp(v["rater1"]["prop_adequacy_ge4"])
    m["ValGeFourTwoPct"] = _pp(v["rater2"]["prop_adequacy_ge4"])
    m["ValMinGeFourPct"] = _pp(min(v["rater1"]["prop_adequacy_ge4"],
                                   v["rater2"]["prop_adequacy_ge4"]))
    m["ValKappa"] = _f2(v["kappa_quadratic"])
    m["ValFlagged"] = str(v["items_flagged_by_any_rater"])

    # Sensitivity analyses: the n each one leaves, and the largest change it
    # makes to any model's H1 gap. "Unchanged" is a claim, so give it a number.
    for key, name in [("sens_a_no_rater_flagged", "SensA"),
                      ("sens_b_no_english_options", "SensB")]:
        blk = b.get(key)
        if not blk:
            continue
        m[f"{name}N"] = str(blk["n_items"])
        worst = max(abs(blk["H1"][k]["diff"] - p["H1"][k]["diff"])
                    for k in MODEL_MACRO)
        m[f"{name}Shift"] = _pp(worst)

    # Corpus construction
    man = res["manifest"]
    fl = man["filter_log"]["validation"]
    m["PoolStart"] = str(fl["start"])
    m["PoolRemaining"] = str(fl["remaining"])
    m["ExclDental"] = str(fl["excluded_subject"])
    m["ExclMultiAnswer"] = str(fl["multi_answer"])
    m["FallbackItems"] = str(man["items_with_fallback"])

    # H4, only once the gate has run
    g = res.get("h4_gate")
    if g:
        from . import e1 as E1  # wilson lives there; avoid a circular import
        m["GateParse"] = _f3(g["parse_rate"])
        m["GateAcc"] = _f3(g["accuracy_all_samples"])
        m["GatePairs"] = str(g["pairs"])
        m["GateProjected"] = str(g["projected_pairs_full_pool"])
        m["GateLuck"] = _f2(g["estimated_luck_share_of_chosen"])
        m["GateItems"] = str(g["n_items"])
        m["GateK"] = str(g["k"])
        m["GateSamples"] = str(g["n_samples"])
        m["GateThreshold"] = str(g["min_pairs_threshold"])
        m["GateUnparsedPct"] = _pp(1 - g["parse_rate"])
        # Accuracy among chains that do state a letter, with an interval. The
        # point estimate sits under chance, but the interval covers it, so the
        # paper must say "indistinguishable from chance", not "below" it.
        n_parsed = round(g["parse_rate"] * g["n_samples"])
        k_corr = round(g["accuracy_parsed_only"] * n_parsed)
        lo, hi = E1.wilson(k_corr, n_parsed)
        m["GateParsedN"] = str(n_parsed)
        m["GateAccParsed"] = _f3(g["accuracy_parsed_only"])
        m["GateAccParsedLo"] = _f3(lo)
        m["GateAccParsedHi"] = _f3(hi)
        zero = g["n_correct_per_item"].get("0", 0)
        m["GateZeroItems"] = str(zero)
        m["GateZeroPct"] = _pp(zero / g["n_items"]) if g["n_items"] else "0.0"

    # Why the chains fail. Reported over the unparseable chains, which is the
    # population the parse rate is about, with the overlapping flags beside the
    # mutually exclusive primary label.
    fm = res.get("h4_failure_modes")
    if fm and fm.get("n_chains"):
        m["FailChains"] = str(fm["n_chains"])
        m["FailUnparsed"] = str(fm["n_unparsed"])
        u = fm["unparsed_only"]
        names = {"code_switch": "CodeSwitch", "repetition": "Repetition",
                 "prompt_echo": "PromptEcho", "truncated": "Truncated",
                 "empty": "Empty", "other": "Other"}
        for key, M in names.items():
            v = u["primary"].get(key, {"n": 0, "prop": 0.0})
            m[f"FailU{M}N"] = str(v["n"])
            m[f"FailU{M}"] = _pp(v["prop"])
        flags = (("swahili", "Swahili"), ("repetition", "Repetition"),
                 ("truncated", "Truncated"), ("prompt_echo", "PromptEcho"))
        for key, M in flags:
            v = u["any_of"].get(key, {"n": 0, "prop": 0.0})
            m[f"FailAny{M}N"] = str(v["n"])
            m[f"FailAny{M}"] = _pp(v["prop"])
        # The same flags over every chain, parsed or not: a chain that echoes
        # the prompt and still emits a letter is as diagnostic as one that
        # does not.
        a = fm["all_chains"]
        for key, M in flags:
            v = a["any_of"].get(key, {"n": 0, "prop": 0.0})
            m[f"FailAll{M}N"] = str(v["n"])
            m[f"FailAll{M}"] = _pp(v["prop"])
        for key, M in (("prompt_echo", "PromptEcho"), ("code_switch", "CodeSwitch")):
            v = fm["parsed_only"]["any_of"].get(
                "swahili" if key == "code_switch" else key, {"n": 0, "prop": 0.0})
            m[f"FailParsed{M}N"] = str(v["n"])
            m[f"FailParsed{M}"] = _pp(v["prop"])
        if fm.get("tokens"):
            m["FailMedianTokens"] = str(fm["tokens"]["median"])
            m["FailAtBudget"] = str(fm["tokens"]["at_budget"])
            m["FailTokenBudget"] = str(fm["tokens"]["max_new"])
    return m


def write_numbers(res, path="paper/numbers.tex") -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = macros(res)
    lines = ["% Generated by src/paper.py from results/. Do not edit by hand.",
             "% Regenerate:  python -c \"from src import paper, figures as F; "
             "paper.write_numbers(F.load())\"",
             ""]
    for k in sorted(mm):
        lines.append(f"\\newcommand{{\\{k}}}{{{mm[k]}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


_USE = re.compile(r"\\([A-Z][A-Za-z]*)\b")
# Macros TeX and the document class already provide, which are not ours to define.
_TEX_BUILTINS = {
    "LaTeX", "TeX", "Large", "LARGE", "Huge", "Big", "Bigg", "Alph", "Roman",
    "Re", "Im", "Pr", "Delta", "Sigma", "Omega", "Gamma", "Lambda", "Theta",
    "Phi", "Psi", "Xi", "Pi", "Upsilon",
}


def check_macros(tex="paper/main.tex", numbers="paper/numbers.tex"):
    """Every \\Macro the paper uses must be defined. An undefined one renders as
    nothing, so a results section can silently lose a number."""
    body = Path(tex).read_text(encoding="utf-8")
    # Strip comments, but not an escaped \% -- "95\% CI \ETermLo{}" is text, and
    # treating its percent as a comment would hide every macro after it on the
    # line, which is exactly the kind of silent miss this function exists to catch.
    body = re.sub(r"(?m)(?<!\\)%.*$", "", body)
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                             Path(numbers).read_text(encoding="utf-8")))
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", body))
    used = {u for u in _USE.findall(body) if u not in _TEX_BUILTINS}
    missing = sorted(used - defined)
    unused = sorted(defined - used)
    return {"used": len(used), "defined": len(defined),
            "missing": missing, "unused": unused}


def check_citations(tex="paper/main.tex", bib="paper/refs.bib"):
    """Every \\cite key must exist in the .bib. A missing one renders as [?] and
    is easy to miss in a draft; an unused entry is harmless but worth seeing."""
    body = re.sub(r"(?m)(?<!\\)%.*$", "", Path(tex).read_text(encoding="utf-8"))
    cited = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]*)\}", body):
        cited |= {k.strip() for k in group.split(",") if k.strip()}
    # Only real entries, not the commented-out TODO block.
    bib_body = re.sub(r"(?m)^\s*%.*$", "",
                      Path(bib).read_text(encoding="utf-8"))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib_body))
    return {"cited": len(cited), "in_bib": len(defined),
            "missing": sorted(cited - defined),
            "uncited": sorted(defined - cited)}


def check_all(tex="paper/main.tex", numbers="paper/numbers.tex",
              bib="paper/refs.bib"):
    """Both checks plus the open \\todo count, as one line for the notebook."""
    body = Path(tex).read_text(encoding="utf-8")
    todos = len(re.findall(r"\\todo\{", re.sub(r"(?m)(?<!\\)%.*$", "", body)))
    return {"macros": check_macros(tex, numbers),
            "citations": check_citations(tex, bib),
            "open_todos": todos}


def build_submission(out_dir="paper/submission", figures="results/figures"):
    """Assemble a flat folder ready to upload to arXiv or Overleaf.

    arXiv wants the source as one directory with no nesting, while the
    repository keeps figures under results/. Rather than ask anyone to retype
    five ``includegraphics`` paths at the last minute -- the kind of edit that
    silently drops a figure -- main.tex uses bare filenames and a
    ``graphicspath`` that lists both layouts. This copies the pieces into one
    place so either works.
    """
    import shutil

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("main.tex", "numbers.tex", "refs.bib"):
        src = Path("paper") / name
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing")
        shutil.copy2(src, out / name)
        copied.append(name)
    figs = sorted(Path(figures).glob("fig*.pdf"))
    if not figs:
        raise FileNotFoundError(
            f"no fig*.pdf in {figures} -- run notebook 04 first")
    for f in figs:
        shutil.copy2(f, out / f.name)
        copied.append(f.name)

    # Every figure the text asks for must have arrived.
    wanted = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                            (out / "main.tex").read_text(encoding="utf-8")))
    missing = sorted(w for w in wanted if not (out / Path(w).name).exists())
    if missing:
        raise FileNotFoundError(f"main.tex references missing figures: {missing}")
    return {"dir": str(out), "files": copied, "n": len(copied)}
