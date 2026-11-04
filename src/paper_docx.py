"""Turn paper/main.tex into a structured document model, for a .docx build.

The Word version is generated from the LaTeX, never maintained beside it.
Keeping two hand-written copies of a paper whose numbers come from results/
would defeat the point of generating the numbers at all: the .docx would drift
the first time a result changed.

This module does the LaTeX half -- expand the macros, resolve the citations,
flatten the markup into runs -- and emits JSON. `tools/build_docx.js` turns
that JSON into the file, because docx-js is the reliable way to write one.

It handles the constructs this manuscript actually uses and raises on anything
it does not recognise rather than dropping it silently, which is how a
converter loses a paragraph without anyone noticing.
"""
import json
import re
from pathlib import Path


# ---------------------------------------------------------------- macros

def load_macros(path="paper/numbers.tex") -> dict:
    src = Path(path).read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", src))


def expand(text: str, macros: dict) -> str:
    """Replace \\Macro{} and \\Macro with its value, longest name first so that
    \\GapLo does not eat the start of \\GapLoLlama."""
    for name in sorted(macros, key=len, reverse=True):
        text = text.replace("\\" + name + "{}", macros[name])
        text = re.sub(r"\\" + name + r"(?![A-Za-z])", macros[name], text)
    return text


# ------------------------------------------------------------ references

_BIB_ENTRY = re.compile(r"@\w+\{([^,]+),(.*?)\n\}", re.S)


def load_bib(path="paper/refs.bib") -> dict:
    src = re.sub(r"(?m)^\s*%.*$", "", Path(path).read_text(encoding="utf-8"))
    out = {}
    for key, body in _BIB_ENTRY.findall(src):
        fields = {}
        for fname, braced, quoted in re.findall(
                r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\")", body):
            fields[fname.lower()] = _tidy(braced or quoted)
        out[key.strip()] = fields
    return out


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("\\&", "&").replace("\\_", "_").replace("\\%", "%")
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\{\\[`'\"^~=.]([A-Za-z])\}", r"\1", s)   # {\`e} -> e
    s = re.sub(r"\\[`'\"^~=.]\{?([A-Za-z])\}?", r"\1", s)
    s = re.sub(r"\{\\c\{([A-Za-z])\}\}|\\c\{([A-Za-z])\}", r"\1\2", s)
    s = s.replace("{", "").replace("}", "")
    # BibTeX's "and" separators are syntax, not prose.
    s = re.sub(r"\s+and\s+others\b", " et al.", s)
    s = re.sub(r"\s+and\s+", ", ", s)
    return s.replace("--", "–")


def format_reference(key: str, f: dict) -> str:
    bits = []
    if f.get("author"):
        bits.append(f["author"])
    if f.get("year"):
        bits.append(f"({f['year']})")
    if f.get("title"):
        bits.append(f["title"] + ".")
    venue = f.get("booktitle") or f.get("journal") or f.get("publisher")
    if venue:
        tail = venue
        if f.get("volume"):
            tail += f", {f['volume']}"
        if f.get("pages"):
            tail += f", pp. {f['pages']}"
        bits.append(tail + ".")
    if f.get("note"):
        bits.append(f["note"])
    return " ".join(bits) if bits else key


# ------------------------------------------------------------ inline text

_REPL = [
    (r"\\todo\{[^}]*\}", ""),
    (r"``", "\u201c"), (r"''", "\u201d"),
    (r"\\ldots", "\u2026"),
    (r"\\kappa", "\u03ba"),
    (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\times", "\u00d7"),
    (r"\\linewidth", ""),
    (r"\\%", "%"), (r"\\&", "&"), (r"\\_", "_"), (r"\\#", "#"), (r"\\\$", "$"),
    (r"\\,", " "), (r"\\ ", " "),
    (r"~", " "),
]


def inline(text: str, bib_order: list, bib: dict):
    """Flatten one paragraph of LaTeX into [{text, bold, italic, mono}] runs."""
    t = text
    for pat, rep in _REPL:
        t = re.sub(pat, rep, t)

    def cite(m):
        nums = []
        for key in (k.strip() for k in m.group(1).split(",")):
            if key not in bib_order:
                bib_order.append(key)
            nums.append(str(bib_order.index(key) + 1))
        return "[" + ", ".join(nums) + "]"

    t = re.sub(r"\\cite[a-zA-Z]*\{([^}]*)\}", cite, t)
    t = re.sub(r"\\(?:ref|label)\{(?:fig|tab|sec):([^}]*)\}",
               lambda m: m.group(1).replace("_", " "), t)
    t = t.replace("$", "")
    t = re.sub(r"---", "\u2014", t)
    t = re.sub(r"--", "\u2013", t)

    runs, pos = [], 0
    pattern = re.compile(r"\\(textbf|emph|textit|texttt|url)\{([^{}]*)\}")
    for m in pattern.finditer(t):
        if m.start() > pos:
            runs.append({"text": t[pos:m.start()]})
        kind, body = m.group(1), m.group(2)
        run = {"text": body}
        if kind == "textbf":
            run["bold"] = True
        elif kind in ("emph", "textit"):
            run["italic"] = True
        else:
            run["mono"] = True
        runs.append(run)
        pos = m.end()
    if pos < len(t):
        runs.append({"text": t[pos:]})

    for r in runs:
        r["text"] = re.sub(r"[ \t]+", " ", r["text"].replace("\n", " "))
    runs = [r for r in runs if r["text"]]
    leftover = [m for r in runs for m in re.findall(r"\\[A-Za-z]+", r["text"])]
    if leftover:
        raise ValueError(f"unhandled LaTeX command(s) {sorted(set(leftover))} "
                         f"in: {text[:120]!r}")
    return runs


# --------------------------------------------------------------- blocks

def _table(block: str, bib_order, bib):
    rows = []
    body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                     block, re.S)
    for line in body.group(1).split("\\\\"):
        line = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", line).strip()
        if not line:
            continue
        rows.append([inline(c.strip(), bib_order, bib) for c in line.split("&")])
    cap = re.search(r"\\caption\{(.*?)\}\s*\n", block, re.S)
    return {"type": "table", "rows": rows,
            "caption": inline(cap.group(1), bib_order, bib) if cap else []}


def _figure(block: str, bib_order, bib, fig_dir):
    img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", block)
    cap = re.search(r"\\caption\{(.*?)\}\s*\n", block, re.S)
    name = Path(img.group(1)).stem + ".png"
    path = Path(fig_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing -- run notebook 04")
    return {"type": "figure", "image": str(path).replace("\\", "/"),
            "caption": inline(cap.group(1), bib_order, bib) if cap else []}


def build_model(tex="paper/main.tex", numbers="paper/numbers.tex",
                bib_path="paper/refs.bib", fig_dir="results/figures") -> dict:
    src = Path(tex).read_text(encoding="utf-8")
    src = re.sub(r"(?m)(?<!\\)%.*$", "", src)
    src = expand(src, load_macros(numbers))
    bib, bib_order = load_bib(bib_path), []

    title = re.search(r"\\title\{(.*?)\}\s*\n", src, re.S).group(1)
    title = re.sub(r"\s+", " ", re.sub(r"\\\\", " ", title)).strip()
    author = re.search(r"\\author\{(.*?)\}\s*\n\\date", src, re.S).group(1)
    authors = ["".join(r["text"] for r in inline(a, bib_order, bib)).strip()
               for a in re.sub(r"\\\\", "\n", author).split("\n") if a.strip()]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                         src, re.S).group(1)

    end_abs = r"\end{abstract}"
    body = src[src.index(end_abs) + len(end_abs):]
    body = body[:body.index(r"\bibliographystyle")]

    blocks, pos = [], 0
    env = re.compile(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", re.S)
    chunks = []
    for m in env.finditer(body):
        chunks.append(("text", body[pos:m.start()]))
        chunks.append(("env", m.group(0)))
        pos = m.end()
    chunks.append(("text", body[pos:]))

    for kind, chunk in chunks:
        if kind == "env":
            blocks.append(_table(chunk, bib_order, bib) if "tabular" in chunk
                          else _figure(chunk, bib_order, bib, fig_dir))
            continue
        for part in re.split(r"\n\s*\n", chunk):
            part = part.strip()
            if not part:
                continue
            sec = re.match(r"\\section\*?\{([^}]*)\}\s*(.*)", part, re.S)
            if sec:
                blocks.append({"type": "heading", "level": 1,
                               "runs": inline(sec.group(1), bib_order, bib)})
                part = sec.group(2).strip()
                if not part:
                    continue
            par = re.match(r"\\paragraph\{([^}]*)\}\s*(.*)", part, re.S)
            if par:
                runs = [{"text": par.group(1).rstrip(".") + ". ", "bold": True}]
                runs += inline(par.group(2), bib_order, bib)
                blocks.append({"type": "paragraph", "runs": runs})
                continue
            if part.startswith("\\begin{itemize}"):
                items = re.findall(r"\\item\s+(.*?)(?=\\item|\\end\{itemize\})",
                                   part, re.S)
                for it in items:
                    blocks.append({"type": "bullet",
                                   "runs": inline(it, bib_order, bib)})
                continue
            blocks.append({"type": "paragraph",
                           "runs": inline(part, bib_order, bib)})

    references = [format_reference(k, bib[k]) for k in bib_order if k in bib]
    return {"title": title.strip(), "authors": authors,
            "abstract": inline(abstract, bib_order, bib),
            "blocks": blocks, "references": references}


def write_model(path, **kw):
    model = build_model(**kw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    return model
