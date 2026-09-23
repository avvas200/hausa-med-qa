"""Frozen configuration. Any change after the pre-registration freeze must be
logged in PREREGISTRATION.md as a deviation."""
import os
from pathlib import Path

SEED = 20260923

# Dataset
MEDMCQA = "openlifescienceai/medmcqa"
EVAL_SPLIT = "validation"          # test labels are not public
TRAIN_SPLIT = "train"
N_EVAL = 500
N_TRAIN_POOL = 500
TRAIN_OVERSAMPLE = 1.4             # sample extra, then drop overlaps with eval
N_HUMAN_VALIDATION = 75
N_EXTRA_PERMUTATIONS = 2           # canonical order + 2 reorderings = 3 (H3)
MAX_QUESTION_WORDS = 150
NEAR_DUP_JACCARD = 0.8

# Items whose meaning depends on option position or on other options break
# under reordering (H3), so they are excluded before sampling.
EXCLUDE_OPTION_PATTERNS = [
    r"\ball of the above\b", r"\bnone of the above\b", r"\ball the above\b",
    r"\ball of these\b", r"\bnone of these\b", r"^\s*both\b", r"^\s*none\b",
    r"^\s*neither\b", r"^\s*all\s*$", r"\b[a-d]\s*(?:and|&|,)\s*[a-d]\b",
    r"\b(?:1|2|3|4)\s*(?:and|&|,)\s*(?:1|2|3|4)\b",
]
# Questions that need an image the model cannot see.
EXCLUDE_QUESTION_PATTERNS = [
    r"\bimage\b", r"\bfigure\b", r"\bshown\b", r"\bpicture\b", r"\bphotograph\b",
    r"\bx-?ray (?:given|below|above)\b", r"\bidentify the (?:structure|lesion|instrument)\b",
]

# Translation
NLLB_MODEL = os.environ.get("HAUSAMED_NLLB", "facebook/nllb-200-distilled-1.3B")
EN, HA = "eng_Latn", "hau_Latn"
TRANSLATION_BATCH = 16
TRANSLATION_MAX_LEN = 512

LETTERS = ["A", "B", "C", "D"]

# Storage (Google Drive, so artefacts survive Colab disconnects)
ROOT = Path(os.environ.get("HAUSAMED_ROOT", "/content/drive/MyDrive/hausa-med-qa"))
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"


def ensure_dirs():
    for p in (DATA, CACHE, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
