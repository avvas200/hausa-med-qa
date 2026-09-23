"""Pre-registered statistics and decision rules (PREREGISTRATION.md, sections 3 and 7).
Committed before any evaluation-set result is generated.

All accuracy inputs are boolean arrays aligned by item (same item order in
every condition). Reordering inputs are (n_items, 3) arrays.
"""
import numpy as np
from scipy.stats import binomtest

N_BOOT = 10_000
BOOT_SEED = 20260923


# ---------- basic tests ----------

def mcnemar_exact(a, b):
    """Exact two-sided McNemar on paired correctness vectors a, b."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int((a & ~b).sum())   # a right, b wrong
    n01 = int((~a & b).sum())   # a wrong, b right
    n = n10 + n01
    p = 1.0 if n == 0 else binomtest(n10, n, 0.5, alternative="two-sided").pvalue
    return {"acc_a": a.mean(), "acc_b": b.mean(), "diff": a.mean() - b.mean(),
            "n10": n10, "n01": n01, "p": float(p)}


def paired_bootstrap(stat_fn, *arrays, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile 95% CI for stat_fn(*resampled arrays), resampling items jointly."""
    arrays = [np.asarray(x) for x in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = stat_fn(*arrays)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = stat_fn(*[x[idx] for x in arrays])
    finite = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan))
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "frac_undefined": float(1 - len(finite) / n_boot)}


def diff_ci(a, b, **kw):
    return paired_bootstrap(lambda x, y: x.mean() - y.mean(),
                            np.asarray(a, float), np.asarray(b, float), **kw)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for {name: p}."""
    names = sorted(pvals, key=pvals.get)
    m, running, out = len(names), 0.0, {}
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


# ---------- robustness metrics (H3) ----------

def consistency_vector(choices):
    """choices: (n, 3) original-option indices picked under each ordering
    (-1 = unparseable). True where all three picks agree and are valid."""
    c = np.asarray(choices)
    return (c[:, 0] >= 0) & (c == c[:, [0]]).all(axis=1)


def robust_correct_vector(correct):
    """correct: (n, 3) booleans. True where correct under all orderings."""
    return np.asarray(correct, bool).all(axis=1)


# ---------- decision rules ----------

def h1_language_gap(en_by_model: dict, ha_by_model: dict, alpha=0.05, min_gap=0.05):
    raw = {m: mcnemar_exact(en_by_model[m], ha_by_model[m]) for m in en_by_model}
    adj = holm({m: r["p"] for m, r in raw.items()})
    out = {}
    for m, r in raw.items():
        ci = diff_ci(en_by_model[m], ha_by_model[m])
        supported = r["diff"] >= min_gap and adj[m] < alpha and ci["ci_low"] > 0
        out[m] = {**r, "p_holm": adj[m], "ci": ci, "supported": bool(supported)}
    return out


def h2_translate_test(en, ha, ha2en, h1_supported: dict, alpha=0.05, min_ratio=0.5):
    """en/ha/ha2en: {model: bool array}. Tested only where H1 was supported."""
    models = [m for m in en if h1_supported.get(m)]
    raw = {m: mcnemar_exact(ha2en[m], ha[m]) for m in models}
    adj = holm({m: r["p"] for m, r in raw.items()}) if raw else {}

    def ratio(e, h, t):
        gap = e.mean() - h.mean()
        return (t.mean() - h.mean()) / gap if gap > 0 else np.nan

    out = {}
    for m in models:
        ci = paired_bootstrap(ratio, np.asarray(en[m], float),
                              np.asarray(ha[m], float), np.asarray(ha2en[m], float))
        supported = ci["point"] >= min_ratio and adj[m] < alpha and ci["ci_low"] > 0
        out[m] = {**raw[m], "p_holm": adj[m], "recovery": ci, "supported": bool(supported)}
    return out


def h3_consistency_gap(en_choices: dict, ha_choices: dict, min_gap=0.05):
    out = {}
    for m in en_choices:
        ce = consistency_vector(en_choices[m])
        ch = consistency_vector(ha_choices[m])
        ci = diff_ci(ce, ch)
        out[m] = {"cons_en": ce.mean(), "cons_ha": ch.mean(), "ci": ci,
                  "supported": bool(ci["point"] >= min_gap and ci["ci_low"] > 0)}
    return out


def h4_intervention(base, tuned_by_seed: dict, alpha=0.025, min_gain=0.03):
    """base: bool array (HA, base model). tuned_by_seed: {seed: bool array}.
    Pooled McNemar sums discordant counts across seeds.
    NOTE: confirm this pooling matches src/stats.py in dpo-reasoning-transfer;
    if it differs, use that implementation so the two studies are comparable."""
    per_seed, n10, n01 = {}, 0, 0
    for s, t in tuned_by_seed.items():
        r = mcnemar_exact(t, base)
        per_seed[s] = {**r, "ci": diff_ci(t, base)}
        n10 += r["n10"]
        n01 += r["n01"]
    n = n10 + n01
    p_pooled = 1.0 if n == 0 else float(binomtest(n10, n, 0.5).pvalue)
    mean_gain = float(np.mean([r["diff"] for r in per_seed.values()]))
    supported = (mean_gain >= min_gain and p_pooled < alpha
                 and all(r["ci"]["ci_low"] > 0 for r in per_seed.values()))
    return {"per_seed": per_seed, "mean_gain": mean_gain, "p_pooled": p_pooled,
            "supported": bool(supported)}
