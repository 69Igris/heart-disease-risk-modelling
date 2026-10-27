"""Evaluation: repeated stratified CV, bootstrap intervals, and the corrected paired t-test.

Reading order:  run_cv  ->  mean_metrics  ->  summarize  ->  paired
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from metrics import all_metrics

KEYS = ["ROC AUC", "PR AUC", "Sensitivity", "Specificity", "Brier", "ECE"]


def run_cv(make, X, y, splits, K=5):
    """Run every (train, test) split and collect out-of-fold (OOF) predictions.

    make(s)  returns a fresh, untrained model for split number s
    splits   list of (train_idx, test_idx); K splits make one repeat of K-fold CV
    Returns the OOF probabilities (one row per repeat, one column per patient)
    and the AUC of each individual fold.
    """
    n_repeats = len(splits) // K
    oof = np.full((n_repeats, len(y)), np.nan)      # NaN = "not predicted yet"
    fold_auc = []

    for s, (train, test) in enumerate(splits):
        if np.intersect1d(train, test).size:         # safety: no row in both sets
            raise ValueError("Training and validation indices overlap")

        model = make(s).fit(X.iloc[train], y[train])
        p = model.predict_proba(X.iloc[test])

        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Model returned invalid probabilities")

        oof[s // K, test] = p                        # s // K = which repeat this split belongs to
        fold_auc.append(roc_auc_score(y[test], p))

    if np.isnan(oof).any():                          # safety: every patient predicted once per repeat
        raise ValueError("Incomplete out-of-fold coverage")
    return dict(oof=oof, fold_auc=np.array(fold_auc))


def mean_metrics(y, oof, idx=None, full=False):
    """Compute the metrics separately for each repeat, then average over repeats.
    idx = optional patient indices (used for bootstrap samples)."""
    y_used = y if idx is None else y[idx]
    per_repeat = []
    for repeat in oof:
        p = repeat if idx is None else repeat[idx]
        per_repeat.append(all_metrics(y_used, p, full=full))
    return pd.DataFrame(per_repeat).mean()


def summarize(results, y, B=500, seed=2026):
    """One table row per model with the mean metrics and 95% bootstrap intervals.

    The bootstrap draws B resamples of the 303 patients (with replacement) and
    recomputes the metrics on each one; the 2.5th and 97.5th percentiles give the interval.
    All models use the SAME resamples so that differences between models are paired.
    Returns (table, bootstrap_values_per_model).
    """
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(B)]

    rows, boot_values = [], {}
    for name, d in results.items():
        if name == "Prior only":                     # constant predictor: no interval needed
            m = mean_metrics(y, d["oof"])
            rows.append({"model": name, **{k: m[k] for k in KEYS}})
            continue

        m = mean_metrics(y, d["oof"], full=True)
        row = {"model": name, **m.to_dict()}
        b = np.array([mean_metrics(y, d["oof"], idx)[KEYS].values for idx in boots])
        boot_values[name] = b
        for j, k in enumerate(KEYS):
            row[k + " lo"], row[k + " hi"] = np.percentile(b[:, j], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows), boot_values


def paired(results, bv, pairs, K=5, y=None):
    """Compare model A with model B on ROC AUC, PR AUC and Brier.

    The interval for the difference (A - B) comes from the paired bootstrap values.
    For ROC AUC we also report the corrected resampled t-test (Nadeau & Bengio, 2003):
    the 25 folds share training data, so the plain t-test variance is too small
    and is inflated by the factor (1/J + 1/(K-1)).
    """
    out = []
    for a, b in pairs:
        for k in ("ROC AUC", "PR AUC", "Brier"):
            j = KEYS.index(k)
            diff = bv[a][:, j] - bv[b][:, j]
            if y is not None:
                mean_diff = float(mean_metrics(y, results[a]["oof"])[k]
                                  - mean_metrics(y, results[b]["oof"])[k])
            else:
                mean_diff = diff.mean()
            row = {"A": a, "B": b, "metric": k, "mean diff (A-B)": mean_diff,
                   "lo": np.percentile(diff, 2.5), "hi": np.percentile(diff, 97.5)}

            if k == "ROC AUC":
                fold_diff = results[a]["fold_auc"] - results[b]["fold_auc"]
                J = len(fold_diff)
                var = fold_diff.var(ddof=1) * (1 / J + 1 / (K - 1))
                t = fold_diff.mean() / np.sqrt(var) if var > 0 else 0.0
                row["corrected t"], row["corrected p"] = t, 2 * stats.t.sf(abs(t), J - 1)
            out.append(row)
    return pd.DataFrame(out)
