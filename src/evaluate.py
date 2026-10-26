"""Repeated stratified CV, patient-level bootstrap and the Nadeau-Bengio corrected resampled t-test."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from metrics import all_metrics

KEYS = ["ROC AUC", "PR AUC", "Sensitivity", "Specificity", "Brier", "ECE"]


def run_cv(make, X, y, splits, K=5):
    R = len(splits) // K; oof = np.full((R, len(y)), np.nan); fold_auc = []
    for s, (tr, te) in enumerate(splits):
        if np.intersect1d(tr, te).size:
            raise ValueError("Training and validation indices overlap")
        p = make(s).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Model returned invalid probabilities")
        oof[s // K, te] = p; fold_auc.append(roc_auc_score(y[te], p))
    if np.isnan(oof).any():
        raise ValueError("Incomplete out-of-fold coverage")
    return dict(oof=oof, fold_auc=np.array(fold_auc))


def mean_metrics(y, oof, idx=None, full=False):
    yy = y if idx is None else y[idx]
    return pd.DataFrame([all_metrics(yy, o if idx is None else o[idx], full=full) for o in oof]).mean()


def summarize(results, y, B=500, seed=2026):
    rng = np.random.default_rng(seed); boots = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    rows, bv = [], {}
    for name, d in results.items():
        if name == "Prior only":
            m = mean_metrics(y, d["oof"]); rows.append({"model": name, **{k: m[k] for k in KEYS}}); continue
        m = mean_metrics(y, d["oof"], full=True); row = {"model": name, **m.to_dict()}
        b = np.array([mean_metrics(y, d["oof"], idx)[KEYS].values for idx in boots]); bv[name] = b
        for j, k in enumerate(KEYS): row[k + " lo"], row[k + " hi"] = np.percentile(b[:, j], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows), bv


def paired(results, bv, pairs, K=5, y=None):
    out = []
    for a, b in pairs:
        for k in ("ROC AUC", "PR AUC", "Brier"):
            j = KEYS.index(k); d = bv[a][:, j] - bv[b][:, j]
            row = {"A": a, "B": b, "metric": k, "mean diff (A-B)": float(mean_metrics(y, results[a]["oof"])[k] - mean_metrics(y, results[b]["oof"])[k]) if y is not None else d.mean(), "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)}
            if k == "ROC AUC":
                dd = results[a]["fold_auc"] - results[b]["fold_auc"]; J = len(dd)
                var = dd.var(ddof=1) * (1 / J + 1 / (K - 1)); t = dd.mean() / np.sqrt(var) if var > 0 else 0.0
                row["corrected t"], row["corrected p"] = t, 2 * stats.t.sf(abs(t), J - 1)
            out.append(row)
    return pd.DataFrame(out)
