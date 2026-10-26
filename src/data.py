"""Data loading, target definition and (fixed, documented) discretization.

Target: num > 0  <=>  angiographic narrowing > 50 % in any major vessel (heart disease present).
The cut points below are fixed constants in this supplied implementation, not fitted on validation folds.
They are modelling choices, not validated clinical rules or proof of preregistration.
"""
from pathlib import Path
import numpy as np, pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUM + CAT

# Full state spaces from the UCI data dictionary (prior knowledge; avoids "unseen state" failures in folds)
CAT_STATES = {"sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2],
              "exang": [0, 1], "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7]}

# (variable, edges, labels). Right-open bins [a, b).
CUTS = {
    "age":      ([-np.inf, 50, 60, np.inf],      ["<50", "50-59", ">=60"]),
    "trestbps": ([-np.inf, 130, np.inf],         ["<130", ">=130"]),                 # modelling threshold, not a hypertension diagnosis
    "chol":     ([-np.inf, 240, np.inf],        ["<240", ">=240"]),                 # modelling threshold
    "apmhr":    ([-np.inf, 0.70, 0.85, np.inf],    ["<70%", "70-84%", ">=85%"]),       # peak HR as % of age-predicted max (220-age)
    "oldpeak":  ([-np.inf, 1, 2, np.inf],       ["<1mm", "1-<2mm", ">=2mm"]),       # supplied ST-depression units; not a diagnostic rule
}
DISC_NUM = ["age", "trestbps", "chol", "apmhr", "oldpeak"]   # discretized stand-ins for continuous variables
DISC_FEATURES = DISC_NUM + CAT                               # every variable the discrete models see
STATES = {**{k: v[1] for k, v in CUTS.items()}, **CAT_STATES}


ROOT = Path(__file__).resolve().parent.parent


def load(path=ROOT / "data" / "processed.cleveland.data") -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS, na_values="?")
    if df.shape[1] != len(COLS) or df["num"].isna().any() or not df["num"].isin(range(5)).all():
        raise ValueError("Expected original Cleveland schema and non-missing num in 0..4")
    df["target"] = (df["num"] > 0).astype(int)
    return df


def add_age_adjusted_hr(df):
    """Continuous input control matching the Bayesian model's HR representation."""
    out = df.copy()
    if (out["age"].dropna() >= 220).any():
        raise ValueError("Age must be below 220 for the defined HR ratio")
    out["apmhr"] = out["thalach"] / (220.0 - out["age"])
    return out


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of discrete states for the BN/NB. Missing stays NaN (never imputed)."""
    out = pd.DataFrame(index=df.index)
    d = add_age_adjusted_hr(df)
    for v in DISC_NUM:
        edges, labels = CUTS[v]
        out[v] = pd.cut(d[v], edges, right=False, labels=labels).astype(object)
    for v in CAT:
        out[v] = d[v].astype(float).where(d[v].notna(), np.nan).astype(object)
        out[v] = out[v].map(lambda z: int(z) if pd.notna(z) else np.nan)
    return out


def cutpoint_table() -> pd.DataFrame:
    rows = []
    for v, (edges, labels) in CUTS.items():
        rows.append({"variable": v, "states": " | ".join(labels),
                     "interior cut points": ", ".join(str(e) for e in edges[1:-1])})
    return pd.DataFrame(rows)
