"""Data loading and target definition for the UCI processed Cleveland file.

Target: num > 0  <=>  angiographic narrowing > 50 % in any major vessel (heart disease present).
"""
from pathlib import Path
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUM + CAT

# Full state spaces from the UCI data dictionary (prior knowledge; avoids "unseen state" failures in folds)
CAT_STATES = {"sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2],
              "exang": [0, 1], "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7]}

ROOT = Path(__file__).resolve().parent.parent


def load(path=ROOT / "data" / "processed.cleveland.data") -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS, na_values="?")
    if df.shape[1] != len(COLS) or df["num"].isna().any() or not df["num"].isin(range(5)).all():
        raise ValueError("Expected original Cleveland schema and non-missing num in 0..4")
    df["target"] = (df["num"] > 0).astype(int)
    return df
