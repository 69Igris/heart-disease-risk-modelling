"""Classical (non-neural) reference models with leak-free preprocessing (all fitted on the training fold only)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data import NUM, CAT, CAT_STATES, DISC_NUM, STATES


def make_preprocessor(mode="continuous", features=None):
    """'continuous': standardised numerics + one-hot categoricals.  'discretized': one-hot of the BN's bins."""
    if mode in ("continuous", "matched_continuous"):
        numeric = NUM if mode == "continuous" else ["age", "trestbps", "chol", "apmhr", "oldpeak"]
        num = [f for f in numeric if features is None or f in features]
        cat = [f for f in CAT if features is None or f in features]
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(categories=[CAT_STATES[c] for c in cat], handle_unknown="ignore", sparse_output=False))]), cat)])
    cols = [f for f in DISC_NUM + CAT if features is None or f in features]
    return ColumnTransformer([("oh", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(categories=[STATES[c] for c in cols], handle_unknown="ignore", sparse_output=False))]), cols)])


class LogReg:
    def __init__(self, mode="continuous", features=None, C=0.5):
        self.pre, self.C = make_preprocessor(mode, features), C
    def fit(self, X, y):
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(self.pre.fit_transform(X), y); return self
    def predict_proba(self, X): return self.clf.predict_proba(self.pre.transform(X))[:, 1]


class Forest:
    def __init__(self, features=None, seed=0):
        self.pre, self.seed = make_preprocessor("continuous", features), seed
    def fit(self, X, y):
        self.clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", random_state=self.seed, n_jobs=1).fit(self.pre.fit_transform(X), y); return self
    def predict_proba(self, X): return self.clf.predict_proba(self.pre.transform(X))[:, 1]


class Prior:
    def fit(self, X, y): self.p = float(np.mean(y)); return self
    def predict_proba(self, X): return np.full(len(X), self.p)
