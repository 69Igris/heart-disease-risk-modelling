"""Classical (non-neural) reference models.

Every model here is a small wrapper with the same three pieces:
    fit(X, y)          learn from the training rows only
    predict_proba(X)   return P(disease) for each row, a number between 0 and 1
The wrapper also owns its preprocessing (fill missing values, scale, one-hot encode).
Because the preprocessing is fitted inside fit(), it only ever sees the training fold,
so nothing from the held-out fold leaks into training.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data import NUM, CAT, CAT_STATES, DISC_NUM, STATES

# Same five variables the Bayesian network bins, but kept as continuous numbers.
# Used for the "matched" control that separates "binning" from "heart-rate feature".
MATCHED_NUMERIC = ["age", "trestbps", "chol", "apmhr", "oldpeak"]


def _keep(columns, features):
    """Keep only the columns listed in `features` (None means keep them all)."""
    return [c for c in columns if features is None or c in features]


def _numeric_pipeline():
    """Numbers: fill missing with the training median, then standardise."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])


def _category_pipeline(columns, states):
    """Categories: fill missing with the most common training value, then one-hot encode.
    The allowed category values come from the data dictionary, so an unseen value
    in a held-out fold cannot crash the encoder (it becomes all zeros)."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(categories=[states[c] for c in columns],
                             handle_unknown="ignore", sparse_output=False)),
    ])


def make_preprocessor(mode="continuous", features=None):
    """Build the preprocessing step.

    mode="continuous"          scaled numeric columns + one-hot categorical columns
    mode="matched_continuous"  same, but with the age-adjusted heart-rate feature
    mode="discretized"         one-hot encoding of the same bins the Bayesian network uses
    features                   optional list to restrict which columns are used
    """
    if mode in ("continuous", "matched_continuous"):
        numeric = NUM if mode == "continuous" else MATCHED_NUMERIC
        num_cols = _keep(numeric, features)
        cat_cols = _keep(CAT, features)
        return ColumnTransformer([
            ("num", _numeric_pipeline(), num_cols),
            ("cat", _category_pipeline(cat_cols, CAT_STATES), cat_cols),
        ])

    cols = _keep(DISC_NUM + CAT, features)
    return ColumnTransformer([("oh", _category_pipeline(cols, STATES), cols)])


class LogReg:
    """Regularised logistic regression (C=0.5, L2 penalty)."""

    def __init__(self, mode="continuous", features=None, C=0.5):
        self.pre = make_preprocessor(mode, features)
        self.C = C

    def fit(self, X, y):
        X_train = self.pre.fit_transform(X)          # fitted on the training fold only
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(X_train, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.pre.transform(X))[:, 1]


class Forest:
    """Random forest: 300 trees, at least 3 rows per leaf, sqrt(p) features per split."""

    def __init__(self, features=None, seed=0):
        self.pre = make_preprocessor("continuous", features)
        self.seed = seed

    def fit(self, X, y):
        X_train = self.pre.fit_transform(X)
        self.clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            random_state=self.seed, n_jobs=1,
        ).fit(X_train, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.pre.transform(X))[:, 1]


class Prior:
    """Baseline that ignores the patient and always predicts the training prevalence."""

    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)
