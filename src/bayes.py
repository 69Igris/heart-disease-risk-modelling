"""Hand-specified Bayesian network and Naive Bayes with pgmpy (same estimator, same missing-data handling)."""
import numpy as np, pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from data import STATES, DISC_FEATURES

Y = "target"

# Domain-informed DAG: risk factors -> disease -> clinical findings.
BN_EDGES = (
    [(v, Y) for v in ["age", "sex", "trestbps", "chol", "fbs"]]            # risk factors -> disease
    + [("age", "trestbps")]                                                 # blood pressure rises with age
    + [(Y, v) for v in ["cp", "exang", "oldpeak", "slope", "restecg", "apmhr", "ca", "thal"]]  # disease -> findings
    + [("exang", "cp"), ("oldpeak", "slope")]                              # symptom / ECG-test dependencies
)
NB_EDGES = [(Y, v) for v in DISC_FEATURES]


class BNClassifier:
    """P(disease | evidence) by exact variable elimination. Missing evidence is simply left out (marginalised)."""
    def __init__(self, edges, ess=5.0, features=None):
        self.edges, self.ess = edges, ess
        self.features = features or DISC_FEATURES

    def _edges(self):
        keep = set(self.features) | {Y}
        return [(a, b) for a, b in self.edges if a in keep and b in keep]

    def fit(self, Xd: pd.DataFrame, y):
        data = Xd[self.features].copy(); data[Y] = np.asarray(y)
        sn = {v: list(STATES[v]) for v in self.features}; sn[Y] = [0, 1]
        self.model = DiscreteBayesianNetwork(self._edges())
        self.model.add_nodes_from(self.features + [Y])
        est = BayesianEstimator(self.model, data, state_names=sn)
        self.model.add_cpds(*est.get_parameters(prior_type="BDeu", equivalent_sample_size=self.ess))
        self.model.check_model()
        self._ve = VariableElimination(self.model); self._cache = {}
        return self

    def predict_proba_one(self, row: pd.Series) -> float:
        ev = {k: row[k] for k in self.features if pd.notna(row[k])}
        key = tuple(sorted(ev.items(), key=lambda kv: kv[0]))
        if key not in self._cache:
            q = self._ve.query([Y], evidence=ev, show_progress=False)
            self._cache[key] = float(q.values[list(q.state_names[Y]).index(1)])
        return self._cache[key]

    def predict_proba(self, Xd: pd.DataFrame):
        return np.array([self.predict_proba_one(r) for _, r in Xd.iterrows()])

    # ----- interpretability: exact Markov-blanket decomposition of the posterior log-odds -----
    def _cpd_lookup(self, node, state, parents_state):
        cpd = self.model.get_cpds(node)
        idx = {node: cpd.state_names[node].index(state)}
        for p, s in parents_state.items():
            idx[p] = cpd.state_names[p].index(s)
        order = [cpd.variables[0]] + list(cpd.variables[1:])
        return cpd.values[tuple(idx[v] for v in order)]

    def contributions(self, row: pd.Series) -> dict:
        """log-odds = [risk-factor prior logit of P(D|parents)] + sum over children of log-likelihood ratios."""
        if row[self.features].isna().any():
            raise ValueError("Additive explanation requires complete evidence; use exact inference for missing evidence")
        out = {}
        pa = self.model.get_parents(Y)
        pa_state = {p: row[p] for p in pa}
        if all(pd.notna(v) for v in pa_state.values()):
            p1 = self._cpd_lookup(Y, 1, pa_state)
            out["risk-factor prior"] = np.log(p1 / (1 - p1))
        for c in self.model.get_children(Y):
            if pd.isna(row[c]):
                out[c] = 0.0; continue
            other = {p: row[p] for p in self.model.get_parents(c) if p != Y}
            l1 = self._cpd_lookup(c, row[c], {**other, Y: 1})
            l0 = self._cpd_lookup(c, row[c], {**other, Y: 0})
            out[c] = float(np.log(l1) - np.log(l0))
        return out
