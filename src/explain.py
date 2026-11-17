"""Phase 2 explanations: Kernel SHAP for the MLP, in log-odds units.

Why log-odds: the Phase 1 Bayesian-network explanation (bayes.BNClassifier.contributions)
is an exact additive decomposition of the posterior log-odds. Explaining the MLP's
log-odds (its output logit) with SHAP puts both models on the same scale.

Why Kernel SHAP on the 13 raw columns: the MLP sees one-hot columns, but people reason
about "chest pain type", not "cp_4". Explaining the raw columns (preprocessing included in
the explained function) gives one value per original variable, like the BN.

Background data: a random sample of real, complete rows from the TRAINING fold, so held-out
patients are explained against data the model was trained on, not against themselves.
(A k-means summary is not used: its centroids would turn category codes such as cp=4 into
fractions like 3.4 that the one-hot encoder cannot represent.)
"""
import pandas as pd
import shap


def mlp_shap(model, X_train: pd.DataFrame, X_explain: pd.DataFrame, n_background=40, nsamples=500, seed=0):
    """Return a DataFrame (rows = X_explain, columns = features) of SHAP values in log-odds,
    and the base value (expected log-odds over the background)."""
    cols = list(X_train.columns)
    # Explained rows may contain NaN (ca/thal): the model's own training-fold imputer handles them.
    background = shap.sample(X_train.dropna().values, n_background, random_state=seed)

    def f(a):
        return model.decision_function(pd.DataFrame(a, columns=cols))

    ex = shap.KernelExplainer(f, background, seed=seed)
    sv = ex.shap_values(X_explain.values, nsamples=nsamples, silent=True)
    return pd.DataFrame(sv, columns=cols, index=X_explain.index), float(ex.expected_value)
