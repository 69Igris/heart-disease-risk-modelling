"""Phase 2 checks: the MLP wrapper is leak-free, deterministic and valid; saved Phase 2 results are consistent."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import load, FEATURES, discretize
from evaluate import mean_metrics
from mlp import MLP, grid
from explain import mlp_shap

FAST = {"hidden": (16,), "weight_decay": 1e-2, "dropout": 0.3}


@pytest.fixture(scope='module')
def data():
    return load()


@pytest.fixture(scope='module')
def split(data):
    return next(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=2026).split(data, data.target))


def test_probabilities_valid_and_deterministic(data, split):
    tr, te = split
    X, y = data[FEATURES], data.target.to_numpy()
    p1 = MLP(tune=False, config=FAST, epochs=60, seed=3).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
    p2 = MLP(tune=False, config=FAST, epochs=60, seed=3).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
    assert np.isfinite(p1).all() and ((p1 > 0) & (p1 < 1)).all()
    assert np.array_equal(p1, p2)


def test_preprocessing_fitted_on_training_rows_only(data, split):
    tr, _ = split
    X, y = data[FEATURES], data.target.to_numpy()
    m = MLP(tune=False, config=FAST, epochs=5, n_seeds=1).fit(X.iloc[tr], y[tr])
    scaler = m.pre.named_transformers_['num'].named_steps['sc']
    num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']      # no missing values in these columns
    assert np.allclose(scaler.mean_, X.iloc[tr][num].mean())     # learned from the training rows...
    assert not np.allclose(scaler.mean_, X[num].mean())          # ...not from all 303 rows


def test_missing_values_and_binned_inputs(data, split):
    tr, te = split
    y = data.target.to_numpy()
    Xd = discretize(data)
    p = MLP(mode='discretized', tune=False, config=FAST, epochs=20, n_seeds=1).fit(Xd.iloc[tr], y[tr]).predict_proba(Xd)
    assert np.isfinite(p).all()                       # rows with missing ca/thal still predicted
    assert data[FEATURES].isna().any(axis=1).sum() == 6


def test_nested_tuning_picks_a_grid_point(data, split):
    tr, _ = split
    X, y = data[FEATURES], data.target.to_numpy()
    m = MLP(seed=0, n_seeds=1).fit(X.iloc[tr], y[tr])
    assert m.config in grid() and 1 <= m.epochs <= 300
    assert len(m.search_) == len(grid())
    assert m.inner_logloss_ == min(r['inner_logloss'] for r in m.search_)


def test_shap_is_additive_in_log_odds(data, split):
    tr, te = split
    X, y = data[FEATURES], data.target.to_numpy()
    m = MLP(tune=False, config=FAST, epochs=40, n_seeds=2).fit(X.iloc[tr], y[tr])
    S, base = mlp_shap(m, X.iloc[tr], X.iloc[te[:8]], n_background=10, nsamples=300)
    assert np.allclose(S.sum(1).to_numpy() + base, m.decision_function(X.iloc[te[:8]]), atol=1e-4)


def test_saved_phase2_predictions_and_metrics(data):
    saved = pd.read_csv(ROOT / 'results/phase2_oof_predictions.csv.gz')
    metrics = pd.read_csv(ROOT / 'results/phase2_metrics.csv').set_index('model')
    assert set(saved.model) == {'MLP', 'MLP (binned)', 'MLP (no ca/thal)'}
    assert saved.groupby(['model', 'repeat', 'patient']).size().eq(1).all()
    assert saved.p.between(0, 1).all()
    y = data.target.to_numpy()
    for name in ['MLP', 'MLP (binned)']:
        oof = saved[saved.model == name].sort_values(['repeat', 'patient']).p.to_numpy().reshape(5, 303)
        actual = mean_metrics(y, oof, full=False)
        for key in ['ROC AUC', 'PR AUC', 'Brier', 'Sensitivity', 'Specificity']:
            assert abs(actual[key] - metrics.loc[name, key]) < 1e-9


def test_phase2_hparams_cover_every_split():
    hp = pd.read_csv(ROOT / 'results/phase2_mlp_hparams.csv')
    assert sorted(hp.split) == list(range(25))
    assert hp.epochs.between(1, 300).all()
