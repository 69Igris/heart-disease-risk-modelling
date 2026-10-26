"""Bayesian network checks and leak-free preprocessing checks for the logistic regression."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.special import expit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import load, discretize, FEATURES, NUM
from bayes import BNClassifier, BN_EDGES
from classical import LogReg

@pytest.fixture(scope='module')
def data():
    return load()


@pytest.fixture(scope='module')
def fitted(data):
    xd = discretize(data)
    return BNClassifier(BN_EDGES).fit(xd, data.target.to_numpy()), xd


def test_cpd_normalisation_and_target_table(fitted):
    model, _ = fitted
    assert model.model.check_model()
    assert np.prod(model.model.get_cpds('target').values.shape[1:]) == 48
    for cpd in model.model.get_cpds():
        np.testing.assert_allclose(cpd.values.sum(axis=0), 1, atol=1e-12)


def test_every_complete_row_explanation_matches_exact_inference(fitted):
    model, xd = fitted
    complete = xd.dropna()
    assert len(complete) == 297
    for _, row in complete.iterrows():
        expected = model.predict_proba_one(row)
        actual = expit(sum(model.contributions(row).values()))
        assert abs(actual - expected) < 1e-12


def test_incomplete_evidence_uses_inference_not_additive_explanation(fitted):
    model, xd = fitted
    row = xd.iloc[0].copy(); row['exang'] = np.nan
    assert 0 <= model.predict_proba_one(row) <= 1
    with pytest.raises(ValueError, match='complete evidence'):
        model.contributions(row)


def test_completely_missing_evidence_is_valid(fitted):
    model, xd = fitted
    row = xd.iloc[0].copy(); row[:] = np.nan
    assert np.isfinite(model.predict_proba_one(row))


def test_model_prediction_does_not_change_after_other_test_row(fitted):
    model, xd = fitted
    p = model.predict_proba_one(xd.iloc[0])
    model.predict_proba_one(xd.iloc[1])
    assert model.predict_proba_one(xd.iloc[0]) == p


def test_training_only_preprocessing(data):
    tr = data.iloc[:242]; te = data.iloc[242:].copy()
    m = LogReg().fit(tr[FEATURES], tr.target.to_numpy())
    imp = m.pre.named_transformers_['num'].named_steps['imp']
    np.testing.assert_allclose(imp.statistics_, tr[NUM].median())
    scaler = m.pre.named_transformers_['num'].named_steps['sc']
    expected = tr[NUM].fillna(tr[NUM].median()).mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected)
    before = scaler.mean_.copy(); te['age'] = 10000
    p = m.predict_proba(te[FEATURES])
    np.testing.assert_array_equal(before, scaler.mean_)
    assert np.isfinite(p).all()


def test_held_out_unknown_category_is_handled(data):
    m = LogReg().fit(data.iloc[:242][FEATURES], data.iloc[:242].target.to_numpy())
    row = data.iloc[[242]][FEATURES].copy(); row['thal'] = 99
    p = m.predict_proba(row)
    assert np.isfinite(p).all()
