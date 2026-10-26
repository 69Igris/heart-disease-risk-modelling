"""Cross-validation split and saved-prediction checks."""
from pathlib import Path
import sys
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import load, FEATURES
from classical import Prior
from evaluate import run_cv, mean_metrics, summarize

@pytest.fixture(scope='module')
def data():
    return load()


def test_saved_splits_cover_each_row_once_per_repeat(data):
    manifest = pd.read_csv(ROOT / 'results/cv_test_assignments.csv')
    assert len(manifest) == 1515
    assert manifest.groupby(['repeat','patient']).size().eq(1).all()
    splits = list(RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=2026).split(data, data.target))
    for s, (tr, te) in enumerate(splits):
        assert not set(tr) & set(te)
        assert set(manifest.loc[manifest.split == s, 'patient']) == set(te)
        assert set(data.iloc[te].target) == {0, 1}


def test_saved_primary_predictions_and_metrics(data):
    saved = pd.read_csv(ROOT / 'results/phase1_oof_predictions.csv.gz')
    metrics = pd.read_csv(ROOT / 'results/phase1_metrics.csv').set_index('model')
    assert len(saved) == 6 * 5 * 303
    assert saved.groupby(['model','repeat','patient']).size().eq(1).all()
    assert saved.p.between(0,1).all()
    for name, group in saved.groupby('model'):
        oof = group.sort_values(['repeat','patient']).p.to_numpy().reshape(5,303)
        actual = mean_metrics(data.target.to_numpy(), oof, full=False)
        for key in ['ROC AUC','PR AUC','Brier','Sensitivity','Specificity']:
            assert abs(actual[key] - metrics.loc[name,key]) < 1e-12


def test_prior_summary_keeps_measured_auc(data):
    splits = list(RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=2026).split(data, data.target))
    y = data.target.to_numpy()
    result = run_cv(lambda s: Prior(), data[FEATURES], y, splits)
    summary, _ = summarize({'Prior only':result}, y, B=2)
    assert summary.iloc[0]['ROC AUC'] == roc_auc_score(y, result['oof'][0])
    assert summary.iloc[0]['PR AUC'] == average_precision_score(y, result['oof'][0])
