"""Discretisation and engineered heart-rate feature checks."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import load, discretize, add_age_adjusted_hr

@pytest.fixture(scope='module')
def data():
    return load()


def test_fixed_bin_boundary_conventions(data):
    d = pd.concat([data.iloc[[0]]] * 3, ignore_index=True)
    d['age'] = [49.99, 50, 60]
    d['trestbps'] = [129.99, 130, 500]
    x = discretize(d)
    assert x.age.tolist() == ['<50', '50-59', '>=60']
    assert x.trestbps.tolist() == ['<130', '>=130', '>=130']


def test_missing_values_are_not_silently_imputed(data):
    x = discretize(data)
    assert x.ca.isna().sum() == 4 and x.thal.isna().sum() == 2
    assert x.drop(columns=['ca', 'thal']).notna().all().all()


def test_matched_hr_feature_is_same_transform(data):
    x = add_age_adjusted_hr(data)
    np.testing.assert_allclose(x.apmhr, data.thalach / (220 - data.age))
    bad = data.iloc[[0]].copy(); bad['age'] = 220
    with pytest.raises(ValueError):
        add_age_adjusted_hr(bad)
