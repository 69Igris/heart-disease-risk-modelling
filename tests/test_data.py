"""Data checks: official file, schema, target, missingness, codebook. Run: python -m pytest -q"""
from pathlib import Path
import sys
import hashlib
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import load, FEATURES, CAT_STATES
from download_data import SHA256

@pytest.fixture(scope='module')
def data():
    return load()


def test_official_file_checksum():
    assert hashlib.sha256((ROOT / 'data/processed.cleveland.data').read_bytes()).hexdigest() == SHA256


def test_schema_target_missingness(data):
    assert data.shape == (303, 15)
    assert data.target.value_counts().to_dict() == {0: 164, 1: 139}
    assert data.ca.isna().sum() == 4 and data.thal.isna().sum() == 2
    assert data[FEATURES].isna().any(axis=1).sum() == 6
    assert 'num' not in FEATURES and 'target' not in FEATURES


def test_no_identical_predictor_rows(data):
    assert not data[FEATURES].duplicated().any()


@pytest.mark.parametrize('column', list(CAT_STATES))
def test_category_codebook(data, column):
    assert set(data[column].dropna()).issubset(CAT_STATES[column])
