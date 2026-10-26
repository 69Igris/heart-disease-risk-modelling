"""Checks on saved explanation output and the Phase 1 / Phase 2 boundary."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def test_saved_explanation_rows_have_missing_not_zero_fabrication():
    w = pd.read_csv(ROOT / 'results/phase1_bn_weights.csv')
    assert len(w) == 303
    assert w.isna().all(axis=1).sum() == 6


def test_no_phase2_implementation_shipped():
    assert not (ROOT / 'phase2').exists()
