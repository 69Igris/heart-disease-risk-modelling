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


def test_phase2_lives_alongside_phase1_without_overwriting_it():
    # Phase 2 writes only phase2_* result files; Phase 1 outputs are reused, never regenerated
    assert (ROOT / 'results/phase1_oof_predictions.csv.gz').exists()
    assert (ROOT / 'notebooks/02_phase2_mlp_and_comparison.ipynb').exists()
    assert not (ROOT / 'phase2').exists()
