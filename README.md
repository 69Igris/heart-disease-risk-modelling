# Heart Disease Risk Modelling - Phase 1
Advanced Machine Learning, Project 15. UCI processed Cleveland data: Bayesian network, discrete Naive Bayes, logistic regression and random forest. **Prediction exercise only, not clinical decision support.** No MLP or SHAP implementation is included; those belong to Phase 2.

## Team
| Student | Enrollment number | Email |
|---|---|---|
| Aaryan Yadav | 2401010011 | aaryan.yadav2024@nst.rishihood.edu.in |
| Shivam Mishra | 2401010442 | shivam.mishra2024@nst.rishihood.edu.in |
| Satyam Kumar | 2401010429 | satyam.kumars2024@nst.rishihood.edu.in |
| Himank Kaushik | 2401010187 | himank.kaushik2024@nst.rishihood.edu.in |
| Mohammed Yaseen | 2401010281 | mohammed.yaseen2024@nst.rishihood.edu.in |

## Quick start (Python 3.12)
Open a terminal in the repository root.

```bash
python -m venv .venv
```
Activate on Windows PowerShell: `.venv\Scripts\Activate.ps1` (if blocked, use Command Prompt and `.venv\Scripts\activate.bat`). On macOS/Linux: `source .venv/bin/activate`.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/download_data.py
python -m pytest -q
```
`pytest` should currently report 24 passed (a pgmpy deprecation warning is expected).

To run the notebook: `python -m pip install jupyterlab`, then `jupyter lab` and open the notebook from the `notebooks/` directory (relative paths assume that working directory). Some cells take minutes.

## Contents
- `data/`: the official UCI processed Cleveland file and its provenance
- `src/download_data.py`, `src/data.py`: file verification, loading and the binary target
- `tests/test_data.py`: data checks
- `notebooks/01_phase1_eda_and_classical_models.ipynb`: data description and exploratory analysis
- `results/figures/`: EDA figures
- `src/data.py`: fixed discretisation cut points and the age-adjusted heart-rate feature
- `tests/test_features.py`: discretisation checks
- notebook: signal screening and the documented cut points (`results/phase1_cutpoints.csv`)
- `src/bayes.py`: hand-specified Bayesian network and Naive Bayes (pgmpy)
- `src/classical.py`: logistic regression (binned / continuous), random forest and prior-only baseline
- `tests/test_models.py`: model and preprocessing checks
- `src/metrics.py`, `src/evaluate.py`: metrics and the repeated stratified cross-validation runner
- `tests/test_evaluation.py`: split and saved-prediction checks
- notebook: evaluation protocol, results, calibration and paired comparisons
- `results/`: split manifest, out-of-fold predictions, metrics, paired comparisons

## Data and attribution
Official page: https://archive.ics.uci.edu/dataset/45/heart+disease
Official ZIP: https://archive.ics.uci.edu/static/public/45/heart+disease.zip
Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI. https://doi.org/10.24432/C52P4X. CC BY 4.0.
Only `processed.cleveland.data` from the official archive is used (not an arbitrary Kaggle `heart.csv`, whose coding differs). SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`; `python src/download_data.py` checks it.

## Interpretation notes
Target: `num > 0` (139 positives / 164 negatives), i.e. existing disease status, not future risk. PR AUC is average precision. Sensitivity and specificity use a 0.5 threshold. All models share 5 folds x 5 repeats; each repeat's metrics are computed on the 303 out-of-fold rows and then averaged, and repeated predictions are not independent patients. A non-significant AUC difference is not evidence of equivalence. No claim of clinical readiness or causal discovery is made.
