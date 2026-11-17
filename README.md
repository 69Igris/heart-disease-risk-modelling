# Heart Disease Risk Modelling - Phase 1 and Phase 2
Project 15. UCI processed Cleveland data.
- **Phase 1 (Advanced ML):** Bayesian network, discrete Naive Bayes, logistic regression and random forest.
- **Phase 2 (Deep Learning):** a regularised PyTorch MLP compared with the Bayesian network on the same 25 splits, with calibration and SHAP.

**Prediction exercise only, not clinical decision support.**

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
`pytest` should currently report 33 passed (26 Phase 1 + 7 Phase 2; a pgmpy deprecation warning is expected).

To re-run everything non-interactively (bootstrap cells take minutes):

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 01_phase1_eda_and_classical_models.ipynb
cd ..
python src/build_latex_tables.py
python -m pytest -q
```
The notebook must be run from `notebooks/`. Compile the report as described in `report/README.md`.

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
- notebook: ablations, prior-strength sensitivity and exact Bayesian-network explanations
- `report/`: IEEE-format LaTeX report (`main.tex`, `references.bib`) and compiled `phase1_report.pdf`; see `report/README.md`
- `presentation/`: 12-slide deck (PPTX and PDF)
- `src/build_latex_tables.py`: writes the report tables from `results/*.csv`
- `tests/test_results.py`: explanation-output and scope checks

## Phase 2 (Deep Learning)
Phase 2 adds a regularised MLP (`src/mlp.py`, PyTorch). Size, weight decay, dropout and number of epochs are chosen by nested cross-validation, and the final model averages 5 seeds. Explanations use Kernel SHAP in log-odds (`src/explain.py`). Everything runs in `notebooks/02_phase2_mlp_and_comparison.ipynb` on the same 25 saved splits, reusing the saved Phase 1 predictions.

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 02_phase2_mlp_and_comparison.ipynb
cd ..
python src/build_phase2_tables.py
python -m pytest -q
```
The notebook takes about 4 minutes on a laptop CPU.
- **Outputs:** `results/phase2_*.csv`, `results/phase2_oof_predictions.csv.gz`, `results/figures/fig_p2_*.png`.
- **Report:** separate IEEE report in `report/phase2/` (`phase2_report.pdf`).
- **Slides:** 16-slide deck in `presentation/phase2_deck.pptx` and `phase2_deck.pdf`.
- **Walkthrough:** `docs/PHASE2_WALKTHROUGH.md`.
- **Tests:** `tests/test_mlp.py`.

Headline result: the MLP ranks no better than the Bayesian network (AUC 0.911 vs 0.896, paired interval includes 0). Its probabilities are much better calibrated (slope 1.00 vs 0.49; Brier 0.117 vs 0.139, interval excludes 0). It behaves almost exactly like logistic regression on the same inputs.

## Data and attribution
Official page: https://archive.ics.uci.edu/dataset/45/heart+disease
Official ZIP: https://archive.ics.uci.edu/static/public/45/heart+disease.zip
Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI. https://doi.org/10.24432/C52P4X. CC BY 4.0.
Only `processed.cleveland.data` from the official archive is used (not an arbitrary Kaggle `heart.csv`, whose coding differs). SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`; `python src/download_data.py` checks it.

## Interpretation notes
Target: `num > 0` (139 positives / 164 negatives), i.e. existing disease status, not future risk. PR AUC is average precision. Sensitivity and specificity use a 0.5 threshold. All models share 5 folds x 5 repeats; each repeat's metrics are computed on the 303 out-of-fold rows and then averaged, and repeated predictions are not independent patients. A non-significant AUC difference is not evidence of equivalence. No claim of clinical readiness or causal discovery is made.

## AI assistance
Parts of the code, analysis, report and slides were drafted with AI assistance (Claude, Anthropic). The team is responsible for the contents; see the disclosure in the report and the Git history for who committed what.
