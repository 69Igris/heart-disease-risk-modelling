# How the code works (five-minute version)

## The whole idea in one loop
```python
for train, test in splits:                          # 25 splits: 5 folds x 5 repeats
    model.fit(X.iloc[train], y[train])              # learn from the training rows only
    p[test] = model.predict_proba(X.iloc[test])     # predict the rows the model never saw
# then: compare p with y using AUC, Brier, calibration, ...
```
Everything else in the project is either (a) a different `model`, (b) a different way to measure `p` against `y`, or (c) a check that this loop is fair.

## Which file does what
| File | One-line job |
|---|---|
| `src/download_data.py` | Check the data file against the official UCI archive (SHA-256). |
| `src/data.py` | Load the file, define the target (`num > 0`), fixed bins, age-adjusted heart-rate feature. |
| `src/bayes.py` | Hand-drawn Bayesian network and Naive Bayes (pgmpy). `fit`, `predict_proba`, and `contributions` (exact log-odds per variable). |
| `src/classical.py` | Logistic regression (binned or continuous inputs), random forest, and a "predict the average" baseline. |
| `src/metrics.py` | The numbers: ROC AUC, PR AUC, sensitivity, specificity, Brier, ECE, calibration slope. |
| `src/evaluate.py` | `run_cv` runs the loop; `summarize` adds bootstrap intervals; `paired` compares two models. |
| `src/build_latex_tables.py` | Turns the result CSVs into report tables (no typed numbers). |
| `tests/` | Small checks that the data, bins, models and saved results behave as claimed. |

## The models in plain words
- **Prior only:** ignores the patient and always predicts the training prevalence (about 46%). Anything useful must beat it.
- **Naive Bayes:** every finding is treated as independent evidence given disease.
- **Bayesian network:** like Naive Bayes, but with a hand-drawn graph (16 edges) so some variables depend on each other and the risk factors point into disease.
- **Logistic regression:** a weighted sum of the inputs turned into a probability. Used with bins (same inputs as the networks) and with continuous numbers.
- **Random forest:** 300 decision trees voting; can pick up interactions without being told.

## Why preprocessing lives inside `fit`
Filling missing values, scaling and one-hot encoding learn numbers from the data (a median, a mean, a spread). If they were learned on all 303 rows, the test fold would leak into training. So each model wrapper owns its preprocessing and fits it only on the training fold.

## The "extras" in one sentence each
- **5 repeats:** with only 303 patients one split is noisy; repeating with different shuffles gives a steadier average.
- **Out-of-fold (OOF):** every patient is predicted by a model that never saw them.
- **Bootstrap interval:** resample the 303 patients 500 times and recompute the metric to see how much it could move.
- **Corrected t-test:** the 25 folds overlap in training data, so the ordinary t-test is too optimistic; the correction widens it. It is exploratory, and a non-significant result is not proof that two models are equal.
- **Calibration slope / ECE / Brier:** AUC only checks ranking. These check whether "0.9" really means about 9 in 10.
- **Matched heart-rate control:** a logistic regression with the same age-adjusted heart-rate feature but no binning, so "binning" is not confused with "different feature".
- **Ablations (no `ca`/`thal`, prior strength):** check whether the conclusions survive removing two work-up variables or changing the Bayesian smoothing.

## Say-it-out-loud summaries (for the viva)
1. "We predict existing heart disease on 303 Cleveland patients and compare a Bayesian network, Naive Bayes, logistic regression and a random forest."
2. "All models see the same 25 splits and fit preprocessing on training folds only."
3. "Ranking (AUC about 0.90 to 0.91) is similar across models; probability quality differs: Naive Bayes and the network are over-confident, logistic regression is close to calibrated."
4. "Phase 2 will add a neural network and compare it on the same splits."
