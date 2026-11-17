# Phase 2: how the neural network part works (five-minute version)

Phase 1 is explained in [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). Phase 2 adds one model, the MLP, and compares it with the Phase 1 models. It uses the same data, target and **25 splits**.

## The whole idea in one loop
```python
for train, test in splits:                       # the SAME 25 splits as Phase 1
    mlp = MLP(seed=s)                            # fresh, untrained
    mlp.fit(X.iloc[train], y[train])             # inside: inner CV picks size, decay, dropout, epochs
    p[test] = mlp.predict_proba(X.iloc[test])    # predict patients the model never saw
# Phase 1 predictions are LOADED from results/phase1_oof_predictions.csv.gz, not refitted
# then: same metrics, same bootstrap, same paired tests as Phase 1
```

## Which file does what
| File | One-line job |
|---|---|
| `src/mlp.py` | The PyTorch MLP (`Net`), training (`train_net`) and the `MLP` wrapper with nested tuning. |
| `src/explain.py` | `mlp_shap`: Kernel SHAP for the MLP on the 13 original variables, in log-odds. |
| `notebooks/02_phase2_mlp_and_comparison.ipynb` | Runs everything and writes `results/phase2_*` and `results/figures/fig_p2_*`. |
| `src/build_phase2_tables.py` | Turns the Phase 2 CSVs into the LaTeX tables in `report/phase2/tables/`. |
| `tests/test_mlp.py` | Checks for leakage, repeatability, valid probabilities, tuning, SHAP additivity and the saved results. |

## The MLP in plain words
- **Inputs:** the same as Phase 1 logistic regression. The 5 numbers are filled with the training median and standardised. The 8 categories are filled with the most common training value and one-hot encoded. That gives 28 columns. All of this is learned from the training fold only.
- **Network:** 28 inputs, then 16 hidden units (ReLU), dropout, and 1 output. The output is the log-odds of disease. A two-layer version (32, 16) was also allowed.
- **Training:** binary cross-entropy (log loss), full-batch AdamW, learning rate 0.003.
- **Keeping it from overfitting 242 rows:**
  - weight decay (an L2 penalty);
  - dropout;
  - a small network;
  - stopping early;
  - averaging 5 networks started from different random seeds.
- **Nested tuning:** inside each training fold, a 3-fold *inner* CV tries 12 settings (2 sizes × 3 weight decays × 2 dropouts). For each setting it records the validation loss after every epoch. The best setting and epoch count (lowest average inner log loss) are used to refit on the whole training fold. The test fold is never used for tuning.

## The two extra MLPs (ablations)
- **MLP (binned):** given the *exact bins* the Bayesian network uses. If it is still well calibrated, the bins are not what made the BN over-confident.
- **MLP (no ca/thal):** the same model without the two diagnostic work-up tests, mirroring the Phase 1 ablation.

## SHAP vs the BN explanation
- The BN already has an *exact* explanation. Its log-odds equal a risk-factor term plus one term per finding (Phase 1, `BNClassifier.contributions`).
- For the MLP we use **Kernel SHAP** on its log-odds, so both explanations are on the same scale.
- Each patient is explained by the fold model that did not see them (repeat 0).
- The background is 40 real training rows. Real rows are used rather than averages, because an average of category codes, such as `cp = 3.4`, means nothing.
- Check: SHAP values + base value = MLP log-odds, to within 5e-7.

## Results in four sentences (numbers from `results/phase2_metrics.csv` and `phase2_paired.csv`)
1. **Ranking is a tie:** the MLP's AUC is 0.911, against the BN's 0.896. MLP − BN = +0.015, with an interval of [−0.001, +0.031] that includes 0.
2. **Probabilities are much better:** the MLP's calibration slope is 1.00 against the BN's 0.49. Its Brier score is 0.117 against 0.139, a difference of −0.022 [−0.036, −0.007] that excludes 0. The MLP is confidently wrong for 5 patients; the BN is for 21.
3. **The BN's problem is its structure, not its bins:** the MLP given the BN's bins is just as calibrated. The BN multiplies correlated findings as if they were independent.
4. **No evidence for nonlinear interactions at n = 303:**
   - The MLP equals logistic regression on the same inputs (ΔAUC +0.000).
   - Tuning picked the small network in 22 of 25 folds.
   - The SHAP plots are almost straight-line.

## Say-it-out-loud summaries (for the viva)
1. "Phase 2 adds a regularised PyTorch MLP and compares it with the Phase 1 Bayesian network on the same 25 splits, reusing the saved Phase 1 predictions."
2. "The MLP is tuned with nested cross-validation, so its scores are not optimistically selected."
3. "It ranks patients no better than the BN, but its probabilities are far better calibrated."
4. "Giving the MLP the BN's own bins changes nothing, so the BN's over-confidence comes from its independence assumptions, not discretisation."
5. "The MLP behaves almost like logistic regression; at 303 patients there is no detectable nonlinear interaction to learn."
6. "SHAP and the BN's exact weights agree on what matters (ca, thal, chest pain, exercise findings), but the BN's weights are about twice as large."
7. "This is a course prediction exercise, not a clinical tool."

## Running it
```bash
python -m pip install -r requirements.txt            # adds torch and shap
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 02_phase2_mlp_and_comparison.ipynb
cd ..
python src/build_phase2_tables.py
python -m pytest -q                                  # 33 passed
```
The notebook takes about 4 minutes on a laptop CPU. All seeds are fixed, so a re-run reproduces the saved results; tiny differences are possible on different hardware or library versions.
