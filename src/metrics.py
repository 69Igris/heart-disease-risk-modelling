"""The metrics used to compare models. Inputs: y = true labels (0/1), p = predicted P(disease)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression


def logit(p):
    """Log-odds of a probability. Clipped so that exactly 0 or 1 does not give infinity."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    """Expected calibration error.

    Sort patients by predicted risk, cut them into `bins` groups of equal size, and in each
    group compare the average predicted risk with the share of patients who really had disease.
    ECE is the size-weighted average of those gaps (0 = perfectly calibrated).
    """
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    total = 0.0
    for g in groups:
        gap = abs(y[g].mean() - p[g].mean())
        total += len(g) / len(y) * gap
    return float(total)


def cal_slope_intercept(y, p):
    """Calibration slope and intercept: fit  logit(P(y=1)) = intercept + slope * logit(p).

    slope = 1 and intercept = 0 means the probabilities are right.
    slope < 1: predictions are too extreme (over-confident).  slope > 1: too compressed.
    A huge C means "practically no regularisation", so this is a plain logistic fit.
    """
    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def all_metrics(y, p, thr=0.5, full=True):
    """Return a dict of metrics. Sensitivity/specificity use the threshold `thr`.
    full=False skips log loss and the calibration slope (used inside the bootstrap for speed)."""
    y = np.asarray(y)
    p = np.asarray(p)
    predicted_positive = p >= thr

    tp = ((predicted_positive == 1) & (y == 1)).sum()   # sick, flagged
    fn = ((predicted_positive == 0) & (y == 1)).sum()   # sick, missed
    tn = ((predicted_positive == 0) & (y == 0)).sum()   # healthy, cleared
    fp = ((predicted_positive == 1) & (y == 0)).sum()   # healthy, flagged

    m = {
        "ROC AUC": roc_auc_score(y, p),                 # how well patients are ranked
        "PR AUC": average_precision_score(y, p),        # average precision
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "Brier": brier_score_loss(y, p),                # mean squared probability error
        "ECE": ece(y, p),
    }
    if full:
        m["Log loss"] = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        m["Cal. slope"], m["Cal. intercept"] = cal_slope_intercept(y, p)
    return m
