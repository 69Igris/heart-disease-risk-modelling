import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def ece(y, p, bins=10):
    """Expected calibration error with equal-frequency bins."""
    order = np.argsort(p); chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in chunks))

def cal_slope_intercept(y, p):
    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def all_metrics(y, p, thr=0.5, full=True):
    y = np.asarray(y); p = np.asarray(p); pred = p >= thr
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    m = {"ROC AUC": roc_auc_score(y, p), "PR AUC": average_precision_score(y, p),
         "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
         "Brier": brier_score_loss(y, p), "ECE": ece(y, p)}
    if full:
        m["Log loss"] = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        m["Cal. slope"], m["Cal. intercept"] = cal_slope_intercept(y, p)
    return m
