"""Phase 2: regularised feed-forward neural network (MLP) in PyTorch.

Same contract as the Phase 1 wrappers in classical.py:
    fit(X, y)          learn from the training rows only (preprocessing included)
    predict_proba(X)   return P(disease) for each row

Inputs: scaled numeric columns + one-hot categorical columns (the exact preprocessor that
Phase 1 logistic regression uses), so the only thing that changes is the model.

Regularisation: L2 weight decay (AdamW), dropout, a small network, early stopping chosen
by inner cross-validation, and an average over a few random initialisations.

Tuning is NESTED: inside every outer training fold, a 3-fold inner CV picks the
architecture, weight decay, dropout AND the number of epochs. The outer held-out fold is
never seen while tuning, so the outer metrics stay honest.
"""
import itertools
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from classical import make_preprocessor

torch.set_num_threads(1)          # tiny networks: one thread is faster and deterministic

# Small search space, fixed before looking at outer-fold results.
GRID = {
    "hidden": [(16,), (32, 16)],
    "weight_decay": [1e-3, 1e-2, 1e-1],
    "dropout": [0.1, 0.3],
}
MAX_EPOCHS = 300
LR = 3e-3


def grid():
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


class Net(nn.Module):
    """Linear -> ReLU -> Dropout blocks, then one output logit."""

    def __init__(self, d_in, hidden=(16,), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x).squeeze(-1)


def _tensor(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


def train_net(Xt, y, cfg, epochs, seed, Xv=None, yv=None):
    """Full-batch AdamW training for a fixed number of epochs.

    If a validation set is given, also return the validation log loss after every epoch
    (used by the inner CV to choose the number of epochs)."""
    torch.manual_seed(seed)
    net = Net(Xt.shape[1], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    # classes are close to balanced (46% positive), so the plain log loss is used
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = _tensor(Xt), _tensor(y)
    curve = []
    if Xv is not None:
        xv, yv_t = _tensor(Xv), _tensor(yv)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward(); opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                curve.append(float(loss_fn(net(xv), yv_t)))
    net.eval()
    return net, np.array(curve)


def net_logits(nets, X):
    """Average logit over an ensemble of nets (averaging logits keeps the output a single model)."""
    with torch.no_grad():
        x = _tensor(X)
        return np.mean([n(x).numpy() for n in nets], axis=0)


class MLP:
    """Regularised MLP with nested hyper-parameter and epoch selection.

    mode      "continuous" (default, scaled numbers + one-hot) or "discretized"
              (one-hot of the same bins the Bayesian network uses; an ablation)
    features  optional subset of columns (e.g. without ca/thal)
    n_seeds   number of random initialisations averaged in the final model
    tune      if False, use `config` and `epochs` as given (no inner CV)
    """

    def __init__(self, mode="continuous", features=None, seed=0, n_seeds=5, inner_folds=3,
                 tune=True, config=None, epochs=150):
        self.mode, self.features, self.seed = mode, features, seed
        self.n_seeds, self.inner_folds, self.tune = n_seeds, inner_folds, tune
        self.config = config or {"hidden": (16,), "weight_decay": 1e-2, "dropout": 0.3}
        self.epochs = epochs

    def _select(self, X, y):
        """Inner CV: average validation-loss curves over inner folds; pick (config, epoch) with the lowest loss."""
        skf = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        inner = list(skf.split(X, y))
        best = (np.inf, None, None)
        self.search_ = []
        for cfg in grid():
            curves = []
            for k, (tr, va) in enumerate(inner):
                pre = make_preprocessor(self.mode, self.features)   # refit preprocessing inside the inner fold too
                Xt = pre.fit_transform(X.iloc[tr]); Xv = pre.transform(X.iloc[va])
                _, c = train_net(Xt, y[tr], cfg, MAX_EPOCHS, self.seed * 100 + k, Xv, y[va])
                curves.append(c)
            mean_curve = np.mean(curves, axis=0)
            e = int(np.argmin(mean_curve))
            self.search_.append({**cfg, "epochs": e + 1, "inner_logloss": float(mean_curve[e])})
            if mean_curve[e] < best[0]:
                best = (float(mean_curve[e]), cfg, e + 1)
        self.inner_logloss_, self.config, self.epochs = best
        return self

    def fit(self, X, y):
        y = np.asarray(y)
        if self.tune:
            self._select(X, y)
        self.pre = make_preprocessor(self.mode, self.features)
        Xt = self.pre.fit_transform(X)                               # training fold only
        self.nets = [train_net(Xt, y, self.config, self.epochs, self.seed * 1000 + i)[0]
                     for i in range(self.n_seeds)]
        return self

    def decision_function(self, X):
        """Log-odds of disease (useful for SHAP in log-odds units, comparable to BN weights of evidence)."""
        return net_logits(self.nets, self.pre.transform(X))

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))

    def n_params(self):
        return sum(p.numel() for p in self.nets[0].parameters())
