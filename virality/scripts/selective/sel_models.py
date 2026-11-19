"""Selective track: the deferral models (fit, predict, export to the JSON that xviral/selective.py evaluates)."""
from __future__ import annotations

import numpy as np

SHORT = ["gap", "gap_sd", "agree", "o_gap", "z_ovl", "width_pick", "width_other", "h_like_sd", "h_n", "log_followers", "level", "level_vs_hist",
         "cos_ab", "abs_logwords", "d_media", "any_media", "sim_centroid_diff", "gap_over_hsd", "gap_rel", "gap_level", "days_apart", "kind_launch"]
TINY = ["gap", "o_gap", "h_like_sd", "level_vs_hist", "log_followers", "cos_ab", "gap_level", "kind_launch"]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def fit_scale(gap: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, bias: bool = False) -> dict:
    """The current rule: P(right) = sigmoid(k * gap) (no intercept: a zero gap is a coin flip), by weighted maximum likelihood."""
    w = np.ones(len(y)) if w is None else w
    ks = np.exp(np.linspace(np.log(0.05), np.log(60), 400))
    p = sigmoid(ks[:, None] * gap[None])
    ll = (w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum(1)
    return {"type": "logistic_gap", "scale": float(ks[np.argmax(ll)]), "bias": 0.0}


def platt(z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e4).fit(z[:, None], y, sample_weight=w)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


class Model:
    """kind: 'scale' | 'linear' | 'gbm' | 'mlp'. fit(X frame as dict of arrays) -> self; raw(X) -> uncalibrated logit; spec() -> JSON."""

    def __init__(self, kind: str, feats: list[str], **hp):
        self.kind, self.feats, self.hp = kind, feats, hp

    def _x(self, X: dict, fit: bool = False) -> np.ndarray:
        A = np.column_stack([np.asarray(X[k], float) for k in self.feats])
        if self.kind == "gbm":
            return A
        if fit:
            self.imp = np.nanmedian(A, 0)
            self.imp = np.where(np.isnan(self.imp), 0.0, self.imp)
        A = np.where(np.isnan(A), self.imp[None], A)
        if fit:
            self.mu, self.sd = A.mean(0), A.std(0) + 1e-6
        return (A - self.mu) / self.sd

    def fit(self, X: dict, y: np.ndarray, w: np.ndarray):
        if self.kind == "scale":
            self.s = fit_scale(np.asarray(X["gap"], float), y, w)
            return self
        A = self._x(X, fit=True)
        if self.kind == "linear":
            from sklearn.linear_model import LogisticRegression
            self.m = LogisticRegression(C=self.hp.get("C", 0.1), max_iter=2000).fit(A, y, sample_weight=w)
        elif self.kind == "gbm":
            import lightgbm as lgb
            hp = {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 4, "max_depth": 2, "min_child_samples": 60, "subsample": 0.7, "subsample_freq": 1,
                  "colsample_bytree": 0.5, "reg_lambda": 20.0, "verbose": -1, "n_jobs": 2, "random_state": 0, **self.hp}
            mono = [1 if k in ("gap", "z_ovl") else 0 for k in self.feats] if hp.pop("monotone", True) else None
            self.m = lgb.LGBMClassifier(monotone_constraints=mono, **hp).fit(A, y, sample_weight=w)
        elif self.kind == "mlp":
            import torch
            Xt, yt, wt = torch.tensor(A, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(w / w.mean(), dtype=torch.float32)
            self.nets = []
            for seed in range(self.hp.get("seeds", 5)):
                torch.manual_seed(seed)
                h = self.hp.get("hidden", 8)
                net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], h), torch.nn.Tanh(), torch.nn.Linear(h, 1))
                opt = torch.optim.AdamW(net.parameters(), lr=self.hp.get("lr", 0.01), weight_decay=self.hp.get("wd", 0.3))
                for _ in range(self.hp.get("epochs", 150)):
                    opt.zero_grad()
                    drop = torch.nn.functional.dropout(Xt, self.hp.get("drop", 0.1))
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(drop)[:, 0], yt, reduction="none") * wt).mean()
                    loss.backward()
                    opt.step()
                self.nets.append(net.eval())
        return self

    def raw(self, X: dict) -> np.ndarray:
        if self.kind == "scale":
            return self.s["scale"] * np.asarray(X["gap"], float)
        A = self._x(X)
        if self.kind == "linear":
            return self.m.decision_function(A)
        if self.kind == "gbm":
            return self.m.predict(A, raw_score=True)
        import torch
        with torch.no_grad():
            return np.mean([n(torch.tensor(A, dtype=torch.float32))[:, 0].numpy() for n in self.nets], 0)

    def spec(self, calibration: dict | None = None) -> dict:
        if self.kind == "scale":
            return dict(self.s)
        base = {"features": self.feats, "calibration": calibration}
        if self.kind == "gbm":
            dump = self.m.booster_.dump_model()

            def conv(n):
                if "leaf_value" in n:
                    return {"leaf_value": n["leaf_value"]}
                assert n["decision_type"] == "<="
                return {"split_feature": n["split_feature"], "threshold": n["threshold"], "default_left": bool(n["default_left"]),
                        "left_child": conv(n["left_child"]), "right_child": conv(n["right_child"])}
            return {"type": "gbm", **base, "init": 0.0, "trees": [conv(t["tree_structure"]) for t in dump["tree_info"]]}
        base |= {"impute": self.imp.tolist(), "mean": self.mu.tolist(), "std": self.sd.tolist(), "max_missing": 0.34}
        if self.kind == "linear":
            return {"type": "linear", **base, "coef": self.m.coef_[0].tolist(), "intercept": float(self.m.intercept_[0])}
        return {"type": "mlp", **base, "nets": [{"w1": n[0].weight.detach().numpy().tolist(), "b1": n[0].bias.detach().numpy().tolist(),
                                                 "w2": n[2].weight.detach().numpy()[0].tolist(), "b2": float(n[2].bias.detach().numpy()[0])} for n in self.nets]}


def weights(d: np.ndarray, how: str) -> np.ndarray:
    a = np.abs(d)
    if how == "flat":
        return np.ones(len(d))
    if how == "gap":
        return np.clip(a, np.log(1.25), np.log(5)) / np.log(5)
    if how == "2x":
        return (a >= np.log(2)).astype(float) + 1e-9
    raise ValueError(how)


def folds(auth: np.ndarray, k: int, seed: int) -> np.ndarray:
    ua = np.unique(auth)
    rng = np.random.default_rng(seed)
    f = dict(zip(ua, rng.permutation(len(ua)) % k))
    return np.array([f[a] for a in auth])


def fit_calibrated(make, X: dict, y, w, auth, inner: int = 4, seed: int = 0):
    """Fit on everything; Platt scale fitted on out-of-fold raw scores (accounts held out). The scale rule needs none."""
    m = make().fit(X, y, w)
    if m.kind == "scale":
        return m, None
    fo = folds(auth, inner, seed + 100)
    z = np.zeros(len(y))
    for j in range(inner):
        tr, te = fo != j, fo == j
        z[te] = make().fit({k: np.asarray(v)[tr] for k, v in X.items()}, y[tr], w[tr]).raw({k: np.asarray(v)[te] for k, v in X.items()})
    return m, platt(z, y, w)


def predict(m: Model, cal: dict | None, X: dict) -> np.ndarray:
    z = m.raw(X)
    return sigmoid(z) if cal is None else sigmoid(cal["a"] * z + cal["b"])


HIST = ["ovl_hist", "ovl_mix", "entropy"]      # need the members' full histograms; kept apart so their value can be read off


def all_features(X: dict) -> list[str]:
    return [c for c in X if not np.all(np.isnan(X[c])) and np.nanstd(X[c]) > 0 and c not in HIST]


def configs(ALL: list[str], HIST: list[str]) -> dict:
    """Every configuration the validation compares: name -> constructor."""
    CFG = {"scale": lambda: Model("scale", ["gap"])}
    for C in (0.01, 0.1):
        CFG[f"linear_tiny_C{C}"] = lambda C=C: Model("linear", TINY, C=C)
        CFG[f"linear_short_C{C}"] = lambda C=C: Model("linear", SHORT, C=C)
    CFG["linear_all_C0.01"] = lambda: Model("linear", ALL, C=0.01)
    CFG["gbm_all"] = lambda: Model("gbm", ALL)
    CFG["gbm_all_hist"] = lambda: Model("gbm", ALL + HIST)
    CFG["linear_tiny_hist_C0.1"] = lambda: Model("linear", TINY + ["ovl_hist"], C=0.1)
    CFG["gbm_short"] = lambda: Model("gbm", SHORT)
    CFG["gbm_tiny"] = lambda: Model("gbm", TINY, colsample_bytree=1.0)
    CFG["gbm_all_60"] = lambda: Model("gbm", ALL, n_estimators=60)
    CFG["gbm_all_deep"] = lambda: Model("gbm", ALL, n_estimators=200, num_leaves=8, max_depth=3, min_child_samples=40)
    CFG["gbm_all_free"] = lambda: Model("gbm", ALL, monotone=False)
    CFG["mlp_short"] = lambda: Model("mlp", SHORT, seeds=3)
    CFG["mlp_tiny"] = lambda: Model("mlp", TINY, seeds=3)
    CFG["mlp_all_wd1"] = lambda: Model("mlp", ALL, seeds=3, wd=1.0, drop=0.2)
    return CFG
