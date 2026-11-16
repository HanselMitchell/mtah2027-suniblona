"""Two-stage gradient boosting (Borghesi et al.): stage 1 alive vs dead, stage 2 home
vs away on alive frames. Both trained on corrupted IDSSE; deterministic by construction.

Backends: LightGBM (default), XGBoost, CatBoost -- all CPU, fixed seeds.
"""

import os

import numpy as np

SEED = 2027
THREADS = int(os.environ.get("MTAH_THREADS", "8"))
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": THREADS,
}
ROUNDS = 400

XGB_PARAMS = {
    "objective": "binary:logistic", "eta": 0.05, "max_depth": 7, "min_child_weight": 50,
    "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "tree_method": "hist",
    "seed": SEED, "nthread": THREADS, "eval_metric": "logloss",
}
CAT_PARAMS = {
    "loss_function": "Logloss", "learning_rate": 0.08, "depth": 7, "l2_leaf_reg": 3.0,
    "random_seed": SEED, "thread_count": THREADS, "verbose": 0, "border_count": 128,
}


def make_labels(arrays):
    """y_alive (1/0) and y_home (1 home, 0 away, -1 not applicable)."""
    alive = (arrays["alive"] == 1).astype(np.int8)
    home = np.where(alive == 1, (arrays["owner"] == 0).astype(np.int8), -1)
    return alive, home


class _Booster:
    """Uniform predict() over the three libraries, plus save/load."""

    def __init__(self, backend, model):
        self.backend, self.model = backend, model

    def predict(self, X):
        if self.backend == "lgb":
            return self.model.predict(X)
        if self.backend == "xgb":
            import xgboost as xgb
            return self.model.predict(xgb.DMatrix(X))
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        if self.backend == "lgb":
            self.model.save_model(path)
        elif self.backend == "xgb":
            self.model.save_model(path)
        else:
            self.model.save_model(path, format="cbm")


def _fit(backend, X, y, params, rounds):
    if backend == "lgb":
        import lightgbm as lgb
        p = dict(PARAMS, **params)
        r = int(p.pop("rounds", rounds))
        return _Booster("lgb", lgb.train(p, lgb.Dataset(X, y), num_boost_round=r))
    if backend == "xgb":
        import xgboost as xgb
        p = dict(XGB_PARAMS, **params)
        r = int(p.pop("rounds", rounds))
        return _Booster("xgb", xgb.train(p, xgb.DMatrix(X, y), num_boost_round=r))
    from catboost import CatBoostClassifier
    p = dict(CAT_PARAMS, **params)
    r = int(p.pop("rounds", rounds))
    m = CatBoostClassifier(iterations=r, **p)
    m.fit(X, y)
    return _Booster("cat", m)


def train(X, y_alive, y_home, rounds=ROUNDS, backend="lgb", params=None):
    params = params or {}
    m_alive = _fit(backend, X, y_alive, params, rounds)
    sel = y_home >= 0
    m_home = _fit(backend, X[sel], y_home[sel], params, rounds)
    return m_alive, m_home


def predict_proba(models, X):
    """(T, 3) probabilities for H, A, D."""
    m_alive, m_home = models
    p_alive = m_alive.predict(X)
    p_home = m_home.predict(X)
    return np.stack([p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive], axis=1)


EXT = {"lgb": ".txt", "xgb": ".json", "cat": ".cbm"}


def save(models, prefix, backend="lgb"):
    models[0].save(prefix + "_alive" + EXT[backend])
    models[1].save(prefix + "_home" + EXT[backend])


def load(prefix, backend="lgb"):
    paths = (prefix + "_alive" + EXT[backend], prefix + "_home" + EXT[backend])
    if backend == "lgb":
        import lightgbm as lgb
        return tuple(_Booster("lgb", lgb.Booster(model_file=p)) for p in paths)
    if backend == "xgb":
        import xgboost as xgb
        out = []
        for p in paths:
            b = xgb.Booster()
            b.load_model(p)
            out.append(_Booster("xgb", b))
        return tuple(out)
    from catboost import CatBoostClassifier
    out = []
    for p in paths:
        m = CatBoostClassifier()
        m.load_model(p, format="cbm")
        out.append(_Booster("cat", m))
    return tuple(out)
