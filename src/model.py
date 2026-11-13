"""Two-stage LightGBM (Borghesi et al.): stage 1 alive vs dead, stage 2 home vs away
on alive frames. Both trained on corrupted IDSSE; deterministic by construction."""

import numpy as np

SEED = 2027
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
    "num_threads": 4,
}
ROUNDS = 400


def make_labels(arrays):
    """y_alive (1/0) and y_home (1 home, 0 away, -1 not applicable)."""
    alive = (arrays["alive"] == 1).astype(np.int8)
    home = np.where(alive == 1, (arrays["owner"] == 0).astype(np.int8), -1)
    return alive, home


def train(X, y_alive, y_home, rounds=ROUNDS):
    import lightgbm as lgb

    m_alive = lgb.train(PARAMS, lgb.Dataset(X, y_alive), num_boost_round=rounds)
    sel = y_home >= 0
    m_home = lgb.train(PARAMS, lgb.Dataset(X[sel], y_home[sel]), num_boost_round=rounds)
    return m_alive, m_home


def predict_proba(models, X):
    """(T, 3) probabilities for H, A, D."""
    m_alive, m_home = models
    p_alive = m_alive.predict(X)
    p_home = m_home.predict(X)
    return np.stack([p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive], axis=1)


def save(models, prefix):
    models[0].save_model(prefix + "_alive.txt")
    models[1].save_model(prefix + "_home.txt")


def load(prefix):
    import lightgbm as lgb

    return (lgb.Booster(model_file=prefix + "_alive.txt"),
            lgb.Booster(model_file=prefix + "_home.txt"))
