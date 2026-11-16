"""Temporal stacking: a second gradient-boosting stage that reads the stage-1
probabilities of the surrounding frames.

Stage 1 sees one frame (plus hand-made window features). Stage 2 sees how the
stage-1 belief evolves before and after the frame -- rolling means, minima and
maxima of p(alive) and p(home | alive) at 0.5-20 s, past and future, and the
differences between the two sides. A change of possession is a frame where the
past and future beliefs disagree, which is exactly what these features express.

Trained on out-of-fold stage-1 probabilities (leave-one-match-out on IDSSE), so
stage 2 learns how much to trust stage 1 on unseen matches.
"""

import numpy as np

from src.features import _nan_rolling, _rolling_extreme

WINDOWS = (6, 12, 25, 50, 125, 250, 500)
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 400,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "verbose": -1, "seed": 2027, "deterministic": True, "force_row_wise": True,
}
ROUNDS = 300


def stack_features(proba, period):
    """proba (T, 3) H/A/D -> (T, F) features, computed inside each period."""
    out = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out.append((idx, _one_period(proba[idx])))
    F = np.empty((len(proba), out[0][1].shape[1]), dtype=np.float32)
    for idx, f in out:
        F[idx] = f
    return F


def _one_period(pr):
    p_alive = 1.0 - pr[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p_home = np.where(p_alive > 1e-6, pr[:, 0] / np.maximum(p_alive, 1e-6), 0.5)
    cols = [p_alive, p_home, pr[:, 0], pr[:, 1]]
    for x in (p_alive, p_home):
        for w in WINDOWS:
            past = _nan_rolling(x, w, "past")
            fut = _nan_rolling(x, w, "future")
            cols += [past, fut, fut - past]
        for w in (25, 125):
            cols += [_rolling_extreme(x, w, "past", np.nanmin), _rolling_extreme(x, w, "future", np.nanmin),
                     _rolling_extreme(x, w, "past", np.nanmax), _rolling_extreme(x, w, "future", np.nanmax)]
    return np.stack(cols, axis=1).astype(np.float32)


def train(F_list, ya_list, yh_list, threads=8):
    import lightgbm as lgb
    F = np.concatenate(F_list)
    ya = np.concatenate(ya_list)
    yh = np.concatenate(yh_list)
    p = dict(PARAMS, num_threads=threads)
    m_a = lgb.train(p, lgb.Dataset(F, ya), num_boost_round=ROUNDS)
    sel = yh >= 0
    m_h = lgb.train(p, lgb.Dataset(F[sel], yh[sel]), num_boost_round=ROUNDS)
    return m_a, m_h


def predict(models, F):
    pa = models[0].predict(F)
    ph = models[1].predict(F)
    return np.stack([pa * ph, pa * (1 - ph), 1 - pa], axis=1)
