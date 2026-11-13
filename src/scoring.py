"""Score a decoded label sequence with the competition's own metric library
(src/possession_metrics.py, copied unchanged from metrica-sports/more-than-a-hack-2027)."""

import numpy as np

from src import possession_metrics as pm

PAD = 200  # X frames inserted between periods, like the competition's halftime hole


def to_codes(labels, period):
    """labels 0/1/2 (H/A/D) per frame -> metric codes with an X gap between periods."""
    parts = []
    for i, p in enumerate(np.unique(period)):
        if i:
            parts.append(np.full(PAD, pm.X, dtype=np.int8))
        parts.append(labels[period == p].astype(np.int8))
    return np.concatenate(parts)


def gt_labels(arrays):
    alive = arrays["alive"] == 1
    return np.where(alive, np.where(arrays["owner"] == 0, 0, 1), 2).astype(np.int8)


def score(gt, pred, period):
    r = pm.evaluate(to_codes(gt, period), to_codes(pred, period))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "n_gt": r.get("n_gt_spells", None), "raw": r}


def baseline_nearest(arrays, window=50):
    """The competition baseline: nearest tracked player's team, D when the ball is
    not tracked, then a centred majority vote of +/- window frames."""
    ball = arrays["ball"]
    seen = np.isfinite(ball[:, 0])
    code = np.full(len(ball), -1, dtype=np.int8)
    dh = np.nanmin(np.linalg.norm(arrays["home"] - ball[:, None], axis=2), axis=1)
    da = np.nanmin(np.linalg.norm(arrays["away"] - ball[:, None], axis=2), axis=1)
    ok = seen & np.isfinite(dh) & np.isfinite(da)
    code[ok] = np.where(dh[ok] < da[ok], 0, 1)
    hc = np.concatenate(([0], np.cumsum(code == 0)))
    ac = np.concatenate(([0], np.cumsum(code == 1)))
    idx = np.arange(len(code))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(code), idx + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    return np.where(h + a == 0, 2, np.where(h >= a, 0, 1)).astype(np.int8)
