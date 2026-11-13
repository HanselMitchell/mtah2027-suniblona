"""Viterbi decoding of per-frame H/A/D probabilities into possessions.

A per-frame argmax flickers, and flicker is exactly what the ranking metric
(possession-level F1 over temporal IoU) punishes. Viterbi picks the single label
sequence that best fits the probabilities while paying a fixed cost for every
change of state, so a change only happens where the evidence for it outweighs
the cost -- and then at the frame where it is best supported, not shifted by a
smoothing window.

States: 0 = H, 1 = A, 2 = D.
Costs (in log-probability units):
  c_ha  H <-> A    a change of possession in open play
  c_to_d  H/A -> D   the ball goes dead
  c_from_d  D -> H/A   a restart
"""

import numpy as np

DEFAULT = {"c_ha": 25.0, "c_to_d": 25.0, "c_from_d": 25.0, "floor": 1e-4}


def viterbi(proba, c_ha, c_to_d, c_from_d, floor=1e-4, **_):
    n = len(proba)
    logp = np.log(np.clip(proba, floor, 1.0))
    cost = np.array([[0.0, c_ha, c_to_d],
                     [c_ha, 0.0, c_to_d],
                     [c_from_d, c_from_d, 0.0]])
    back = np.empty((n, 3), dtype=np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost          # cand[i, j]: from i to j
        best = np.argmax(cand, axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_periods(proba, period, params):
    """Run Viterbi separately inside each period (no possession crosses halftime)."""
    out = np.empty(len(proba), dtype=np.int8)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = viterbi(proba[idx], **params)
    return out
