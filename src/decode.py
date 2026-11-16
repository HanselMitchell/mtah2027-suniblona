"""Viterbi decoding of per-frame H/A/D probabilities into possessions.

A per-frame argmax flickers, and flicker is exactly what the ranking metric
(possession-level F1 over temporal IoU) punishes. Viterbi picks the single label
sequence that best fits the probabilities while paying a fixed cost for every
change of state, so a change only happens where the evidence for it outweighs
the cost -- and then at the frame where it is best supported, not shifted by a
smoothing window.

On top of the switching costs, every segment has a *minimum duration*
(`min_poss` frames for H/A, `min_dead` frames for D). This is a semi-Markov
constraint: a 0.3 s "possession" in the middle of a pass is impossible rather
than merely expensive, which removes the short false splits that cost two
possessions each under spellF1, without raising the switching cost so much that
real, quick changes of possession are lost.

The recursion is written over segment *entries* rather than an expanded state
space, so it costs O(T * 3) regardless of the minimum durations:

    E_c[t]  best score of a path whose segment of class c starts at t
    S_c[t]  best score of a path in class c at t whose current segment has
            already lasted >= K_c frames
    S_c[t] = max(S_c[t-1] + lp_c[t],  E_c[t-K_c+1] + sum(lp_c[t-K_c+2 .. t]))
    E_c[t] = max_{c' != c} S_c'[t-1] - cost(c' -> c) + lp_c[t]

States: 0 = H, 1 = A, 2 = D.
Costs (in log-probability units):
  c_ha      H <-> A    a change of possession in open play
  c_to_d    H/A -> D   the ball goes dead
  c_from_d  D -> H/A   a restart
"""

import numpy as np
from numba import njit

DEFAULT = {"c_ha": 25.0, "c_to_d": 25.0, "c_from_d": 25.0, "floor": 1e-4,
           "min_poss": 1, "min_dead": 1}


@njit(cache=True)
def _viterbi_md(lp, cost, K):
    n = lp.shape[0]
    NEG = -1e18
    cl = np.zeros((n + 1, 3))
    for t in range(n):
        for c in range(3):
            cl[t + 1, c] = cl[t, c] + lp[t, c]
    E = np.full((n, 3), NEG)
    S = np.full((n, 3), NEG)
    e_from = np.full((n, 3), -1, np.int8)
    s_stay = np.zeros((n, 3), np.bool_)
    for c in range(3):
        E[0, c] = lp[0, c]
        S[0, c] = lp[0, c]          # the first segment may be shorter than K
        s_stay[0, c] = True         # marks "start of sequence"
    for t in range(1, n):
        for c in range(3):
            best, arg = NEG, -1
            for p in range(3):
                if p == c:
                    continue
                v = S[t - 1, p] - cost[p, c]
                if v > best:
                    best, arg = v, p
            E[t, c] = best + lp[t, c]
            e_from[t, c] = arg
        for c in range(3):
            stay = S[t - 1, c] + lp[t, c]
            s0 = t - K[c] + 1
            chain = NEG
            if s0 >= 0:
                chain = E[s0, c] + cl[t + 1, c] - cl[s0 + 1, c]
            if stay >= chain:
                S[t, c] = stay
                s_stay[t, c] = True
            else:
                S[t, c] = chain
    path = np.empty(n, np.int8)
    c = 0
    for k in range(1, 3):
        if S[n - 1, k] > S[n - 1, c]:
            c = k
    t = n - 1
    while t >= 0:
        if s_stay[t, c]:
            path[t] = c
            t -= 1
        else:
            s0 = t - K[c] + 1
            for u in range(s0, t + 1):
                path[u] = c
            if s0 == 0:
                break
            c = e_from[s0, c]
            t = s0 - 1
    return path


def viterbi(proba, c_ha, c_to_d, c_from_d, floor=1e-4, min_poss=1, min_dead=1, **_):
    logp = np.log(np.clip(np.asarray(proba, dtype=np.float64), floor, 1.0))
    cost = np.array([[0.0, c_ha, c_to_d],
                     [c_ha, 0.0, c_to_d],
                     [c_from_d, c_from_d, 0.0]])
    K = np.array([max(1, int(min_poss)), max(1, int(min_poss)), max(1, int(min_dead))],
                 dtype=np.int64)
    return _viterbi_md(logp, cost, K)


def decode_periods(proba, period, params):
    """Run Viterbi separately inside each period (no possession crosses halftime)."""
    out = np.empty(len(proba), dtype=np.int8)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = viterbi(proba[idx], **params)
    return out
