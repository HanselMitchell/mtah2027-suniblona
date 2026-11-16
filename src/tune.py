"""Coordinate-descent tuning of the Viterbi decoder on held-out IDSSE probabilities.

Objective: mean official spellF1 over the given matches (macroF1 breaks ties).
"""

import numpy as np

from src.decode import decode_periods
from src.scoring import fast_score

DECODE_INIT = {"c_ha": 8.0, "c_to_d": 10.0, "c_from_d": 25.0, "floor": 1e-4,
               "min_poss": 1, "min_dead": 1}
GRID = {
    "c_ha": [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0],
    "c_to_d": [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0],
    "c_from_d": [4.0, 6.0, 8.0, 12.0, 16.0, 25.0, 40.0, 60.0],
    "min_poss": [1, 10, 20, 30, 45, 60, 90],
    "min_dead": [1, 25, 50, 75, 100, 150, 250],
    "floor": [1e-6, 1e-4, 1e-3, 1e-2],
}


def evaluate(probs, gts, periods, matches, params):
    rows = [fast_score(gts[m], decode_periods(probs[m], periods[m], params), periods[m])
            for m in matches]
    return (float(np.mean([r["spellF1"] for r in rows])),
            float(np.mean([r["macroF1"] for r in rows])))


def tune_decode(probs, gts, periods, matches, init, grid=GRID, passes=3, verbose=False):
    best = dict(init)
    best_score = evaluate(probs, gts, periods, matches, best)
    cache = {}
    for _ in range(passes):
        improved = False
        for key, values in grid.items():
            for v in values:
                cand = dict(best, **{key: v})
                sig = tuple(sorted(cand.items()))
                if sig not in cache:
                    cache[sig] = evaluate(probs, gts, periods, matches, cand)
                if cache[sig] > best_score:
                    best, best_score, improved = cand, cache[sig], True
                    if verbose:
                        print("   ", best_score, best, flush=True)
        if not improved:
            break
    return best, best_score
