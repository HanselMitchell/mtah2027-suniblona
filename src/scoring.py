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


# ---------------------------------------------------------------------------------------
# Fast path for tuning: identical numbers to pm.spell_scores / pm.macro_f1 (checked in
# scripts/experiment.py --selftest), but overlaps are found with a sweep instead of the
# O(n_gt * n_pred) double loop.

def _spells(codes, runs):
    return [(a, b, k) for a, b, k in pm.segments(codes, runs) if k in pm.POSSESSION_LABELS]


def fast_spell_f1(gt_codes, pred_codes):
    runs = pm.valid_runs(gt_codes)
    gs = _spells(gt_codes, runs)
    ps = _spells(pred_codes, runs)
    if not gs or not ps:
        return 0.0
    p_start = np.array([p[0] for p in ps])
    p_stop = np.array([p[1] for p in ps])
    overlaps = []
    for i, (a, b, k) in enumerate(gs):
        lo = np.searchsorted(p_stop, a, side="right")   # first pred with stop > a
        hi = np.searchsorted(p_start, b, side="left")   # preds with start < b
        for j in range(lo, hi):
            if ps[j][2] != k:
                continue
            inter = min(b, p_stop[j]) - max(a, p_start[j])
            union = max(b, p_stop[j]) - min(a, p_start[j])
            overlaps.append((inter / float(union), i, j))
    overlaps.sort(key=lambda item: (-item[0], item[1], item[2]))
    f1s = []
    for thr in pm.SPELL_IOUS:
        tp = int(pm.matched_gt_flags(overlaps, len(gs), thr).sum())
        fp, fn = len(ps) - tp, len(gs) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def fast_macro_f1(gt_codes, pred_codes):
    cm = pm.confusion(gt_codes, pred_codes)
    return float(pm.macro_f1(pm.scored_confusion(cm)))


def fast_score(gt, pred, period):
    g, p = to_codes(gt, period), to_codes(pred, period)
    return {"spellF1": fast_spell_f1(g, p), "macroF1": fast_macro_f1(g, p)}
