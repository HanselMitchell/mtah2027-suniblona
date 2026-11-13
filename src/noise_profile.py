"""Aggregate statistics of how incomplete a tracking source is.

Only aggregates are printed -- counts, percentiles, shares -- never positions or
labels. These numbers drive the corruption we apply to clean training data, so
that the model is trained on data that is broken the way the competition data is.
"""

import numpy as np

FPS = 25


def runs(mask):
    """Lengths of maximal True runs in a 1-D boolean array."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def pct(values, ps=(10, 25, 50, 75, 90, 99)):
    if len(values) == 0:
        return "n/a"
    return "  ".join("p{}={:.0f}".format(p, v) for p, v in zip(ps, np.percentile(values, ps)))


def noise_profile(arrays, graded=None):
    """Return a dict of aggregate noise statistics plus a printable text."""
    ball = arrays["ball"]
    n = len(ball)
    graded = np.ones(n, dtype=bool) if graded is None else graded
    out = {}
    lines = []

    # -- ball visibility -------------------------------------------------------------------
    seen = np.isfinite(ball[:, 0])
    out["ball_seen_graded"] = float(seen[graded].mean())
    gaps = []
    # gaps measured inside graded stretches only
    for start, length in _segments(graded):
        gaps.extend(runs(~seen[start:start + length]).tolist())
    gaps = np.array(gaps)
    out["ball_gap_lengths"] = gaps
    lines.append("ball visible on {:.1f}% of graded frames, {} gaps".format(
        100 * out["ball_seen_graded"], len(gaps)))
    lines.append("  gap length (frames): " + pct(gaps))
    buckets = [(1, 5), (5, 25), (25, 125), (125, 500), (500, 10 ** 9)]
    total_missing = max(gaps.sum(), 1)
    for lo, hi in buckets:
        sel = (gaps >= lo) & (gaps < hi)
        lines.append("  gaps {:>4}-{:<5} frames: n={:>5}  share of missing frames {:5.1f}%".format(
            lo, hi if hi < 10 ** 9 else "inf", int(sel.sum()), 100 * gaps[sel].sum() / total_missing))
    vis = runs(seen & graded)
    lines.append("  visible run length (frames): " + pct(vis))

    # -- ball motion / jumps ----------------------------------------------------------------
    both = seen[1:] & seen[:-1] & graded[1:]
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1)[both] * FPS
    out["ball_speed"] = step
    lines.append("ball speed m/s (visible pairs): " + pct(step, (50, 90, 99, 99.9)))
    lines.append("  share of steps > 40 m/s (likely jumps): {:.3f}%".format(
        100 * float(np.mean(step > 40)) if len(step) else 0.0))

    # -- players ----------------------------------------------------------------------------
    for side in ("home", "away"):
        xy = arrays[side]
        present = np.isfinite(xy[:, :, 0])
        count = present.sum(1)[graded]
        out[side + "_count"] = count
        lines.append("{} players/frame: mean {:.2f}  ".format(side, count.mean())
                     + pct(count, (5, 50, 95)))
        frags = []
        for slot in range(present.shape[1]):
            col = present[:, slot] & graded
            frags.extend(runs(col).tolist())
        frags = np.array(frags)
        out[side + "_fragments"] = frags
        lines.append("  {} slots, {} track fragments, length (frames): {}".format(
            present.shape[1], len(frags), pct(frags)))
        moving = present[1:] & present[:-1]
        disp = np.linalg.norm(np.diff(xy, axis=0), axis=2)[moving] * FPS
        lines.append("  player speed m/s: " + pct(disp, (50, 90, 99, 99.9)))

    out["text"] = "\n".join(lines)
    return out


def _segments(mask):
    """(start, length) of True runs."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))
