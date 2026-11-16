"""Make clean, labelled tracking (IDSSE) look like the competition's automatic ATD.

Every number here comes from the aggregate noise profile of the ATD
(scripts/convert_atd.py, scripts/gap_context.py), never from its labels:

  * the ball is missing on ~41% of graded frames, in long gaps (median ~11 s,
    99.7% of missing frames in gaps > 5 s), and ~97% of missing frames behave
    like dead ball (players as slow as in IDSSE dead-ball time);
  * a minority of gaps (~10% of missing frames) happen with players moving fast,
    i.e. during live play, typically 5-13 s long;
  * ~10.2 of 11 players per team are tracked, in fragments of median ~30 s;
  * small position jitter, and ~0.2% of ball steps are implausible jumps;
  * (version 2) the ATD ball trajectory is temporally smoothed by the provider:
    the change of ball velocity across 3 frames has p90 ~2.2 m/s and p99 ~4.2 m/s
    on graded ATD frames, against 6.3 / 17.9 in IDSSE. A Gaussian smoothing of the
    clean ball with sigma ~ U(5, 7) frames reproduces those percentiles
    (0.71/1.36/2.53/5.1 at sigma 5, 0.67/1.27/2.04/3.76 at sigma 7, vs ATD
    0.68/1.40/2.16/4.21 for p50/p75/p90/p99), while ball speed is preserved.

Parameters are drawn from ranges rather than fixed, so the model does not learn
one exact corruption.
"""

import numpy as np

FPS = 25


def _runs(mask):
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _smooth_noise(rng, n, dims, sigma, knot=2 * FPS):
    """Gaussian noise drawn every `knot` frames and linearly interpolated."""
    k = n // knot + 2
    knots = rng.normal(0, sigma, size=(k, dims))
    grid = np.arange(n) / knot
    return np.stack([np.interp(grid, np.arange(k), knots[:, d]) for d in range(dims)],
                    axis=1).astype(np.float32)


def _gauss_smooth(x, sigma, period):
    """Gaussian smoothing of (T, 2) positions, separately within each period."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = x.copy()
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        for d in range(x.shape[1]):
            col = x[idx, d].astype(np.float64)
            out[idx, d] = np.convolve(np.pad(col, r, mode="edge"), k, mode="valid")
    return out


def corrupt(arrays, seed, version=2):
    rng = np.random.default_rng(seed)
    out = dict(arrays)
    ball = arrays["ball"].copy()
    if version >= 2:
        ball = _gauss_smooth(ball, rng.uniform(5.0, 7.0), arrays["period"])
    alive = arrays["alive"]
    period = arrays["period"]
    n = len(ball)
    hide = np.zeros(n, dtype=bool)

    # ---- ball missing during dead ball --------------------------------------------------
    p_dead_run = rng.uniform(0.8, 0.95)          # share of stoppages where the ball is lost
    starts, ends = _runs(alive == 0)
    for s, e in zip(starts, ends):
        if rng.random() > p_dead_run:
            continue
        length = e - s
        a = rng.uniform(-0.05, 0.25) * length    # lost a bit before/after the stoppage starts
        b = rng.uniform(-0.05, 0.25) * length    # found a bit before/after the restart
        lo = int(max(0, s + a))
        hi = int(min(n, e - b))
        if hi > lo:
            hide[lo:hi] = True

    # ---- ball missing during live play (long gaps, players still moving) --------------
    live_idx = np.flatnonzero(alive == 1)
    if len(live_idx):
        target = rng.uniform(0.015, 0.05) * len(live_idx)
        hidden = 0
        guard = 0
        while hidden < target and guard < 10000:
            guard += 1
            centre = live_idx[rng.integers(len(live_idx))]
            length = int(rng.uniform(5, 13) * FPS)
            lo, hi = centre, min(n, centre + length)
            if period[lo] != period[hi - 1]:
                continue
            hidden += int((~hide[lo:hi]).sum())
            hide[lo:hi] = True

    # a few very short dropouts, as in the ATD
    for _ in range(rng.integers(5, 25)):
        c = rng.integers(n)
        hide[c:c + rng.integers(1, 25)] = True

    if version >= 3:
        # (v3) the ATD practically never sees the ball beyond the lines (4.2% of visible graded
        # frames vs 16% in v2-corrupted IDSSE, whose DFL feed parks a dead ball outside)
        length, width = [float(v) for v in arrays["pitch"]]
        bx, by = ball[:, 0], ball[:, 1]
        d_line = np.minimum.reduce([bx, length - bx, by, width - by])
        hide |= np.isfinite(d_line) & (d_line < -rng.uniform(0.3, 1.5))
        # (v3) the ATD re-acquires a lost ball once it is struck: after a gap of >= 1 s the
        # first visible step has median speed ~12 m/s (p25 5, p75 17) vs ~0.4 m/s in v2.
        # Extend each such gap until the (clean) ball starts moving.
        spd = np.zeros(n)
        spd[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
        moving = spd > rng.uniform(3.0, 6.0)
        starts, ends = _runs(hide)
        cap = int(15 * FPS)
        for s, e in zip(starts, ends):
            if e - s < FPS or e >= n:
                continue
            nxt = np.flatnonzero(moving[e:min(n, e + cap)])
            stop = e + (nxt[0] if len(nxt) else 0)
            if period[min(stop, n - 1)] == period[s]:
                hide[e:stop] = True

    ball[hide] = np.nan
    # slowly varying position error (white noise would inflate speeds) and rare jumps
    ball += _smooth_noise(rng, n, 2, sigma=0.3)
    seen = np.isfinite(ball[:, 0])
    jumps = seen & (rng.random(n) < rng.uniform(0.0005, 0.0015))
    ball[jumps] += rng.normal(0, 6, size=(int(jumps.sum()), 2)).astype(np.float32)
    out["ball"] = ball

    # ---- players: fragmented tracks, occasional bad stretches, jitter ------------------
    bad = np.zeros(n, dtype=bool)
    for _ in range(int(n / (FPS * 120) * rng.uniform(0.5, 1.5))):
        c = rng.integers(n)
        bad[c:c + int(rng.uniform(2, 10) * FPS)] = True
    for side in ("home", "away"):
        xy = arrays[side].copy()
        present = np.isfinite(xy[:, :, 0])
        mean_on = rng.uniform(900, 2500)
        mean_off = rng.uniform(40, 130)
        for p in range(xy.shape[1]):
            if not present[:, p].any():
                continue
            drop = np.zeros(n, dtype=bool)
            t = int(rng.exponential(mean_on))
            while t < n:
                off = max(1, int(rng.exponential(mean_off)))
                drop[t:t + off] = True
                t += off + max(1, int(rng.exponential(mean_on)))
            for s0, e0 in zip(*_runs(bad)):
                if rng.random() < 0.35:
                    drop[s0:e0] = True
            xy[drop, p] = np.nan
            xy[:, p] += _smooth_noise(rng, n, 2, sigma=0.2)
        out[side] = xy
    return out
