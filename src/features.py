"""Per-frame features from team-level tracking, identical for IDSSE and the ATD.

Input: the dict from src/tracking_io.py (positions in metres, NaN = missing).
Output: X (T, F) float32 and the list of feature names.

Design choices and where they come from:
  * no player identities -- only team membership (the ATD has none);
  * no absolute x/y -- only symmetric quantities (distance to the nearest line,
    nearest goal line, ...), so the side a team attacks does not matter;
  * ball visibility and gap timing as features: in the ATD a lost ball is mostly
    a stoppage (scripts/gap_context.py);
  * player speeds at several time scales, past and future (Lang et al. 2022);
  * k-nearest players per team to the ball, possession/duel zones
    (Borghesi et al.; Vidal-Codina et al. 2022);
  * restart formations: kick-off, throw-in, corner, goal kick (Vidal-Codina et al.).
"""

import warnings

import numpy as np

FPS = 25
MAX_SPEED = 12.0
WINDOWS = (12, 25, 50, 125, 250, 500)  # frames: 0.5, 1, 2, 5, 10, 20 s


# ---------------------------------------------------------------------------------------
# helpers

def _ffill_idx(valid):
    """Index of the last valid frame at or before t (-1 if none)."""
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx)


def _bfill_idx(valid):
    """Index of the next valid frame at or after t (len if none)."""
    n = len(valid)
    idx = np.where(valid, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _nan_rolling(x, w, side):
    """Mean of x over [t-w, t] (past) or [t, t+w] (future), ignoring NaN."""
    v = np.where(np.isfinite(x), x, 0.0).astype(np.float64)
    c = np.isfinite(x).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    n = len(x)
    t = np.arange(n)
    if side == "past":
        lo, hi = np.maximum(0, t - w), t + 1
    else:
        lo, hi = t, np.minimum(n, t + w + 1)
    s = cv[hi] - cv[lo]
    k = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / k
    out[k == 0] = np.nan
    return out


def _speeds(xy, lag):
    """Per-player speed (m/s) over `lag` frames, NaN where unknown or implausible."""
    sp = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    sp[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=2) * (FPS / lag)
    sp[sp > MAX_SPEED] = np.nan
    return sp


def _nanstat(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


# ---------------------------------------------------------------------------------------

def build_features(arrays):
    ball = arrays["ball"].astype(np.float64)
    home = arrays["home"].astype(np.float64)
    away = arrays["away"].astype(np.float64)
    period = arrays["period"]
    length, width = [float(v) for v in arrays["pitch"]]
    n = len(ball)
    f = {}

    # ---- ball visibility and gaps -----------------------------------------------------
    seen = np.isfinite(ball[:, 0])
    last = _ffill_idx(seen)
    nxt = _bfill_idx(seen)
    t = np.arange(n)
    since = np.where(last >= 0, t - last, 3000).clip(0, 3000)
    until = np.where(nxt < n, nxt - t, 3000).clip(0, 3000)
    f["ball_seen"] = seen.astype(float)
    f["ball_since_seen_s"] = since / FPS
    f["ball_until_seen_s"] = until / FPS
    f["ball_gap_len_s"] = np.where(seen, 0, (since + until)) / FPS
    for w in (125, 500):
        f["ball_seen_frac_past%d" % w] = _nan_rolling(seen.astype(float), w, "past")
        f["ball_seen_frac_fut%d" % w] = _nan_rolling(seen.astype(float), w, "future")

    # effective ball position: observed, else the last known (else the next known)
    eff = ball.copy()
    ok_last = ~seen & (last >= 0)
    eff[ok_last] = ball[last[ok_last]]
    ok_next = ~np.isfinite(eff[:, 0]) & (nxt < n)
    eff[ok_next] = ball[nxt[ok_next]]

    bspeed = np.full(n, np.nan)
    lag = 5
    bspeed[lag:] = np.linalg.norm(ball[lag:] - ball[:-lag], axis=1) * (FPS / lag)
    bspeed[bspeed > 45] = np.nan
    f["ball_speed"] = bspeed
    for w in (12, 50):
        f["ball_speed_past%d" % w] = _nan_rolling(bspeed, w, "past")
        f["ball_speed_fut%d" % w] = _nan_rolling(bspeed, w, "future")

    bx, by = eff[:, 0], eff[:, 1]
    d_side = np.minimum(by, width - by)
    d_goal = np.minimum(bx, length - bx)
    f["ball_d_touch"] = d_side
    f["ball_d_goalline"] = d_goal
    f["ball_d_line"] = np.minimum(d_side, d_goal)
    f["ball_d_center"] = np.hypot(bx - length / 2, by - width / 2)
    f["ball_d_corner"] = np.hypot(d_goal, d_side)
    f["ball_d_halfway"] = np.abs(bx - length / 2)

    # ---- players near the ball, per team -----------------------------------------------
    near_feats = {}
    for side, xy in (("h", home), ("a", away)):
        d = np.linalg.norm(xy - eff[:, None, :], axis=2)
        d_sorted = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
        d_sorted[~np.isfinite(d_sorted)] = np.nan
        for k in range(3):
            f["%s_d%d" % (side, k + 1)] = d_sorted[:, k] if d_sorted.shape[1] > k else np.nan
        for r in (1.0, 2.0, 5.0, 9.15):
            f["%s_n_within%g" % (side, r)] = np.sum(d < r, axis=1).astype(float)
        f["%s_count" % side] = np.isfinite(xy[:, :, 0]).sum(1).astype(float)
        near_feats[side] = d_sorted[:, 0]
    dh, da = near_feats["h"], near_feats["a"]
    f["d1_diff"] = dh - da
    f["d1_logratio"] = np.log((dh + 0.5) / (da + 0.5))
    nearest_home = np.where(np.isfinite(dh) & np.isfinite(da), (dh < da).astype(float), np.nan)
    f["nearest_home"] = nearest_home
    # only counts when someone is really at the ball (possession zone ~1-2 m)
    at_ball = np.minimum(np.nan_to_num(dh, nan=99), np.nan_to_num(da, nan=99)) < 2.0
    owner_sig = np.where(at_ball & seen, nearest_home, np.nan)
    f["owner_sig"] = owner_sig
    for w in WINDOWS:
        f["nearest_home_past%d" % w] = _nan_rolling(nearest_home, w, "past")
        f["nearest_home_fut%d" % w] = _nan_rolling(nearest_home, w, "future")
        f["owner_sig_past%d" % w] = _nan_rolling(owner_sig, w, "past")
        f["owner_sig_fut%d" % w] = _nan_rolling(owner_sig, w, "future")
        f["d1_diff_past%d" % w] = _nan_rolling(np.clip(f["d1_diff"], -20, 20), w, "past")
        f["d1_diff_fut%d" % w] = _nan_rolling(np.clip(f["d1_diff"], -20, 20), w, "future")
    # time since the last frame anyone was at the ball, and who
    at_idx = _ffill_idx(at_ball & seen)
    f["since_at_ball_s"] = np.where(at_idx >= 0, t - at_idx, 3000).clip(0, 3000) / FPS
    nxt_at = _bfill_idx(at_ball & seen)
    f["until_at_ball_s"] = np.where(nxt_at < n, nxt_at - t, 3000).clip(0, 3000) / FPS
    lh = np.where(at_idx >= 0, nearest_home[np.maximum(at_idx, 0)], np.nan)
    nh = np.where(nxt_at < n, nearest_home[np.minimum(nxt_at, n - 1)], np.nan)
    f["last_at_ball_home"] = lh
    f["next_at_ball_home"] = nh

    # ---- player movement ----------------------------------------------------------------
    allxy = np.concatenate([home, away], axis=1)
    sp = _speeds(allxy, FPS)
    sp_short = _speeds(allxy, 5)
    f["spd_med"] = _nanstat(np.nanmedian, sp)
    f["spd_mean"] = _nanstat(np.nanmean, sp)
    f["spd_p90"] = _nanstat(lambda a, axis: np.nanpercentile(a, 90, axis=axis), sp)
    f["spd_frac_fast"] = _nanstat(np.nanmean, (sp > 3).astype(float) + np.where(np.isnan(sp), np.nan, 0))
    f["spd_frac_still"] = _nanstat(np.nanmean, (sp < 0.7).astype(float) + np.where(np.isnan(sp), np.nan, 0))
    f["spd_short_med"] = _nanstat(np.nanmedian, sp_short)
    nh_ = home.shape[1]
    f["spd_med_h"] = _nanstat(np.nanmedian, sp[:, :nh_])
    f["spd_med_a"] = _nanstat(np.nanmedian, sp[:, nh_:])
    for w in WINDOWS:
        f["spd_med_past%d" % w] = _nan_rolling(f["spd_med"], w, "past")
        f["spd_med_fut%d" % w] = _nan_rolling(f["spd_med"], w, "future")
        f["spd_fast_past%d" % w] = _nan_rolling(f["spd_frac_fast"], w, "past")
        f["spd_fast_fut%d" % w] = _nan_rolling(f["spd_frac_fast"], w, "future")
    # acceleration of the whole group: stoppages start with a collective slowdown
    f["spd_change_1s"] = f["spd_med_fut25"] - f["spd_med_past25"]
    f["spd_change_5s"] = f["spd_med_fut125"] - f["spd_med_past125"]

    # ---- team shape ---------------------------------------------------------------------
    cen = {}
    for side, xy in (("h", home), ("a", away)):
        cx = _nanstat(np.nanmean, xy[:, :, 0])
        cy = _nanstat(np.nanmean, xy[:, :, 1])
        cen[side] = (cx, cy)
        f["%s_spread_x" % side] = _nanstat(np.nanstd, xy[:, :, 0])
        f["%s_spread_y" % side] = _nanstat(np.nanstd, xy[:, :, 1])
        f["%s_ball_to_centroid" % side] = np.hypot(bx - cx, by - cy)
    f["centroid_dist"] = np.hypot(cen["h"][0] - cen["a"][0], cen["h"][1] - cen["a"][1])
    f["ball_centroid_diff"] = f["h_ball_to_centroid"] - f["a_ball_to_centroid"]
    # which side of the ball each block sits on, along the pitch (sign-free)
    f["ball_between_blocks"] = ((bx - cen["h"][0]) * (bx - cen["a"][0]) < 0).astype(float)

    # ---- restart formations -----------------------------------------------------------
    x, y = allxy[:, :, 0], allxy[:, :, 1]
    with np.errstate(invalid="ignore"):
        f["n_beyond_touch"] = np.sum((y < -0.3) | (y > width + 0.3), axis=1).astype(float)
        f["n_beyond_goalline"] = np.sum((x < -0.3) | (x > length + 0.3), axis=1).astype(float)
        cd = np.minimum.reduce([np.hypot(x - cx_, y - cy_) for cx_ in (0, length)
                                for cy_ in (0, width)])
        f["n_near_corner"] = np.sum(cd < 3, axis=1).astype(float)
        in_goal_area = ((x < 5.5) | (x > length - 5.5)) & (np.abs(y - width / 2) < 9.16)
        f["n_goal_area"] = np.sum(in_goal_area, axis=1).astype(float)
        f["n_near_center"] = np.sum(np.hypot(x - length / 2, y - width / 2) < 3, axis=1).astype(float)
        f["n_in_box_any"] = np.sum(((x < 16.5) | (x > length - 16.5))
                                   & (np.abs(y - width / 2) < 20.16), axis=1).astype(float)
        # kick-off shape: each team entirely on one side of the halfway line
        ko = []
        for xy in (home, away):
            left = np.sum(xy[:, :, 0] < length / 2 + 1, axis=1)
            right = np.sum(xy[:, :, 0] > length / 2 - 1, axis=1)
            cnt = np.isfinite(xy[:, :, 0]).sum(1)
            ko.append(np.maximum(left, right) / np.maximum(cnt, 1))
        f["kickoff_shape"] = np.minimum(ko[0], ko[1])
        # players giving room around the ball (free kicks): nobody between 2 and 9 m
        dall = np.linalg.norm(allxy - eff[:, None, :], axis=2)
        f["n_ring_2_9"] = np.sum((dall > 2) & (dall < 9.15), axis=1).astype(float)
        f["n_within1_all"] = np.sum(dall < 1, axis=1).astype(float)
    for name in ("n_beyond_touch", "n_near_corner", "n_goal_area", "kickoff_shape", "n_ring_2_9"):
        f[name + "_past125"] = _nan_rolling(f[name], 125, "past")
        f[name + "_fut125"] = _nan_rolling(f[name], 125, "future")

    # ---- time within the period ---------------------------------------------------------
    first = np.zeros(n, dtype=np.int64)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        first[idx] = idx[0]
    f["period_time_s"] = (t - first) / FPS

    names = list(f.keys())
    X = np.empty((n, len(names)), dtype=np.float32)
    for j, k in enumerate(names):
        X[:, j] = f[k]
    return X, names
