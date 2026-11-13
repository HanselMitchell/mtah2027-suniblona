"""Step 3b: what is happening on the pitch while the ATD has lost the ball?

No labels are used on the competition match. We compare how players move during
ATD ball gaps with how they move during ALIVE and DEAD play in IDSSE (where the
ball state is labelled), to estimate how many ATD gaps are dead-ball stoppages.

    python scripts/gap_context.py

Reads data/atd/BAR-ATH_tactical.npz and data/idsse/*.npz.
Prints aggregates only -- safe to share.
"""

import glob
import os
import sys

import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.match import graded_mask  # noqa: E402
from src.noise_profile import _segments  # noqa: E402

FPS = 25
MAX_SPEED = 12.0  # m/s; faster steps are identity switches / jumps, not running


def team_speed(arrays):
    """Per-frame median speed (m/s) over all tracked players, NaN-safe.

    Uses 1-second displacement per slot (less jitter than frame-to-frame)."""
    xy = np.concatenate([arrays["home"], arrays["away"]], axis=1)
    lag = FPS
    disp = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    disp[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=2) / (lag / FPS)
    disp[disp > MAX_SPEED] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(disp, axis=1)


def boundary_distance(ball, pitch):
    """Distance (m) from ball to the nearest pitch line; negative = outside."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    return np.minimum.reduce([x, length - x, y, width - y])


def pct(v, ps=(10, 25, 50, 75, 90)):
    v = v[np.isfinite(v)]
    if not len(v):
        return "n/a"
    return "  ".join("p{}={:.2f}".format(p, q) for p, q in zip(ps, np.percentile(v, ps)))


def main():
    # ---- IDSSE: the reference, with labels ---------------------------------------------
    alive_speed, dead_speed = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "idsse", "*.npz"))):
        a = dict(np.load(path))
        s = team_speed(a)
        alive_speed.append(s[a["alive"] == 1])
        dead_speed.append(s[a["alive"] == 0])
    alive_speed = np.concatenate(alive_speed)
    dead_speed = np.concatenate(dead_speed)
    print("IDSSE median player speed (m/s)")
    print("  ALIVE frames: " + pct(alive_speed))
    print("  DEAD  frames: " + pct(dead_speed))

    # threshold that best separates alive/dead in IDSSE (unsupervised w.r.t. the ATD)
    grid = np.linspace(0.3, 4.0, 75)
    acc = [((alive_speed > g).mean() + (dead_speed <= g).mean()) / 2 for g in grid]
    thr = float(grid[int(np.argmax(acc))])
    p_slow_alive = float((alive_speed[np.isfinite(alive_speed)] <= thr).mean())
    p_slow_dead = float((dead_speed[np.isfinite(dead_speed)] <= thr).mean())
    print("  'slow' threshold {:.2f} m/s: slow on {:.1f}% of ALIVE, {:.1f}% of DEAD frames".format(
        thr, 100 * p_slow_alive, 100 * p_slow_dead))

    # ---- ATD: unlabelled ----------------------------------------------------------------
    atd_path = os.path.join(ROOT, "data", "atd", "BAR-ATH_tactical.npz")
    a = dict(np.load(atd_path))
    graded = graded_mask(len(a["ball"]))
    s = team_speed(a)
    seen = np.isfinite(a["ball"][:, 0])
    print("\nATD median player speed (m/s), graded frames")
    print("  ball VISIBLE: " + pct(s[graded & seen]))
    print("  ball MISSING: " + pct(s[graded & ~seen]))

    def slow_share(mask):
        v = s[mask]
        v = v[np.isfinite(v)]
        return float((v <= thr).mean()) if len(v) else float("nan")

    for name, mask in (("visible", graded & seen), ("missing", graded & ~seen)):
        q = slow_share(mask)
        # mixture: q = f*p_dead + (1-f)*p_alive  ->  f = share that behaves like DEAD
        f = (q - p_slow_alive) / (p_slow_dead - p_slow_alive)
        print("  ball {}: slow on {:.1f}% of frames -> est. {:.0f}% dead-like".format(
            name, 100 * q, 100 * np.clip(f, 0, 1)))

    # per-gap view: long gaps only, classify each gap by its median speed
    pitch = tuple(a["pitch"].tolist())
    bd = boundary_distance(a["ball"], pitch)
    gap_rows = []
    for start, length in _segments(graded & ~seen):
        if length < 25:
            continue
        before = start - 1
        last_bd = bd[before] if before >= 0 and seen[before] else np.nan
        gap_rows.append((length, np.nanmedian(s[start:start + length]), last_bd))
    gap_rows = np.array(gap_rows, dtype=float)
    if len(gap_rows):
        slow = gap_rows[:, 1] <= thr
        print("\nATD ball gaps >= 1 s: n={}".format(len(gap_rows)))
        print("  gaps whose median speed is 'slow' (dead-like): {} ({:.0f}%), "
              "covering {:.0f}% of missing frames".format(
                  int(slow.sum()), 100 * slow.mean(),
                  100 * gap_rows[slow, 0].sum() / gap_rows[:, 0].sum()))
        print("  gap length s, slow gaps: " + pct(gap_rows[slow, 0] / FPS))
        print("  gap length s, fast gaps: " + pct(gap_rows[~slow, 0] / FPS))
        near = gap_rows[:, 2] <= 3.0
        print("  ball last seen within 3 m of / beyond a line: {:.0f}% of gaps "
              "(slow gaps {:.0f}%, fast gaps {:.0f}%)".format(
                  100 * np.nanmean(near), 100 * np.nanmean(near[slow]) if slow.any() else 0,
                  100 * np.nanmean(near[~slow]) if (~slow).any() else 0))


if __name__ == "__main__":
    main()
