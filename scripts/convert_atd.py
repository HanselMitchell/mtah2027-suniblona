"""Step 3: convert the competition ATD (tactical) to our array format, and print
an aggregate noise profile.

    python scripts/convert_atd.py                 # uses $MTAH_DATA_DIR
    python scripts/convert_atd.py --data-dir "E:\\path\\to\\bundle"

Reads   <data-dir>/tactical/BAR-ATH_tactical_atd.txt
        <data-dir>/tactical/BAR-ATH_tactical_atd.xml   (optional, see src/atd_raw.py)
        <data-dir>/broadcast/BAR-ATH_broadcast_atd.xml (only if the tactical XML is missing)
Writes  data/atd/BAR-ATH_tactical.npz   (stays on your machine -- gitignored)

The printed profile contains only aggregates (percentiles, counts, shares).

Frame alignment: index i of every array is video frame i (= submission row i),
because ATD frame_id = video_frame + 1 (official starter/load_atd.py).
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import atd_raw  # noqa: E402
from src.noise_profile import noise_profile  # noqa: E402
from src.match import N_FRAMES, PITCH, HALVES, graded_mask  # noqa: E402

OUT = os.path.join(ROOT, "data", "atd", "BAR-ATH_tactical.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"),
                        help="folder that contains tactical/ and broadcast/")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    if not args.data_dir:
        sys.exit("set MTAH_DATA_DIR or pass --data-dir")

    raw = os.path.join(args.data_dir, "tactical", "BAR-ATH_tactical_atd.txt")
    meta = os.path.join(args.data_dir, "tactical", "BAR-ATH_tactical_atd.xml")
    meta_b = os.path.join(args.data_dir, "broadcast", "BAR-ATH_broadcast_atd.xml")
    if not os.path.isfile(raw):
        sys.exit("not found: {}".format(raw))

    start = time.time()
    print("parsing ATD ...", flush=True)
    fid, players, ball = atd_raw.parse_raw(raw)
    video = fid - 1
    if video.min() < 0 or video.max() >= N_FRAMES or len(np.unique(video)) != len(video):
        sys.exit("unexpected frame ids {}..{}".format(fid.min(), fid.max()))
    graded = graded_mask()

    if os.path.isfile(meta):
        teams = atd_raw.teams_from_xml(meta)
        print("slot teams from {}".format(os.path.basename(meta)))
    else:
        full = np.full((N_FRAMES,) + players.shape[1:], np.nan, dtype=np.float32)
        full[video] = players
        teams, k = atd_raw.infer_teams(full, graded)
        l2r = atd_raw.home_attacks_left_to_right(meta_b) if os.path.isfile(meta_b) else True
        first = np.zeros(N_FRAMES, dtype=bool)
        first[HALVES[0][1]:HALVES[0][2] + 1] = True
        ok = atd_raw.check_orientation(full, teams, first, l2r)
        print("tactical XML missing: inferred home = slots 0..{}, away = {}..{} "
              "(orientation check {})".format(k - 1, k, len(teams) - 1, "OK" if ok else "FAILED"))
        if not ok:
            sys.exit("home/away orientation check failed -- refusing to guess")

    length, width = PITCH
    n = N_FRAMES
    arrays = {
        "frame_id": np.arange(n, dtype=np.int64) + 1,
        "period": np.full(n, -1, dtype=np.int8),
        "ball": np.full((n, 2), np.nan, dtype=np.float32),
        "home": np.full((n, int((teams == 0).sum()), 2), np.nan, dtype=np.float32),
        "away": np.full((n, int((teams == 1).sum()), 2), np.nan, dtype=np.float32),
        "alive": np.full(n, -1, dtype=np.int8),
        "owner": np.full(n, -1, dtype=np.int8),
        "pitch": np.array([length, width], dtype=np.float32),
    }
    scale = np.array([length, width], dtype=np.float32)
    arrays["ball"][video] = ball * scale
    arrays["home"][video] = players[:, teams == 0] * scale
    arrays["away"][video] = players[:, teams == 1] * scale

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("saved {} in {:.0f} s\n".format(os.path.relpath(args.out, ROOT), time.time() - start))

    profile = noise_profile(arrays, graded)
    print("=== NOISE PROFILE (aggregates only - safe to share) ===")
    print(profile["text"])


if __name__ == "__main__":
    main()
