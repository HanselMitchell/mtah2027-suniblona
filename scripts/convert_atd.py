"""Step 3: convert the competition ATD (tactical) to our array format, and print
an aggregate noise profile.

    python scripts/convert_atd.py --data-dir "C:\\Users\\Rainer Yesaya\\dataset barcelona"

Reads   <data-dir>/tactical/BAR-ATH_tactical_atd.xml and _atd.txt
Writes  data/atd/BAR-ATH_tactical.npz   (stays on your machine -- gitignored)

The printed profile contains only aggregates (percentiles, counts, shares).
That is the only part to share. Never share the .npz or the raw files.

Frame alignment: index i of every array is video frame i (= submission row i),
because ATD frame_id = video_frame + 1 (see DATA.md of the competition repo).
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.tracking_io import dataset_to_arrays  # noqa: E402
from src.noise_profile import noise_profile  # noqa: E402
from src.match import N_FRAMES, PITCH, graded_mask  # noqa: E402

OUT = os.path.join(ROOT, "data", "atd", "BAR-ATH_tactical.npz")


def align_to_video(arrays, n_frames):
    """Reindex every per-frame array so that index == video frame."""
    video = arrays["frame_id"] - 1
    if video.min() < 0 or video.max() >= n_frames:
        raise ValueError("frame ids out of range: {}..{}".format(video.min() + 1, video.max() + 1))
    if len(np.unique(video)) != len(video):
        raise ValueError("duplicate frame ids in ATD")
    out = {"pitch": arrays["pitch"]}
    for key, value in arrays.items():
        if key == "pitch":
            continue
        if value.dtype.kind == "f":
            full = np.full((n_frames,) + value.shape[1:], np.nan, dtype=value.dtype)
        else:
            full = np.full((n_frames,) + value.shape[1:], -1, dtype=value.dtype)
        full[video] = value
        out[key] = full
    out["frame_id"] = np.arange(n_frames) + 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"),
                        help="folder that contains tactical/ and broadcast/")
    parser.add_argument("--meta", help="override: path to the ATD .xml")
    parser.add_argument("--raw", help="override: path to the ATD .txt")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--no-align", action="store_true", help="for testing on other data")
    args = parser.parse_args()

    meta = args.meta or os.path.join(args.data_dir, "tactical", "BAR-ATH_tactical_atd.xml")
    raw = args.raw or os.path.join(args.data_dir, "tactical", "BAR-ATH_tactical_atd.txt")
    for path in (meta, raw):
        if not os.path.isfile(path):
            sys.exit("not found: {}".format(path))

    from kloppy import metrica

    start = time.time()
    print("parsing ATD (about a minute) ...", flush=True)
    dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
    arrays = dataset_to_arrays(dataset, pitch=PITCH)
    if args.no_align:
        graded = None
    else:
        arrays = align_to_video(arrays, N_FRAMES)
        graded = graded_mask()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("saved {} in {:.0f} s\n".format(os.path.relpath(args.out, ROOT), time.time() - start))

    profile = noise_profile(arrays, graded)
    print("=== NOISE PROFILE (aggregates only - safe to share) ===")
    print(profile["text"])


if __name__ == "__main__":
    main()
