"""Step 8: label every frame of the competition match.

    python scripts/predict.py --data-dir "C:\\path\\to\\dataset barcelona" --out outputs\\TEAM.csv

Needs data/atd/BAR-ATH_tactical.npz (scripts/convert_atd.py), models/final_*.txt
(scripts/train_final.py) and configs/decode.json.

Output: frame,label for all 199,319 video frames. Graded halves are labelled
H/A/D by the model + Viterbi (decoded separately per half); everything outside
the graded halves is X.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import build_features  # noqa: E402
from src import model as M  # noqa: E402
from src.decode import viterbi  # noqa: E402
from src.match import N_FRAMES, HALVES  # noqa: E402

ATD = os.path.join(ROOT, "data", "atd", "BAR-ATH_tactical.npz")


def home_team_name(meta_path):
    """Same rule kloppy uses: Score/@idLocalTeam if present, else the first Team."""
    root = ET.parse(meta_path).getroot()
    names = {t.get("id"): (t.findtext("Name") or "").strip() for t in root.iter("Team")}
    score = next(root.iter("Score"), None)
    if score is not None and score.get("idLocalTeam") in names:
        return names[score.get("idLocalTeam")]
    return next(iter(names.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"),
                        help="folder with tactical/ (only used to check team order)")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission.csv"))
    args = parser.parse_args()

    if args.data_dir:
        meta = os.path.join(args.data_dir, "tactical", "BAR-ATH_tactical_atd.xml")
        home = home_team_name(meta)
        print("home team in ATD: {}".format(home))
        if "barcelona" not in home.lower():
            sys.exit("home team is not Barcelona -- H/A would be swapped. Stop.")

    arrays = dict(np.load(ATD))
    if len(arrays["ball"]) != N_FRAMES:
        sys.exit("ATD arrays have {} frames, expected {}".format(len(arrays["ball"]), N_FRAMES))
    # period = graded half (0 outside), so period-relative features match training
    period = np.zeros(N_FRAMES, dtype=np.int8)
    for h, first, last in HALVES:
        period[first:last + 1] = h
    arrays["period"] = period

    print("building features ...", flush=True)
    X, names = build_features(arrays)
    with open(os.path.join(ROOT, "models", "final_features.txt")) as fh:
        trained = fh.read().split("\n")
    if names != trained:
        sys.exit("feature list differs from the trained model -- retrain.")
    models = M.load(os.path.join(ROOT, "models", "final"))
    proba = M.predict_proba(models, X)
    with open(os.path.join(ROOT, "configs", "decode.json")) as fh:
        params = json.load(fh)

    labels = np.full(N_FRAMES, "X", dtype="<U1")
    for h, first, last in HALVES:
        path = viterbi(proba[first:last + 1], **params)
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    graded = labels != "X"
    counts = {k: int(np.sum(labels[graded] == k)) for k in "HAD"}
    n_poss, durs = 0, []
    for h, first, last in HALVES:
        seg = labels[first:last + 1]
        brk = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        st = np.r_[0, brk]
        en = np.r_[brk, len(seg)]
        for s, e in zip(st, en):
            if seg[s] in "HA":
                n_poss += 1
                durs.append((e - s) / 25)
    print("wrote {}".format(args.out))
    print("graded frames: H {H}  A {A}  D {D}".format(**counts))
    print("possessions: {}  mean duration {:.1f} s  home share of alive {:.1f}%".format(
        n_poss, np.mean(durs), 100 * counts["H"] / max(counts["H"] + counts["A"], 1)))


if __name__ == "__main__":
    main()
