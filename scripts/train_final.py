"""Step 6 (final): train the two-stage model on all 7 corrupted IDSSE matches.

    python scripts/train_final.py

Writes models/final_alive.txt and models/final_home.txt.
Deterministic: fixed corruption seeds, fixed LightGBM seed, deterministic mode.
"""

import glob
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.corrupt import corrupt  # noqa: E402
from src.features import build_features  # noqa: E402
from src import model as M  # noqa: E402

SEEDS = (11, 12)
SUBSAMPLE = 4
MODEL_PREFIX = os.path.join(ROOT, "models", "final")


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "idsse", "*.npz")))
    if len(files) != 7:
        sys.exit("expected 7 IDSSE matches in data/idsse/, found {}. "
                 "Run scripts/download_idsse.py first.".format(len(files)))
    Xs, ya, yh = [], [], []
    start = time.time()
    for path in files:
        match = os.path.basename(path)[:-4]
        arrays = dict(np.load(path))
        a, h = M.make_labels(arrays)
        for i, seed in enumerate(SEEDS):
            c = corrupt(arrays, seed=zlib.crc32("{}:{}".format(match, seed).encode()))
            X, names = build_features(c)
            sl = slice(i, None, SUBSAMPLE)
            Xs.append(X[sl])
            ya.append(a[sl])
            yh.append(h[sl])
            del X, c
        print("  features {} done ({:.0f} s)".format(match, time.time() - start), flush=True)
    X = np.concatenate(Xs)
    print("training on {} frames x {} features ...".format(*X.shape), flush=True)
    models = M.train(X, np.concatenate(ya), np.concatenate(yh))
    os.makedirs(os.path.dirname(MODEL_PREFIX), exist_ok=True)
    M.save(models, MODEL_PREFIX)
    with open(MODEL_PREFIX + "_features.txt", "w") as fh:
        fh.write("\n".join(names))
    print("saved models/final_*.txt in {:.0f} s".format(time.time() - start))


if __name__ == "__main__":
    main()
