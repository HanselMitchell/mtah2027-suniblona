"""Step 6 (final): train the chosen model(s) on all 7 corrupted IDSSE matches.

    python scripts/train_final.py

Reads configs/model.json:
    {"feat": 2,
     "members": [{"name": "lgb_c2", "backend": "lgb", "corrupt": 2, "params": {}, "weight": 1.0},
                 ...]}
Each member is trained on IDSSE corrupted with its own corruption version.

Writes models/<name>_alive.*, models/<name>_home.* and models/features.txt.
Deterministic: fixed corruption seeds, fixed library seeds, fixed thread count.
"""

import glob
import json
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
MODELS = os.path.join(ROOT, "models")


def build_training_set(files, feat, cver):
    Xs, ya, yh = [], [], []
    start = time.time()
    for path in files:
        match = os.path.basename(path)[:-4]
        arrays = dict(np.load(path))
        a, h = M.make_labels(arrays)
        for i, seed in enumerate(SEEDS):
            c = corrupt(arrays, seed=zlib.crc32("{}:{}".format(match, seed).encode()), version=cver)
            X, names = build_features(c, version=feat)
            sl = slice(i, None, SUBSAMPLE)
            Xs.append(X[sl])
            ya.append(a[sl])
            yh.append(h[sl])
            del X, c
        print("  corruption v{}: features {} done ({:.0f} s)".format(cver, match, time.time() - start),
              flush=True)
    return np.concatenate(Xs), np.concatenate(ya), np.concatenate(yh), names


def main():
    with open(os.path.join(ROOT, "configs", "model.json")) as fh:
        cfg = json.load(fh)
    files = sorted(glob.glob(os.path.join(ROOT, "data", "idsse", "*.npz")))
    if len(files) != 7:
        sys.exit("expected 7 IDSSE matches in data/idsse/, found {}. "
                 "Run scripts/download_idsse.py first.".format(len(files)))
    os.makedirs(MODELS, exist_ok=True)
    start = time.time()
    for cver in sorted({m.get("corrupt", cfg.get("corrupt", 2)) for m in cfg["members"]}):
        X, ya, yh, names = build_training_set(files, cfg["feat"], cver)
        for mem in cfg["members"]:
            if mem.get("corrupt", cfg.get("corrupt", 2)) != cver:
                continue
            print("training {} on {} frames x {} features ...".format(mem["name"], *X.shape),
                  flush=True)
            models = M.train(X, ya, yh, backend=mem["backend"], params=mem.get("params", {}))
            M.save(models, os.path.join(MODELS, mem["name"]), backend=mem["backend"])
        del X
    with open(os.path.join(MODELS, "features.txt"), "w") as fh:
        fh.write("\n".join(names))
    print("saved models/ in {:.0f} s".format(time.time() - start))


if __name__ == "__main__":
    main()
