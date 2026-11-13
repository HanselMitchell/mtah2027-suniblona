"""Step 7: leave-one-match-out validation on corrupted IDSSE, scored with the
competition's own spellF1. This -- not the public 5 minutes -- is where every
design decision is taken.

    python scripts/validate_idsse.py            # train 7 folds, save probabilities
    python scripts/validate_idsse.py --tune     # also grid-search the Viterbi costs

Features are cached in data/cache/ so re-runs are quick.
"""

import argparse
import glob
import itertools
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
from src.decode import decode_periods, DEFAULT  # noqa: E402
from src.scoring import score, gt_labels, baseline_nearest  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")
TRAIN_SEEDS = (11, 12)
EVAL_SEED = 99
SUBSAMPLE = 4


def features_cached(match, arrays, seed):
    path = os.path.join(CACHE, "{}_s{}.npz".format(match, seed))
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], list(d["names"]), d["ball_seen"]
    c = corrupt(arrays, seed=zlib.crc32("{}:{}".format(match, seed).encode()))
    X, names = build_features(c)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, X=X, names=np.array(names), ball_seen=np.isfinite(c["ball"][:, 0]))
    return X, names, np.isfinite(c["ball"][:, 0])


def corrupted(match, arrays, seed):
    return corrupt(arrays, seed=zlib.crc32("{}:{}".format(match, seed).encode()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--rounds", type=int, default=M.ROUNDS)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, "data", "idsse", "*.npz")))
    matches = [os.path.basename(f)[:-4] for f in files]
    data = {m: dict(np.load(f)) for m, f in zip(matches, files)}

    print("building features ...", flush=True)
    feats = {}
    for m in matches:
        for s in TRAIN_SEEDS + (EVAL_SEED,):
            feats[(m, s)] = features_cached(m, data[m], s)[0]
    names = features_cached(matches[0], data[matches[0]], EVAL_SEED)[1]

    probs = {}
    for held in matches:
        t0 = time.time()
        Xs, ya, yh = [], [], []
        for m in matches:
            if m == held:
                continue
            a, h = M.make_labels(data[m])
            for s in TRAIN_SEEDS:
                sl = slice(TRAIN_SEEDS.index(s), None, SUBSAMPLE)
                Xs.append(feats[(m, s)][sl])
                ya.append(a[sl])
                yh.append(h[sl])
        models = M.train(np.concatenate(Xs), np.concatenate(ya), np.concatenate(yh), args.rounds)
        probs[held] = M.predict_proba(models, feats[(held, EVAL_SEED)])
        print("  fold {} trained in {:.0f} s".format(held, time.time() - t0), flush=True)
    np.savez(os.path.join(CACHE, "loo_probs.npz"), **probs)

    def evaluate(params):
        rows = []
        for m in matches:
            gt = gt_labels(data[m])
            pred = decode_periods(probs[m], data[m]["period"], params)
            rows.append(score(gt, pred, data[m]["period"]))
        return (float(np.mean([r["spellF1"] for r in rows])),
                float(np.mean([r["macroF1"] for r in rows])), rows)

    print("\nper match (held out, corrupted)      spellF1  macroF1")
    base_rows, arg_rows = [], []
    for m in matches:
        gt = gt_labels(data[m])
        per = data[m]["period"]
        c = corrupted(m, data[m], EVAL_SEED)
        b = score(gt, baseline_nearest(c, 50), per)
        a = score(gt, np.argmax(probs[m], 1).astype(np.int8), per)
        base_rows.append(b)
        arg_rows.append(a)
        print("  {}  baseline_smooth50         {:.4f}   {:.4f}".format(m, b["spellF1"], b["macroF1"]))
        print("  {}  model argmax              {:.4f}   {:.4f}".format(m, a["spellF1"], a["macroF1"]))
    print("mean baseline_smooth50  spellF1 {:.4f}  macroF1 {:.4f}".format(
        np.mean([r["spellF1"] for r in base_rows]), np.mean([r["macroF1"] for r in base_rows])))
    print("mean model argmax       spellF1 {:.4f}  macroF1 {:.4f}".format(
        np.mean([r["spellF1"] for r in arg_rows]), np.mean([r["macroF1"] for r in arg_rows])))
    sp, mf, _ = evaluate(DEFAULT)
    print("mean model + viterbi    spellF1 {:.4f}  macroF1 {:.4f}  {}".format(sp, mf, DEFAULT))

    if args.tune:
        best = (sp, mf, DEFAULT)
        grid = [5, 10, 20, 40, 80]
        for c_ha, c_to_d, c_from_d in itertools.product(grid, grid, grid):
            params = {"c_ha": c_ha, "c_to_d": c_to_d, "c_from_d": c_from_d, "floor": 1e-4}
            s, m_, _ = evaluate(params)
            if s > best[0]:
                best = (s, m_, params)
                print("  better: spellF1 {:.4f}  macroF1 {:.4f}  {}".format(s, m_, params), flush=True)
        print("best Viterbi costs: {}".format(best[2]))
        with open(os.path.join(ROOT, "configs_decode.json"), "w") as fh:
            json.dump(best[2], fh, indent=2)


if __name__ == "__main__":
    main()
