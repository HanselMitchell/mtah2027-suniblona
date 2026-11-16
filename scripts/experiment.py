"""Leave-one-match-out experiments on corrupted IDSSE, scored with the official
spellF1 / macroF1 definitions. Every design decision in this repository is taken
here -- never on the competition match or its public 5 minutes.

    python scripts/experiment.py --tag v2_lgb --feat 2 --model lgb
    python scripts/experiment.py --tag v1_lgb --feat 1 --model lgb
    python scripts/experiment.py --ensemble v2_lgb v2_xgb --tag ens   # average saved OOF probs
    python scripts/experiment.py --selftest                           # fast scorer == official

Per run it writes data/cache/oof_<tag>.npz (held-out probabilities) and appends a
line to outputs/experiments.jsonl with:
  argmax      spellF1/macroF1 of the per-frame argmax
  tuned       decode parameters tuned on all 7 held-out matches (optimistic)
  nested      decode parameters tuned on 6 matches, scored on the 7th (honest)
"""

import argparse
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
from src.decode import decode_periods  # noqa: E402
from src.scoring import fast_score, score, gt_labels  # noqa: E402
from src.tune import tune_decode, DECODE_INIT  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")
TRAIN_SEEDS = (11, 12)
EVAL_SEED = 99
SUBSAMPLE = 4


def load_idsse():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "idsse", "*.npz")))
    return {os.path.basename(f)[:-4]: dict(np.load(f)) for f in files}


def features(match, arrays, seed, version, subsample_offset=None, cver=2):
    """Cached features; training seeds are stored subsampled to save memory."""
    sub = "" if subsample_offset is None else "_sub%d" % subsample_offset
    path = os.path.join(CACHE, "f{}c{}_{}_s{}{}.npz".format(version, cver, match, seed, sub))
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], list(d["names"])
    c = corrupt(arrays, seed=zlib.crc32("{}:{}".format(match, seed).encode()), version=cver)
    X, names = build_features(c, version=version)
    if subsample_offset is not None:
        X = X[subsample_offset::SUBSAMPLE]
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, X=X, names=np.array(names))
    return X, names


def run_loo(data, version, model, params, cver=2, eval_cver=None):
    matches = sorted(data)
    tr = {}
    for m in matches:
        a, h = M.make_labels(data[m])
        for i, s in enumerate(TRAIN_SEEDS):
            cv = cver[i] if isinstance(cver, (list, tuple)) else cver
            X, names = features(m, data[m], s, version, subsample_offset=i, cver=cv)
            tr[(m, s)] = (X, a[i::SUBSAMPLE], h[i::SUBSAMPLE])
    probs = {}
    for held in matches:
        t0 = time.time()
        Xs = [tr[(m, s)][0] for m in matches if m != held for s in TRAIN_SEEDS]
        ya = [tr[(m, s)][1] for m in matches if m != held for s in TRAIN_SEEDS]
        yh = [tr[(m, s)][2] for m in matches if m != held for s in TRAIN_SEEDS]
        models = M.train(np.concatenate(Xs), np.concatenate(ya), np.concatenate(yh),
                         backend=model, params=params)
        Xe, _ = features(held, data[held], EVAL_SEED, version,
                         cver=eval_cver or (cver[-1] if isinstance(cver, (list, tuple)) else cver))
        probs[held] = M.predict_proba(models, Xe)
        print("  fold {} {:.0f} s".format(held, time.time() - t0), flush=True)
    return probs, names


def summarize(data, probs, tag, extra=None):
    matches = sorted(data)
    gts = {m: gt_labels(data[m]) for m in matches}
    per = {m: data[m]["period"] for m in matches}

    arg = [fast_score(gts[m], np.argmax(probs[m], 1).astype(np.int8), per[m]) for m in matches]
    res = {"tag": tag,
           "argmax": {k: float(np.mean([r[k] for r in arg])) for k in ("spellF1", "macroF1")}}

    best, _ = tune_decode(probs, gts, per, matches, DECODE_INIT)
    rows = [fast_score(gts[m], decode_periods(probs[m], per[m], best), per[m]) for m in matches]
    res["tuned"] = {"params": best,
                    "spellF1": float(np.mean([r["spellF1"] for r in rows])),
                    "macroF1": float(np.mean([r["macroF1"] for r in rows])),
                    "per_match": [round(r["spellF1"], 4) for r in rows]}

    nested = []
    for held in matches:
        others = [m for m in matches if m != held]
        p, _ = tune_decode(probs, gts, per, others, best)
        nested.append(fast_score(gts[held], decode_periods(probs[held], per[held], p), per[held]))
    res["nested"] = {"spellF1": float(np.mean([r["spellF1"] for r in nested])),
                     "macroF1": float(np.mean([r["macroF1"] for r in nested])),
                     "per_match": [round(r["spellF1"], 4) for r in nested]}
    if extra:
        res.update(extra)
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "experiments.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def selftest(data):
    from src.decode import DEFAULT
    m = sorted(data)[0]
    gt = gt_labels(data[m])
    rng = np.random.default_rng(0)
    proba = np.clip(np.eye(3)[gt] * 0.6 + rng.dirichlet([1, 1, 1], len(gt)) * 0.4, 0, 1)
    for params in (DEFAULT, dict(DEFAULT, c_ha=3, c_to_d=3, c_from_d=3)):
        pred = decode_periods(proba, data[m]["period"], params)
        a = score(gt, pred, data[m]["period"])
        b = fast_score(gt, pred, data[m]["period"])
        print("official spell {:.6f} macro {:.6f} | fast spell {:.6f} macro {:.6f}".format(
            a["spellF1"], a["macroF1"], b["spellF1"], b["macroF1"]))
        assert abs(a["spellF1"] - b["spellF1"]) < 1e-12 and abs(a["macroF1"] - b["macroF1"]) < 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--feat", type=int, default=2)
    ap.add_argument("--corrupt", default="3", help="one version, or one per training seed: 2,3")
    ap.add_argument("--eval-corrupt", type=int)
    ap.add_argument("--model", default="lgb", choices=("lgb", "xgb", "cat"))
    ap.add_argument("--params", default="{}", help="JSON overrides for the model")
    ap.add_argument("--ensemble", nargs="+")
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stack", help="tag of saved stage-1 OOF probabilities to stack on")
    args = ap.parse_args()

    data = load_idsse()
    if args.selftest:
        selftest(data)
        return
    if args.stack:
        from src import stack as S
        base = dict(np.load(os.path.join(CACHE, "oof_{}.npz".format(args.stack))))
        matches = sorted(data)
        F = {m: S.stack_features(base[m], data[m]["period"]) for m in matches}
        labels = {m: M.make_labels(data[m]) for m in matches}
        probs = {}
        for held in matches:
            others = [m for m in matches if m != held]
            sl = slice(0, None, 2)
            models = S.train([F[m][sl] for m in others], [labels[m][0][sl] for m in others],
                             [labels[m][1][sl] for m in others], threads=M.THREADS)
            probs[held] = S.predict(models, F[held])
            print("  stack fold {}".format(held), flush=True)
        np.savez(os.path.join(CACHE, "oof_{}.npz".format(args.tag)), **probs)
        summarize(data, probs, args.tag, {"stack_on": args.stack})
        return
    if args.ensemble:
        loaded = [dict(np.load(os.path.join(CACHE, "oof_{}.npz".format(t)))) for t in args.ensemble]
        w = args.weights or [1.0] * len(loaded)
        probs = {m: sum(wi * l[m] for wi, l in zip(w, loaded)) / sum(w) for m in loaded[0]}
        np.savez(os.path.join(CACHE, "oof_{}.npz".format(args.tag)), **probs)
        summarize(data, probs, args.tag, {"ensemble": args.ensemble, "weights": w})
        return
    params = json.loads(args.params)
    t0 = time.time()
    cver = [int(v) for v in args.corrupt.split(",")]
    cver = cver[0] if len(cver) == 1 else cver
    probs, names = run_loo(data, args.feat, args.model, params, cver=cver,
                           eval_cver=args.eval_corrupt)
    np.savez(os.path.join(CACHE, "oof_{}.npz".format(args.tag)), **probs)
    summarize(data, probs, args.tag, {"feat": args.feat, "corrupt": args.corrupt, "eval_corrupt": args.eval_corrupt, "model": args.model, "params": params,
                                      "n_features": len(names), "minutes": (time.time() - t0) / 60})


if __name__ == "__main__":
    main()
