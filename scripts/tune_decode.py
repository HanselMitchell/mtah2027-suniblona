"""Choose the Viterbi transition costs on the leave-one-match-out IDSSE
probabilities saved by validate_idsse.py. Scored with the official spellF1.
Never touches the competition match.

    python scripts/tune_decode.py
"""

import glob
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.decode import decode_periods  # noqa: E402
from src.scoring import score, gt_labels  # noqa: E402


def main():
    probs = dict(np.load(os.path.join(ROOT, "data", "cache", "loo_probs.npz")))
    data = {os.path.basename(f)[:-4]: dict(np.load(f))
            for f in sorted(glob.glob(os.path.join(ROOT, "data", "idsse", "*.npz")))}
    gts = {m: gt_labels(d) for m, d in data.items()}

    def evaluate(params):
        rows = [score(gts[m], decode_periods(probs[m], data[m]["period"], params),
                      data[m]["period"]) for m in data]
        return (float(np.mean([r["spellF1"] for r in rows])),
                float(np.mean([r["macroF1"] for r in rows])),
                [round(r["spellF1"], 4) for r in rows])

    results = []

    def run(grid_ha, grid_to, grid_from, label):
        for c_ha, c_to, c_from in itertools.product(grid_ha, grid_to, grid_from):
            params = {"c_ha": float(c_ha), "c_to_d": float(c_to), "c_from_d": float(c_from),
                      "floor": 1e-4}
            s, m, per = evaluate(params)
            results.append((s, m, params, per))
        results.sort(key=lambda r: -r[0])
        print("{}: best spellF1 {:.4f} macroF1 {:.4f} {}".format(
            label, results[0][0], results[0][1], results[0][2]), flush=True)

    run([10, 25, 50, 100], [10, 25, 50, 100], [10, 25, 50, 100], "coarse")
    b = results[0][2]
    around = lambda v: sorted({max(2.0, v * f) for f in (0.6, 0.8, 1.0, 1.25, 1.6)})
    run(around(b["c_ha"]), around(b["c_to_d"]), around(b["c_from_d"]), "fine")

    print("\ntop 5:")
    for s, m, p, per in results[:5]:
        print("  spellF1 {:.4f} macroF1 {:.4f} {}  per match {}".format(s, m, p, per))
    with open(os.path.join(ROOT, "configs", "decode.json"), "w") as fh:
        json.dump(results[0][2], fh, indent=2)
    print("saved configs/decode.json")


if __name__ == "__main__":
    main()
