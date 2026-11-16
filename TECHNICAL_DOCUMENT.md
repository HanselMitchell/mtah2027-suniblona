# Ball possession from automatic tracking — technical document

**More Than A Hack 2027 · Ball Possession & Spell Classification · team suniblona**

## 1. Approach in one paragraph

We label every tactical-video frame `H`/`A`/`D` from the provided automatic tracking (ATD) only.
A two-stage gradient-boosting classifier (alive vs dead, then home vs away) produces per-frame
probabilities from ~210 team-level spatio-temporal features. A semi-Markov Viterbi decoder with
switching costs **and minimum segment durations** turns those probabilities into possessions.
There are no labels for the competition match, so the model is trained on **seven public
Bundesliga matches (IDSSE)** whose clean tracking is deliberately **corrupted to look like the
ATD**. That corruption is calibrated only on *aggregate, label-free* statistics of the ATD. All
model and decoder choices were made by leave-one-match-out (LOMO) cross-validation on corrupted
IDSSE, scored with the official `possession_metrics.py`. We never used the public 5-minute ground
truth for training, tuning or model selection.

Why this design: possession is a property of intervals, not frames (spellF1 ranks the board).
A per-frame model is good at the evidence, and a structured decoder is good at enforcing that
possessions are contiguous and not absurdly short. Training on another league is the only
legitimate way to obtain labels. The main risk is therefore the **domain gap** between clean
IDSSE and the noisy ATD, and most of the work went into closing it (§3).

## 2. Assets used

| asset | used? | how |
|---|---|---|
| Tactical ATD (`_atd.txt`) | **yes** | only model input; parsed by our own reader (`src/atd_raw.py`) |
| Tactical ATD metadata (`_atd.xml`) | not available in our bundle | slot→team split inferred algorithmically (below) |
| Broadcast ATD metadata | yes, 1 field | `attack_direction_first_half` of the home team, as a sanity check of the inferred team split |
| Homography | no | the ATD is already on pitch coordinates; nothing to add for this task |
| Smart Tagging | no (tried to plan, dropped) | it cannot be reproduced on IDSSE, so its weight in the model could not be validated without labels |
| Videos (tactical / broadcast) | no | — |

**Slot→team split without the XML.** EPTS lists home slots first, so the split is a single
index *k*. We choose the *k* that minimises the number of frames in which either side would have
more than 11 tracked objects. This gives home = slots 0–23 and away = 24–45, and both
neighbouring splits are far worse. As a check, the home block must sit in the half it defends
during the first half, which it does. The two goalkeeper slots (always tracked, at opposite ends,
swapping at half time) confirm the split. No frame-specific constant chosen by a person is
involved. When `_atd.xml` is present, the reader uses it instead (same rule as the official
starter).

## 3. Training data and the corruption model

**Data.** IDSSE: Bassek et al. (2025), *Scientific Data* 12:195, CC-BY 4.0, 7 matches of the
German Bundesliga (2) and 2. Bundesliga (5) (DFL/Sportec Solutions). It contains positions at 25 Hz with the provider's
official **ball status** (alive/dead) and **ball possession** (team) per frame, produced by
Sportec Solutions' tracking and quality-control process (not by us). Labels: alive → owning team
(`H`/`A`), dead → `D`, which is the same mechanical derivation the competition ground truth uses.
Total ≈ 1.0 M frames, ≈ 3,060 possessions. No data about BAR-ATH was used.

**Corruption** (`src/corrupt.py`). It is applied to the clean IDSSE before feature extraction,
with random parameters drawn per match and seed. Every target number comes from aggregate,
label-free statistics of the ATD (`scripts/convert_atd.py` and `scripts/gap_context.py` print
them):

- *v1 (inherited):* ball missing during ~80–95% of stoppages (ATD: ball visible on 59% of graded
  frames, gaps median ~11 s), plus 5–13 s gaps during live play, short dropouts, slowly varying
  position error, rare jumps, and player tracks fragmented to ~10.2 of 11 per team.
- *v2 (new): ball smoothing.* The ATD ball trajectory is temporally smoothed by the provider. The
  3-frame velocity change has p90/p99 = 2.2/4.2 m/s in the ATD vs 6.3/17.9 in IDSSE. A Gaussian
  smoothing with σ ~ U(5, 7) frames reproduces the ATD percentiles while preserving ball speed.
  Without this, every touch, deflection or pass feature means something different in training and
  in test.
- *v3 (new): visibility artefacts.* An adversarial validation (a classifier separating IDSSE
  frames from ATD frames) pointed at two sensor artefacts. (a) The ATD re-acquires a lost ball
  only once it is struck: the median speed of the first visible step after a gap is 12 m/s, vs
  0.4 m/s in v2-corrupted IDSSE. We therefore extend gaps until the ball starts moving (median
  then ~10 m/s). (b) The DFL feed parks dead balls *outside* the lines (16% of visible frames after
  v2 corruption, vs 4% in the ATD). We hide the ball beyond the lines, which removes a "free" dead
  cue the ATD never provides.

Sanity check of realism: the official baseline (`baseline_smooth50`) scores spellF1 **0.283** on
v3-corrupted IDSSE. The organisers report **0.243** for it on the full BAR-ATH match. Clean IDSSE
would make every method look far better than it is.

## 4. Features (`src/features.py`, 211 per frame, all sign-free)

No player identities (the ATD has none) and no absolute x/y. Only quantities that are symmetric
under swapping attack direction.

- **Ball visibility / gaps:** seen flag, time since/until seen, gap length, visible share in
  ±5 s and ±20 s windows.
- **Ball kinematics:** speed (multi-lag), *velocity change and turn angle across ±3 frames*
  (kicks, deflections), speed gain, rolling max of speed and velocity change over 0.5/2/5 s past
  and future.
- **Ball ↔ players, per team:** 1st–3rd nearest distance, counts within 1/2/5/9.15 m, nearest-team
  indicator and distance difference/log-ratio, each with rolling means over 0.5–20 s past and
  future; rolling minimum of each team's nearest distance.
- **Touch / control events:** a *touch* is a velocity change with a player within 2.5 m; a
  *control* frame is a slow ball with a player within 1.5 m. Features: team of last/next
  touch/control, time since/until, whether the last two touches were by the same team, touch rate
  and team share over 2–20 s, and how often the nearest team flips (contested balls).
- **Pitch lines / restarts:** distance to touchline, goal line, nearest line, corner, centre;
  where the ball was when lost and when found again, ball beyond the lines; restart formations
  (players beyond the touch/goal line, near a corner flag, in the goal area, kick-off shape, a ring
  of players 2–9 m from the ball for free kicks).
- **Collective movement:** median/mean/p90 player speed, share of fast/still players, per-team
  speed and acceleration, 0.5–20 s windows past and future (stoppages start with a collective
  slowdown).
- **Team shape:** spreads, centroid distance, ball position relative to each block, and the
  ball's advance towards the home team's attacking end (sign-free).

## 5. Model and decoder

- **Two-stage LightGBM** (`src/model.py`): `p(alive)` and `p(home | alive)`, combined into
  `P(H, A, D)`. 400 rounds, 63 leaves, learning rate 0.05, deterministic mode, fixed seed and
  thread count. Training data: every 4th frame of the 7 matches × 2 corruption seeds.
- **Ensemble of two corruption regimes:** one model trained on v2-corrupted and one on
  v3-corrupted IDSSE, probabilities averaged. This hedges against our uncertainty about how
  exactly the ATD loses the ball, and it gives smoother, better calibrated probabilities (argmax
  spellF1 0.36 → 0.39).
- **Semi-Markov Viterbi** (`src/decode.py`, numba): per half, maximises the sum of
  log-probabilities minus a cost per change of state (H↔A, →D, D→H/A), subject to a **minimum
  duration** per segment. The recursion is written over segment entries, so it stays O(3T) for
  any minimum length. Chosen parameters (LOMO): `c_ha=2`, `c_to_d=32`, `c_from_d=60`,
  `min_poss=30` frames (1.2 s), `min_dead=1`, probability floor 1e-4. A cheap H↔A switch combined
  with a 1.2 s minimum duration keeps genuine quick turnovers while removing the sub-second
  flicker that costs two possessions each under spellF1. Expensive transitions to and from D
  stop brief ball-tracking dropouts from splitting possessions.

**Possession rules implemented** (inherited from the provider definition the ground truth uses):
a possession is a maximal run of one team while the ball is alive; loose balls and challenges
belong to the team in possession until the other team gains control. The model learns this from
the provider's labels, and the minimum-duration constraint prevents momentary contacts from
becoming possessions. Restarts start a new possession after a `D` segment; dead-ball ownership is
never predicted.

## 6. Validation experiments (LOMO on corrupted IDSSE, official metric code)

"nested" = decoder parameters tuned on 6 matches and scored on the 7th, which is the honest
number. "test corr." is the corruption applied to the held-out match. Only rows with the same
test corruption are directly comparable, and v3 is the most ATD-like.

| # | features | train corruption | test corr. | model | decoder | spellF1 (nested) | macroF1 |
|---|---|---|---|---|---|---|---|
| 0 | v1 | v1 | v1 | LGBM | original Viterbi | 0.607 | 0.920 |
| 1 | v1 | v1 | v1 | LGBM | min-duration Viterbi | 0.613 | 0.920 |
| 2a | v1 | v1 | **v3** | LGBM | original Viterbi | 0.553 | 0.901 |
| 2b | v1 | v1 | **v3** | LGBM | min-duration | 0.562 | 0.901 |
| 3 | v1 | v2 | v2 | LGBM | min-duration | 0.568 | 0.908 |
| 4 | **v2** | v2 | v2 | LGBM | min-duration | 0.594 | 0.918 |
| 5 | v2 | v2 | v2 | LGBM + temporal stacking | min-duration | 0.601 | 0.916 |
| 6 | v2 | v3 | v3 | LGBM | min-duration | 0.584 | 0.916 |
| 7 | v2 | v2 | v3 | LGBM | min-duration | 0.595 | 0.916 |
| 8 | v2 | v2+v3 mixed seeds | v3 | LGBM | min-duration | 0.586 | 0.915 |
| **9** | **v2** | **v2 ‖ v3 ensemble** | v3 | **2× LGBM** | **min-duration** | **0.596** | **0.917** |
| 10 | v2 | v3 | v3 | XGBoost | min-duration | 0.587 | 0.915 |
| 11 | v2 | ensemble #9 + XGB | v3 | 3 models | min-duration | 0.596 | 0.917 |
| 12 | v2 | ensemble #9 + stacking | v3 | — | min-duration | 0.598 | 0.914 |
| ref | — | — | v3 | official baseline_smooth50 | — | 0.283 | 0.725 |

Reading the table:

- **Before vs after, on realistic test data:** the previous pipeline (row 2a) scores 0.553 /
  0.901 and the final system (row 9) scores 0.596 / 0.917, i.e. **+0.043 spellF1, +0.016
  macroF1**.
- **Corrupting the ball realistically matters most.** The same v1 model falls from 0.607 (v1 test)
  to 0.553 when tested on realistic v3 data. Earlier validation numbers were therefore
  optimistic, which matches the gap the previous version showed on the competition match.
- **New features help under realistic noise:** v1 → v2 features gives +0.026 spellF1 (rows 3→4),
  better on 7 of 7 matches.
- **Decoder:** the minimum-duration constraint and re-tuned costs add ~+0.01 over the original
  Viterbi.
- Ensembles, XGBoost and temporal stacking land within ±0.005 of row 9, below the ~0.01
  run-to-run noise of this protocol. We kept row 9, the simplest configuration at the top. The
  extra models did not justify their cost, and stacking is mildly leaky under LOMO.

**Pipeline correctness check** (not a selection criterion): on the public 5-minute window the
final file scores spellF1 0.537 / macroF1 0.680 with 18 possessions (ground truth: 17). The
previous version scored ~0.40 / ~0.60 there. The organisers state that differences of up to ~0.04
(full match) or ~0.06 per possession (public window) are noise, so we read this only as
"frames aligned, labels meaning what we think". On the full match the model predicts 386
possessions (mean 10.0 s) with a home share of alive time of 61%.

## 7. Limitations and what we would do next

- **Domain gap is not closed.** The ATD and corrupted IDSSE remain perfectly separable by an
  adversarial classifier. Part of that is legitimate (Barcelona's high line and possession style
  vs German teams), but part is unmodelled sensor behaviour: identity swaps, referee or
  ghost tracks, and how gaps start.
- **Long ball gaps during live play** (~10% of missing ATD frames) are the main failure mode. The
  model falls back on player movement, and possession changes inside such gaps are usually
  missed or placed late.
- **Restarts:** the exact restart frame is often detected late, because the ATD finds the ball only
  after it is struck, and a whole short possession after a restart can be lost.
- With more time: reconstruct the ball inside gaps from the homography and video (a ball
  detector), learn the corruption adversarially (domain-adversarial training or importance
  weighting), use Smart Tagging set pieces and team phases as a validated prior (it would need a
  proxy on IDSSE), and add a sequence model (BiGRU/TCN) in place of stacking.

## 8. Reproduction

```bash
pip install -r requirements.txt                     # Python 3.12
set MTAH_DATA_DIR=<folder with tactical/ and broadcast/>   # PowerShell: $env:MTAH_DATA_DIR="..."
python scripts/download_idsse.py                    # 7 IDSSE matches (figshare) -> data/idsse/*.npz
python scripts/convert_atd.py                       # ATD -> data/atd/BAR-ATH_tactical.npz
python scripts/train_final.py                       # configs/model.json -> models/
python scripts/predict.py                           # configs/decode.json -> submission/suniblona.csv
# optional: the experiments of section 6
python scripts/experiment.py --selftest
python scripts/experiment.py --tag v2c3_lgb --feat 2 --corrupt 3
python scripts/experiment.py --tag v2c2_on_c3 --feat 2 --corrupt 2 --eval-corrupt 3
python scripts/experiment.py --ensemble v2c2_on_c3 v2c3_lgb --tag ens_c2c3
```

Everything is seeded and LightGBM runs in deterministic mode with a fixed thread count
(`MTAH_THREADS`, default 8). Retraining from scratch and predicting reproduced the submitted file
byte for byte (identical MD5). Runtime on a 16-thread laptop CPU: ~15 min for IDSSE conversion,
~6 min training, ~1 min prediction. No data from the match is stored in the repository
(`.gitignore` excludes videos, tracking, homography, tagging, all CSV/XML/TXT data and derived
arrays).
