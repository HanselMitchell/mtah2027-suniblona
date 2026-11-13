"""Convert a kloppy TrackingDataset into compact numpy arrays.

The same structure is used for every source (IDSSE training matches and the
competition ATD), so the feature code downstream never needs to know where the
data came from:

    frame_id   (T,)        int
    period     (T,)        int8
    ball       (T, 2)      float32, metres, NaN where the ball is not tracked
    home       (T, P, 2)   float32, metres, NaN padded
    away       (T, P, 2)   float32, metres, NaN padded
    alive      (T,)        int8   1 alive, 0 dead, -1 unknown
    owner      (T,)        int8   0 home, 1 away, -1 none/unknown
"""

import numpy as np

from kloppy.domain import BallState, Ground


def _team_player_ids(dataset):
    home_ids, away_ids = [], []
    for team in dataset.metadata.teams:
        ids = [p.player_id for p in team.players]
        if team.ground == Ground.HOME:
            home_ids = ids
        else:
            away_ids = ids
    return home_ids, away_ids


def dataset_to_arrays(dataset, pitch=None):
    """Return a dict of arrays (see module docstring) from a kloppy dataset."""
    dims = dataset.metadata.pitch_dimensions
    length = pitch[0] if pitch else float(dims.pitch_length)
    width = pitch[1] if pitch else float(dims.pitch_width)

    home_ids, away_ids = _team_player_ids(dataset)
    home_team = [t for t in dataset.metadata.teams if t.ground == Ground.HOME][0]

    n = len(dataset.frames)
    n_home, n_away = max(len(home_ids), 1), max(len(away_ids), 1)
    home_index = {pid: i for i, pid in enumerate(home_ids)}
    away_index = {pid: i for i, pid in enumerate(away_ids)}

    frame_id = np.zeros(n, dtype=np.int64)
    period = np.zeros(n, dtype=np.int8)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    home = np.full((n, n_home, 2), np.nan, dtype=np.float32)
    away = np.full((n, n_away, 2), np.nan, dtype=np.float32)
    alive = np.full(n, -1, dtype=np.int8)
    owner = np.full(n, -1, dtype=np.int8)

    for t, frame in enumerate(dataset.frames):
        frame_id[t] = frame.frame_id
        period[t] = frame.period.id
        if frame.ball_coordinates is not None and frame.ball_coordinates.x is not None:
            ball[t, 0] = frame.ball_coordinates.x * length
            ball[t, 1] = frame.ball_coordinates.y * width
        if frame.ball_state == BallState.ALIVE:
            alive[t] = 1
        elif frame.ball_state == BallState.DEAD:
            alive[t] = 0
        if frame.ball_owning_team is not None:
            owner[t] = 0 if frame.ball_owning_team.team_id == home_team.team_id else 1
        for player, pdata in frame.players_data.items():
            if pdata.coordinates is None or pdata.coordinates.x is None:
                continue
            pid = player.player_id
            if pid in home_index:
                home[t, home_index[pid]] = (pdata.coordinates.x * length,
                                            pdata.coordinates.y * width)
            elif pid in away_index:
                away[t, away_index[pid]] = (pdata.coordinates.x * length,
                                            pdata.coordinates.y * width)

    return {
        "frame_id": frame_id, "period": period, "ball": ball,
        "home": home, "away": away, "alive": alive, "owner": owner,
        "pitch": np.array([length, width], dtype=np.float32),
    }


def summarize(arrays, name=""):
    """Human-readable summary: what a person checks before trusting a file."""
    n = len(arrays["frame_id"])
    alive = arrays["alive"]
    owner = arrays["owner"]
    ball_seen = np.isfinite(arrays["ball"][:, 0])
    players_seen = (np.isfinite(arrays["home"][:, :, 0]).sum(1)
                    + np.isfinite(arrays["away"][:, :, 0]).sum(1))
    lines = [
        "{}: {} frames, periods {}".format(name, n, sorted(set(arrays["period"].tolist()))),
        "  ball tracked      {:.1f}% of frames".format(100 * ball_seen.mean()),
        "  players per frame {:.1f} (mean)".format(players_seen.mean()),
        "  alive / dead / ?  {:.1f}% / {:.1f}% / {:.1f}%".format(
            100 * (alive == 1).mean(), 100 * (alive == 0).mean(), 100 * (alive == -1).mean()),
    ]
    live = alive == 1
    if live.any():
        lines.append("  home share alive  {:.1f}%   owner unknown when alive {:.1f}%".format(
            100 * (owner[live] == 0).mean() / max((owner[live] >= 0).mean(), 1e-9),
            100 * (owner[live] < 0).mean()))
        # possessions = maximal runs of one owner while alive, within a period
        label = np.where(live, owner, 9)
        changes = (label[1:] != label[:-1]) | (arrays["period"][1:] != arrays["period"][:-1])
        starts = np.concatenate(([0], np.flatnonzero(changes) + 1))
        n_poss = int(np.sum(np.isin(label[starts], (0, 1))))
        lines.append("  possessions       {}".format(n_poss))
    return "\n".join(lines)
