"""Fast reader for the Metrica FIFA/EPTS raw tracking text, without kloppy.

Line format (see the DataFormatSpecification in any *_atd.xml):

    <frame_id>:<x>,<y>,<speed>;<x>,<y>,<speed>;...:<ball x>,<ball y>,<ball z>

Coordinates are fractions of the pitch. frame_id counts from 1 and equals
video_frame + 1 (official starter/load_atd.py).

Team of each track slot
-----------------------
The metadata XML lists, for every slot, the team it belongs to. When the XML is
present we read it (same rule as the official starter: the first <Team> is home).
When it is missing, we infer the split *algorithmically* from the tracking itself:
EPTS lists home slots first, so the split is a single index k, chosen as the k
that minimises frames where a side would have more than 11 tracked objects
(infer_teams). As a sanity check the home block must, over the first half, sit
on the side the home team defends, read from the metadata of the other recording
of the same match (broadcast XML, `attack_direction_first_half`). No label and no
hand-picked constant tied to frames of the game is involved.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np


def parse_raw(path, n_frames=None):
    """Return frame_id (T,), players (T, S, 2) fractions, ball (T, 2) fractions."""
    frame_ids, players, balls = [], [], []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fid, pl, bl = line.split(":")
            frame_ids.append(int(fid))
            players.append(pl.replace(";", ","))
            balls.append(bl)
    n = len(frame_ids)
    p = np.array(",".join(players).split(","), dtype=np.float32)
    s = p.size // (3 * n)
    p = p.reshape(n, s, 3)[:, :, :2]
    b = np.array(",".join(balls).split(","), dtype=np.float32).reshape(n, 3)[:, :2]
    return np.array(frame_ids, dtype=np.int64), p, b


def teams_from_xml(meta_path):
    """0/1 per slot from the metadata XML (first team = home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if pl.get("teamId") == home else 1 for pl in root.iter("Player")],
                    dtype=np.int8)


def home_attacks_left_to_right(meta_path):
    root = ET.parse(meta_path).getroot()
    team = next(root.iter("Team"))
    for par in team.iter("ProviderParameter"):
        if par.findtext("Name") == "attack_direction_first_half":
            return (par.findtext("Value") or "").strip() == "left_to_right"
    return True


def infer_teams(players, graded, max_per_team=11):
    """Home/away of each slot without the metadata XML.

    EPTS lists the home team's slots first, then the away team's, so the split is one
    integer k. We take the k that minimises how often either side would have more than
    `max_per_team` tracked objects in a frame (a team cannot field 12). Returns 0/1 per slot.
    """
    seen = np.isfinite(players[graded, :, 0]).astype(np.int16)
    best = None
    for k in range(1, players.shape[1]):
        a = seen[:, :k].sum(1)
        b = seen[:, k:].sum(1)
        c = int(np.maximum(a - max_per_team, 0).sum() + np.maximum(b - max_per_team, 0).sum())
        if best is None or c < best[0]:
            best = (c, k)
    teams = (np.arange(players.shape[1]) >= best[1]).astype(np.int8)
    return teams, best[1]


def check_orientation(players, teams, first_half, home_l2r):
    """True if, in the first half, the home slots sit on the side home defends (sanity check)."""
    with np.errstate(invalid="ignore"):
        x = players[first_half, :, 0]
        hx = np.nanmean(x[:, teams == 0])
        ax = np.nanmean(x[:, teams == 1])
    return (hx < ax) if home_l2r else (hx > ax)
