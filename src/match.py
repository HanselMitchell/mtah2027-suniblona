"""Constants of the competition match, all taken from the public repository
(metrica-sports/more-than-a-hack-2027: DATA.md and games.py). None is a label."""

import numpy as np

N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)  # metres, games.py
# graded video-frame blocks, inclusive (DATA.md "Which frames are graded")
HALVES = ((1, 11185, 90651), (2, 110808, 190197))


def graded_mask(n=N_FRAMES):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in HALVES:
        mask[first:last + 1] = True
    return mask


def half_of_frame(n=N_FRAMES):
    """Array of 0 (not graded), 1 or 2."""
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in HALVES:
        half[first:last + 1] = h
    return half
