"""Helpers for working with raw SoccerNet video file metadata."""
from __future__ import annotations

import re

HALF_TOKENS = {
    1: {
        "1",
        "01",
        "1st",
        "1sthalf",
        "first",
        "firsthalf",
        "half1",
        "1half",
    },
    2: {
        "2",
        "02",
        "2nd",
        "2ndhalf",
        "second",
        "secondhalf",
        "half2",
        "2half",
    },
}


def infer_half_from_stem(stem: str) -> int:
    """Infer match half (1 or 2) from a SoccerNet video filename stem."""
    if not stem:
        return 1

    lower = stem.lower()
    tokens = [token for token in re.split(r"[^0-9a-z]+", lower) if token]

    for token in reversed(tokens):
        if token in HALF_TOKENS[2]:
            return 2
        if token in HALF_TOKENS[1]:
            return 1

    if re.search(r"(?:^|[^0-9])2(?:[^0-9]|$)", lower):
        return 2

    return 1
