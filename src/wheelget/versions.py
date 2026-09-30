from __future__ import annotations

import re

_PRERELEASE_RE = re.compile(r"(?:a|b|rc|dev)\d*$", re.IGNORECASE)


def is_prerelease(value: str) -> bool:
    base = value.split("+", 1)[0]
    return bool(_PRERELEASE_RE.search(base))


def version_key(value: str) -> tuple[tuple[int, int, str], ...]:
    key = []
    for part in re.split(r"[.\-+]", value):
        match = re.match(r"(\d+)(.*)", part)
        if match:
            key.append((0, int(match.group(1)), match.group(2)))
        else:
            key.append((1, 0, part))
    return tuple(key)
