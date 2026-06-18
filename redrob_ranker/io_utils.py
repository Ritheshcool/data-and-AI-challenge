"""Streaming I/O and small parsing helpers for candidate records."""
import datetime
import gzip
import json
import io


def open_maybe_gzip(path):
    """Open .jsonl or .jsonl.gz transparently as a text stream."""
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path):
    """Yield candidate dicts one at a time (constant memory)."""
    with open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_date(s):
    """Parse an ISO date string (YYYY-MM-DD...) -> datetime.date, or None."""
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def months_between(start, end):
    """Whole months from start to end (both datetime.date). Negative if end<start."""
    if start is None or end is None:
        return None
    return (end.year - start.year) * 12 + (end.month - start.month)


def safe_get(d, *keys, default=None):
    """Nested getter: safe_get(c, 'profile', 'years_of_experience')."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur
