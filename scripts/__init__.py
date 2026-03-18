"""Shared utilities for FlowSim CLI scripts."""


def parse_sweep_point(s: str) -> tuple[int, int, int]:
    """Parse a ``BS:INPUT_LEN:CTX`` string into an int 3-tuple.

    Raises :class:`ValueError` on bad input.
    """
    parts = s.strip().split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Bad sweep point {s!r}: expected BS:INPUT_LEN:CTX "
            f"(e.g. 1:2048:0)"
        )
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(
            f"Bad sweep point {s!r}: all three values must be integers"
        )


def load_sweep_file(path: str) -> list[tuple[int, int, int]]:
    """Read sweep points from a file (one ``BS:INPUT_LEN:CTX`` per line).

    Blank lines and ``#`` comments are skipped.
    Raises :class:`ValueError` on bad entries.
    """
    points: list[tuple[int, int, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            points.append(parse_sweep_point(line))
    return points
