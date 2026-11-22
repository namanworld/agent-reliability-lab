import math
from collections.abc import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    if len(values) == 0:
        raise ValueError("percentile() needs at least one value")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be between 0 and 100, got {p}")

    sorted_values = sorted(values)
    position = p / 100 * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)

    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)
