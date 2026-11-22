import math


def percentile(values: list[float], p: float) -> float:
    if not values or not 0 <= p <= 100:
        raise ValueError(
            f"Percentile inputs are invalid. Sequence Length: {len(values)}, Percentile: {p}."
        )

    sorted_values = sorted(values)
    position = p / 100 * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)

    if lo == hi:
        return sorted_values[lo]

    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)
