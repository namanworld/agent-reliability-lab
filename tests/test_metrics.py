import pytest

from arl.metrics import percentile

LATENCIES = [120, 80, 300, 150, 100]  # deliberately unsorted


def test_median_of_odd_count():
    assert percentile(LATENCIES, 50) == 120


def test_median_of_even_count_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


def test_interpolates_between_ranks():
    # sorted: [80, 100, 120, 150, 300]; position = 0.95 * 4 = 3.8
    assert percentile(LATENCIES, 95) == pytest.approx(270)


def test_p0_and_p100_are_min_and_max():
    assert percentile(LATENCIES, 0) == 80
    assert percentile(LATENCIES, 100) == 300


def test_single_value():
    assert percentile([42.0], 99) == 42.0


def test_empty_input_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


@pytest.mark.parametrize("p", [-1, 100.5])
def test_p_out_of_range_raises(p):
    with pytest.raises(ValueError):
        percentile(LATENCIES, p)


def test_does_not_mutate_input():
    data = [3, 1, 2]
    percentile(data, 50)
    assert data == [3, 1, 2]