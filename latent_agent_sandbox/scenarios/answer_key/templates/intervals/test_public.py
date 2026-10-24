from pkg.intervals import merge_intervals


def test_overlap():
    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]


def test_touch_flag():
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 4], [4, 5]]
    assert merge_intervals([[1, 4], [4, 5]], touch=True) == [[1, 5]]


def test_empty():
    assert merge_intervals([]) == []
