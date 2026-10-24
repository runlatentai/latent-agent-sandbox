# Task: merge_intervals

Implement `merge_intervals(intervals, touch=False)` in `pkg/intervals.py`.

* `intervals` is a list of `[start, end]` integer pairs with `start <= end`, in any order.
* Return a new list of merged `[start, end]` pairs sorted by `start`.
* Two intervals overlap when one's `start` is strictly less than the other's `end` and they share at least one
  point; overlapping intervals are merged into one. Intervals that only touch (`a.end == b.start`) are merged
  only when `touch` is `True`.
* An interval of one point (`start == end`) is kept as is unless it lies inside another interval or, with
  `touch=True`, on another interval's boundary.
* The input must not be modified. An empty input returns `[]`.
* Results are lists of lists of ints, never tuples.
