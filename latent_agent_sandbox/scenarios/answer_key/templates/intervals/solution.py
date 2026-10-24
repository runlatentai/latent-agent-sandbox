"""Interval merging. See SPEC.md."""


def merge_intervals(intervals, touch=False):
    items = sorted([list(p) for p in intervals], key=lambda p: (p[0], p[1]))
    out = []
    for s, e in items:
        if out and (s < out[-1][1] or (touch and s == out[-1][1]) or (s == e and s < out[-1][1])):
            out[-1][1] = max(out[-1][1], e)
        elif out and s == out[-1][1] and s == e and touch:
            pass
        else:
            out.append([s, e])
    return out
