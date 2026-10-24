"""Tiny text helpers."""


def slug(s):
    return "-".join(w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split())


def dedent_lines(text):
    lines = text.splitlines()
    pad = min((len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()), default=0)
    return "\n".join(ln[pad:] for ln in lines)
