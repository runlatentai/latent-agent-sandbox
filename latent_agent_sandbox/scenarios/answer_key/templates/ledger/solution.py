"""Shared-expense ledger. See SPEC.md."""

import re

_AMT = re.compile(r"^\d+(\.\d{1,2})?$")
_NAME = re.compile(r"^[a-z]+$")


def _cents(tok):
    if not _AMT.match(tok):
        raise ValueError(tok)
    whole, _, frac = tok.partition(".")
    return int(whole) * 100 + int((frac + "00")[:2])


def balances(lines):
    out = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 4 and parts[1] == "pays" and _NAME.match(parts[0]) and _NAME.match(parts[2]):
            amt = _cents(parts[3])
            out[parts[0]] = out.get(parts[0], 0) - amt
            out[parts[2]] = out.get(parts[2], 0) + amt
        elif len(parts) == 5 and parts[1] == "splits" and parts[3] == "among" and _NAME.match(parts[0]):
            names = parts[4].split(",")
            if not names or not all(_NAME.match(n) for n in names):
                raise ValueError(raw)
            amt = _cents(parts[2])
            out[parts[0]] = out.get(parts[0], 0) - amt
            share, rem = divmod(amt, len(names))
            for i, n in enumerate(names):
                out[n] = out.get(n, 0) + share + (1 if i < rem else 0)
        else:
            raise ValueError(raw)
    return out


def summary(lines):
    b = balances(lines)
    rows = []
    for n in sorted(b):
        c = b[n]
        rows.append(f"{n}: {'+' if c >= 0 else '-'}{abs(c) // 100}.{abs(c) % 100:02d}")
    return "\n".join(rows)
