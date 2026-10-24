"""Number formatting helpers."""


def cents(n):
    sign = "-" if n < 0 else ""
    return f"{sign}{abs(n) // 100}.{abs(n) % 100:02d}"


def thousands(n):
    return f"{n:,}"
