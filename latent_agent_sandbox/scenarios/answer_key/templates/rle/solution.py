"""Run-length codec. See SPEC.md."""


def encode(s):
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        j = i
        while j < len(s) and s[j] == c:
            j += 1
        n = j - i
        piece = ("\\" + c) if (c.isdigit() or c == "\\") else c
        out.append(piece + (str(n) if n > 1 else ""))
        i = j
    return "".join(out)


def decode(s):
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 1
            c = s[i]
        i += 1
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        n = int(s[i:j]) if j > i else 1
        out.append(c * n)
        i = j
    return "".join(out)
