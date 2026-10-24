from pkg.rle import decode, encode


def test_basic():
    assert encode("aaab") == "a3b"
    assert decode("a3b") == "aaab"


def test_digits_escaped():
    assert encode("x11") == "x\\12"
    assert decode("x\\12") == "x11"


def test_empty():
    assert encode("") == ""
    assert decode("") == ""
