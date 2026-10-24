from pkg.ledger import balances, summary


def test_pays():
    assert balances(["ana pays bo 12.50"]) == {"ana": -1250, "bo": 1250}


def test_split_remainder():
    assert balances(["ana splits 10 among bo,cy,di"]) == {"ana": -1000, "bo": 334, "cy": 333, "di": 333}


def test_summary():
    assert summary(["ana pays bo 0.05"]) == "ana: -0.05\nbo: +0.05"
