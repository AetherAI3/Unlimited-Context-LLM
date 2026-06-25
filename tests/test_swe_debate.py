from bench.swe_debate import classify_complexity


def test_heavy_on_hard_keywords():
    assert classify_complexity("There is a security vulnerability in the auth check") == "heavy"
    assert classify_complexity("intermittent race condition causes a deadlock") == "heavy"


def test_heavy_on_long_problem():
    assert classify_complexity("x " * 400) == "heavy"


def test_light_on_short_simple():
    assert classify_complexity("Fix typo in the docstring of add()") == "light"


def test_medium_default():
    assert classify_complexity("Update the parser to handle empty input lists") == "medium"
