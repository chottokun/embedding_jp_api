from app.main import _apply_prefix


def test_apply_prefix_empty_prefix():
    """
    If the prefix is empty or None (though the type hint is str, we can test empty str),
    the inputs should be returned completely unmodified.
    """
    inputs = ["hello", "world", "prefix: hello"]
    assert _apply_prefix(inputs, "") == inputs


def test_apply_prefix_with_unprefixed_inputs():
    """
    If inputs do not start with the prefix, they should have the prefix prepended.
    """
    inputs = ["hello", "world"]
    prefix = "prefix: "
    expected = ["prefix: hello", "prefix: world"]
    assert _apply_prefix(inputs, prefix) == expected


def test_apply_prefix_with_already_prefixed_inputs():
    """
    If inputs already start with the prefix, they should remain unchanged.
    """
    inputs = ["prefix: hello", "prefix: world"]
    prefix = "prefix: "
    expected = ["prefix: hello", "prefix: world"]
    assert _apply_prefix(inputs, prefix) == expected


def test_apply_prefix_mixed_inputs():
    """
    If inputs are a mix of prefixed and unprefixed strings,
    only the unprefixed ones should get the prefix prepended.
    """
    inputs = ["hello", "prefix: world", "test", "prefix: test2"]
    prefix = "prefix: "
    expected = ["prefix: hello", "prefix: world", "prefix: test", "prefix: test2"]
    assert _apply_prefix(inputs, prefix) == expected


def test_apply_prefix_empty_list():
    """
    If the inputs list is empty, it should return an empty list.
    """
    assert _apply_prefix([], "prefix: ") == []


def test_apply_prefix_with_empty_string_input():
    """
    If an input is an empty string, it should get prefixed because it doesn't
    startswith the prefix (unless the prefix itself is empty).
    """
    inputs = ["", "hello"]
    prefix = "prefix: "
    expected = ["prefix: ", "prefix: hello"]
    assert _apply_prefix(inputs, prefix) == expected


def test_apply_prefix_prefix_matches_input_partially():
    """
    If the input starts with a substring of the prefix but not the full prefix,
    the full prefix should still be prepended.
    """
    inputs = ["prehello", "pre", "prefix: hello"]
    prefix = "prefix: "
    expected = ["prefix: prehello", "prefix: pre", "prefix: hello"]
    assert _apply_prefix(inputs, prefix) == expected
