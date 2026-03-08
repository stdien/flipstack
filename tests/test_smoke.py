"""Smoke test to ensure test suite runs."""


def test_import() -> None:
    """Verify flipstack package imports."""
    import flipstack

    assert flipstack is not None
