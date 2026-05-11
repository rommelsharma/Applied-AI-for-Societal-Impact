"""Smoke test: package imports without executing training."""


def test_version():
    import jaguar_reid

    assert hasattr(jaguar_reid, "__version__")
    assert isinstance(jaguar_reid.__version__, str)
