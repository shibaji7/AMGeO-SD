import amgeo_sd


def test_version():
    assert amgeo_sd.__version__


def test_dependencies_import():
    import aacgmv2, pydarn, pydarnio, sklearn, xarray  # noqa: F401, E401
