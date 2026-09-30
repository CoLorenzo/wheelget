from wheelget.versions import is_prerelease, version_key


def test_version_key_numeric_order():
    assert version_key("2.9.1") < version_key("2.14.1")
    assert version_key("2.14.1") < version_key("2.15.0")
    assert version_key("2.14.1") < version_key("2.14.1+cu126")


def test_is_prerelease():
    assert is_prerelease("2.15.0rc1")
    assert is_prerelease("2.15.0a0")
    assert is_prerelease("2.15.0.dev20260101")
    assert not is_prerelease("2.15.0")
    assert not is_prerelease("2.14.1+cu126")
