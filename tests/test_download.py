import hashlib

import pytest

from wheelget.http import FetchError, download


def test_download_from_file_url(tmp_path):
    source = tmp_path / "thing-1.0.whl"
    source.write_bytes(b"wheel-bytes")
    dest = tmp_path / "out" / "thing-1.0.whl"
    result = download(source.as_uri(), dest, quiet=True)
    assert result == dest
    assert dest.read_bytes() == b"wheel-bytes"
    assert not (tmp_path / "out" / "thing-1.0.whl.part").exists()


def test_download_verifies_sha256(tmp_path):
    source = tmp_path / "thing-1.0.whl"
    source.write_bytes(b"wheel-bytes")
    good = hashlib.sha256(b"wheel-bytes").hexdigest()
    dest = tmp_path / "out" / "good.whl"
    assert download(source.as_uri(), dest, quiet=True, sha256=good).exists()
    with pytest.raises(FetchError):
        download(source.as_uri(), tmp_path / "out" / "bad.whl", quiet=True, sha256="0" * 64)
    assert not (tmp_path / "out" / "bad.whl").exists()
    assert not (tmp_path / "out" / "bad.whl.part").exists()


def test_download_skips_existing(tmp_path):
    source = tmp_path / "src" / "thing-1.0.whl"
    source.parent.mkdir()
    source.write_bytes(b"new")
    dest = tmp_path / "thing-1.0.whl"
    dest.write_bytes(b"old")
    assert download(source.as_uri(), dest, quiet=True).read_bytes() == b"old"
    assert download(source.as_uri(), dest, quiet=True, force=True).read_bytes() == b"new"
