import pytest

from wheelget.cuda import parse_cuda
from wheelget.errors import WheelgetError


def test_parse_dotted():
    assert parse_cuda("12.6") == (12, 6)
    assert parse_cuda("12") == (12, 0)
    assert parse_cuda("11.8.0") == (11, 8)


def test_parse_variant_shorthand():
    assert parse_cuda("cu126") == (12, 6)
    assert parse_cuda("cu130") == (13, 0)
    assert parse_cuda("cu118") == (11, 8)
    assert parse_cuda("cuda12.6") == (12, 6)
    assert parse_cuda("126") == (12, 6)


def test_parse_invalid():
    with pytest.raises(WheelgetError):
        parse_cuda("banana")
