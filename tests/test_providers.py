import json

import pytest

from wheelget.errors import WheelgetError
from wheelget.http import FetchError
from wheelget.providers.torch import TorchProvider
from wheelget.providers.vllm import VllmProvider
from wheelget.wheels import Target

LINUX = Target(python=(3, 12), os="linux", arch="x86_64", glibc=(2, 39))

RELEASES_URL = "https://api.github.com/repos/vllm-project/vllm/releases?per_page=100"
PYPI_URL = "https://pypi.org/simple/torch/"
WHL_ROOT = "https://download.pytorch.org/whl"


class FakeClient:
    def __init__(self, pages):
        self.pages = dict(pages)
        self.requested = []

    def get_text(self, url, **kwargs):
        self.requested.append(url)
        if url not in self.pages:
            raise FetchError(f"HTTP 404 for {url}", status=404)
        return self.pages[url]

    def get_json(self, url, **kwargs):
        return json.loads(self.get_text(url, **kwargs))


def releases(*versions):
    return json.dumps(
        [
            {"tag_name": f"v{version}", "draft": False, "prerelease": False}
            for version in versions
        ]
    )


def vllm_index(*variants):
    links = "".join(f'<a href="{variant}/">{variant}/</a>' for variant in variants)
    return f"<body>{links}</body>"


def vllm_wheel_page(*filenames):
    links = "".join(
        f'<a href="../../../abc/{filename}">{filename}</a>' for filename in filenames
    )
    return f"<body>{links}</body>"


def test_vllm_picks_highest_compatible_variant():
    pages = {
        RELEASES_URL: releases("0.30.0", "0.29.0"),
        "https://wheels.vllm.ai/0.30.0/": vllm_index("cpu", "cu129", "cu130"),
        "https://wheels.vllm.ai/0.30.0/cu129/vllm/": vllm_wheel_page(
            "vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
        "https://wheels.vllm.ai/0.30.0/cu130/vllm/": vllm_wheel_page(
            "vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
    }
    provider = VllmProvider()

    cuda13 = provider.resolve(
        FakeClient(pages), version=None, variant=None, cuda=(13, 0), target=LINUX
    )
    assert cuda13.version == "0.30.0"
    assert cuda13.variant == "cu130"
    assert cuda13.note is None

    cuda12 = provider.resolve(
        FakeClient(pages), version=None, variant=None, cuda=(12, 6), target=LINUX
    )
    assert cuda12.variant == "cu129"
    assert cuda12.wheel.url == (
        "https://wheels.vllm.ai/abc/vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    )


def test_vllm_falls_back_to_older_release():
    pages = {
        RELEASES_URL: releases("0.31.0", "0.30.0"),
        "https://wheels.vllm.ai/0.31.0/": vllm_index("cu130"),
        "https://wheels.vllm.ai/0.30.0/": vllm_index("cu129"),
        "https://wheels.vllm.ai/0.30.0/cu129/vllm/": vllm_wheel_page(
            "vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
    }
    resolution = VllmProvider().resolve(
        FakeClient(pages), version=None, variant=None, cuda=(12, 6), target=LINUX
    )
    assert resolution.version == "0.30.0"
    assert resolution.variant == "cu129"
    assert resolution.note is not None


def test_vllm_forced_variant_must_exist():
    pages = {
        RELEASES_URL: releases("0.30.0"),
        "https://wheels.vllm.ai/0.30.0/": vllm_index("cu129", "cu130"),
    }
    with pytest.raises(WheelgetError):
        VllmProvider().resolve(
            FakeClient(pages), version=None, variant="cu128", cuda=(12, 6), target=LINUX
        )


def torch_link(variant, version, py="312", plat="manylinux_2_28_x86_64"):
    filename = f"torch-{version}%2B{variant}-cp{py}-cp{py}-{plat}.whl"
    sha = "ab" * 32
    return f"https://download-r2.pytorch.org/whl/{variant}/{filename}#sha256={sha}"


def torch_page(*urls):
    return "<body>" + "".join(f'<a href="{url}"></a>' for url in urls) + "</body>"


def torch_common_pages():
    return {
        PYPI_URL: json.dumps({"versions": ["2.14.1", "2.9.1", "2.15.0rc1"]}),
        f"{WHL_ROOT}/": '<a href="cpu/">cpu/</a><a href="cu126/">cu126/</a><a href="cu132/">cu132/</a>',
        f"{WHL_ROOT}/cu126/torch/": torch_page(torch_link("cu126", "2.14.1")),
        f"{WHL_ROOT}/cu132/torch/": torch_page(torch_link("cu132", "2.14.1")),
    }


def test_torch_picks_variant_for_latest():
    pages = torch_common_pages()
    provider = TorchProvider()

    cuda12 = provider.resolve(
        FakeClient(pages), version=None, variant=None, cuda=(12, 6), target=LINUX
    )
    assert cuda12.version == "2.14.1"
    assert cuda12.variant == "cu126"
    assert cuda12.wheel.local_version == "cu126"
    assert cuda12.wheel.sha256 == "ab" * 32

    cuda13 = provider.resolve(
        FakeClient(pages), version=None, variant=None, cuda=(13, 0), target=LINUX
    )
    assert cuda13.variant == "cu132"


def test_torch_falls_back_to_newest_compatible_version():
    pages = {
        PYPI_URL: json.dumps({"versions": ["2.15.0", "2.14.1", "2.9.1"]}),
        f"{WHL_ROOT}/": '<a href="cu126/">cu126/</a><a href="cu129/">cu129/</a>',
        f"{WHL_ROOT}/cu129/torch/": torch_page(torch_link("cu129", "2.9.1")),
        f"{WHL_ROOT}/cu126/torch/": torch_page(torch_link("cu126", "2.14.1")),
    }
    resolution = TorchProvider().resolve(
        FakeClient(pages), version=None, variant=None, cuda=(12, 6), target=LINUX
    )
    assert resolution.version == "2.14.1"
    assert resolution.variant == "cu126"
    assert resolution.note is not None


def test_torch_pinned_version_and_variant():
    pages = torch_common_pages()
    resolution = TorchProvider().resolve(
        FakeClient(pages), version="2.14.1", variant="cu132", cuda=(12, 6), target=LINUX
    )
    assert resolution.variant == "cu132"
    assert resolution.wheel.url.startswith(
        "https://download-r2.pytorch.org/whl/cu132/torch-2.14.1%2Bcu132-"
    )


def test_torch_unknown_variant():
    pages = torch_common_pages()
    with pytest.raises(WheelgetError):
        TorchProvider().resolve(
            FakeClient(pages), version=None, variant="cu999", cuda=(12, 6), target=LINUX
        )
