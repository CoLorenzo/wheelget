import zipfile
from pathlib import Path

from wheelget.cli import main
from wheelget.errors import WheelgetError
from wheelget.providers.base import Resolution
from wheelget.providers.vllm import torch_pin_from_wheel
from wheelget.wheels import Wheel


def make_wheel(path: Path, metadata: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("thing-1.0.dist-info/METADATA", metadata)
    return path


def fake_wheel(name, version, filename, source: Path) -> Wheel:
    return Wheel(
        name=name,
        version=version,
        py_tags=("cp38",),
        abi_tags=("abi3",),
        plat_tags=("manylinux_2_28_x86_64",),
        filename=filename,
        url=source.as_uri(),
    )


class FakeVllmProvider:
    def __init__(self, resolution):
        self.resolution = resolution

    def resolve(self, client, *, version=None, variant=None, cuda=None, target=None):
        return self.resolution


class FakeTorchProvider:
    def __init__(self, wheel, fail_variants=()):
        self.wheel = wheel
        self.fail_variants = set(fail_variants)
        self.calls = []

    def resolve(self, client, *, version=None, variant=None, cuda=None, target=None):
        self.calls.append((version, variant))
        if variant in self.fail_variants:
            raise WheelgetError(f"no torch {version} for variant {variant}")
        return Resolution(package="torch", version=version, variant=variant or "cu126", wheel=self.wheel)


def test_torch_pin_from_wheel(tmp_path):
    pinned = make_wheel(
        tmp_path / "vllm.whl",
        "Metadata-Version: 2.1\nRequires-Dist: torch==2.13.0\nRequires-Dist: torchvision==0.28.0\n",
    )
    assert torch_pin_from_wheel(pinned) == "2.13.0"
    unpinned = make_wheel(tmp_path / "plain.whl", "Metadata-Version: 2.1\nName: vllm\n")
    assert torch_pin_from_wheel(unpinned) is None
    not_a_zip = tmp_path / "broken.whl"
    not_a_zip.write_bytes(b"not a zip")
    assert torch_pin_from_wheel(not_a_zip) is None


def test_get_vllm_downloads_pinned_torch(tmp_path, monkeypatch, capsys):
    vllm_source = make_wheel(
        tmp_path / "src" / "vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
        "Metadata-Version: 2.1\nRequires-Dist: torch==2.13.0\n",
    )
    torch_source = tmp_path / "src" / "torch-2.13.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"
    torch_source.write_bytes(b"torch-wheel")
    vllm_wheel = fake_wheel("vllm", "0.30.0+cu129", vllm_source.name, vllm_source)
    torch_wheel = fake_wheel("torch", "2.13.0+cu129", torch_source.name, torch_source)
    torch_provider = FakeTorchProvider(torch_wheel)
    monkeypatch.setattr(
        "wheelget.cli.get_provider",
        lambda name: FakeVllmProvider(
            Resolution(package="vllm", version="0.30.0", variant="cu129", wheel=vllm_wheel)
        ),
    )
    monkeypatch.setattr("wheelget.cli.TorchProvider", lambda: torch_provider)

    out = tmp_path / "out"
    assert main(["get", "vllm", "--cuda", "12.9", "-o", str(out), "-q"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert Path(lines[0]).name == vllm_source.name
    assert Path(lines[1]).name == torch_source.name
    assert Path(lines[1]).read_bytes() == b"torch-wheel"
    assert torch_provider.calls == [("2.13.0", "cu129")]


def test_get_vllm_torch_companion_falls_back_to_other_variant(tmp_path, monkeypatch, capsys):
    vllm_source = make_wheel(
        tmp_path / "src" / "vllm-0.30.0+cu132-cp38-abi3-manylinux_2_28_x86_64.whl",
        "Metadata-Version: 2.1\nRequires-Dist: torch==2.13.0\n",
    )
    torch_source = tmp_path / "src" / "torch-2.13.0+cu130-cp313-cp313-manylinux_2_28_x86_64.whl"
    torch_source.write_bytes(b"torch-wheel")
    vllm_wheel = fake_wheel("vllm", "0.30.0+cu132", vllm_source.name, vllm_source)
    torch_wheel = fake_wheel("torch", "2.13.0+cu130", torch_source.name, torch_source)
    torch_provider = FakeTorchProvider(torch_wheel, fail_variants={"cu132"})
    monkeypatch.setattr(
        "wheelget.cli.get_provider",
        lambda name: FakeVllmProvider(
            Resolution(package="vllm", version="0.30.0", variant="cu132", wheel=vllm_wheel)
        ),
    )
    monkeypatch.setattr("wheelget.cli.TorchProvider", lambda: torch_provider)

    assert main(["get", "vllm", "--cuda", "13.0", "-o", str(tmp_path / "out"), "-q"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert Path(lines[1]).name == torch_source.name
    assert torch_provider.calls == [("2.13.0", "cu132"), ("2.13.0", None)]


def test_get_vllm_without_torch_pin(tmp_path, monkeypatch, capsys):
    vllm_source = make_wheel(
        tmp_path / "src" / "vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl",
        "Metadata-Version: 2.1\nName: vllm\n",
    )
    monkeypatch.setattr(
        "wheelget.cli.get_provider",
        lambda name: FakeVllmProvider(
            Resolution(
                package="vllm",
                version="0.30.0",
                variant="cu129",
                wheel=fake_wheel("vllm", "0.30.0", vllm_source.name, vllm_source),
            )
        ),
    )
    monkeypatch.setattr(
        "wheelget.cli.TorchProvider",
        lambda: (_ for _ in ()).throw(AssertionError("TorchProvider non deve essere usato")),
    )

    assert main(["get", "vllm", "--cuda", "12.9", "-o", str(tmp_path / "out"), "-q"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 1
