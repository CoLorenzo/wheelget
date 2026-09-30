from wheelget.wheels import (
    Target,
    compatible_variants,
    parse_wheel,
    pick_wheel,
    select_variant,
    variant_cuda_major,
)

LINUX = Target(python=(3, 12), os="linux", arch="x86_64", glibc=(2, 39))
OLD_GLIBC = Target(python=(3, 12), os="linux", arch="x86_64", glibc=(2, 17))


def test_parse_torch_wheel():
    wheel = parse_wheel(
        "torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl",
        url="https://example.invalid/torch.whl",
        sha256="abc",
    )
    assert wheel is not None
    assert wheel.name == "torch"
    assert wheel.base_version == "2.14.1"
    assert wheel.local_version == "cu126"
    assert wheel.sha256 == "abc"


def test_parse_vllm_wheel():
    wheel = parse_wheel("vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl")
    assert wheel is not None
    assert wheel.name == "vllm"
    assert wheel.base_version == "0.30.0"
    assert wheel.local_version is None


def test_abi3_and_manylinux():
    wheel = parse_wheel("vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl")
    assert pick_wheel([wheel], LINUX) is wheel
    assert pick_wheel([wheel], OLD_GLIBC) is None


def test_exact_cpython_beats_generic():
    generic = parse_wheel("thing-1.0-py3-none-any.whl")
    exact = parse_wheel("thing-1.0-cp312-cp312-linux_x86_64.whl")
    wrong = parse_wheel("thing-1.0-cp311-cp311-linux_x86_64.whl")
    assert pick_wheel([generic, exact], LINUX) is exact
    assert pick_wheel([generic, wrong], LINUX) is generic


def test_platform_mismatch():
    windows = parse_wheel("torch-2.14.1+cu126-cp312-cp312-win_amd64.whl")
    aarch64 = parse_wheel("torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_aarch64.whl")
    assert pick_wheel([windows, aarch64], LINUX) is None


def test_select_variant():
    variants = ["cu128", "cu129", "cu130"]
    assert select_variant(variants, 13) == "cu130"
    assert select_variant(variants, 12) == "cu129"
    assert select_variant(variants, 11) is None
    assert compatible_variants(variants, 12) == ["cu129", "cu128"]


def test_variant_cuda_major():
    assert variant_cuda_major("cu118") == 11
    assert variant_cuda_major("cu130") == 13
    assert variant_cuda_major("cu132") == 13
    assert variant_cuda_major("cpu") is None
