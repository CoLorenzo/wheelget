from __future__ import annotations

import os
import platform as _platform
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass

WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$",
    re.IGNORECASE,
)

_ARCH_CANON = {
    "amd64": "x86_64",
    "x64": "x86_64",
    "i386": "i686",
    "i686": "i686",
    "x86_64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "arm64",
}

_LEGACY_MANYLINUX = {"manylinux1": 5, "manylinux2010": 12, "manylinux2014": 17}


def _canon_arch(arch: str) -> str:
    return _ARCH_CANON.get(arch.lower(), arch.lower())


def _glibc_version() -> tuple[int, int] | None:
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        return None
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


@dataclass(frozen=True)
class Target:
    python: tuple[int, int]
    os: str
    arch: str
    glibc: tuple[int, int] | None = None

    @classmethod
    def current(cls, python: tuple[int, int] | None = None) -> Target:
        if python is None:
            python = (sys.version_info.major, sys.version_info.minor)
        if sys.platform.startswith("linux"):
            os_name = "linux"
        elif sys.platform == "darwin":
            os_name = "macos"
        elif os.name == "nt":
            os_name = "windows"
        else:
            os_name = sys.platform
        glibc = _glibc_version() if os_name == "linux" else None
        return cls(python=python, os=os_name, arch=_canon_arch(_platform.machine()), glibc=glibc)


@dataclass(frozen=True)
class Wheel:
    name: str
    version: str
    py_tags: tuple[str, ...]
    abi_tags: tuple[str, ...]
    plat_tags: tuple[str, ...]
    filename: str
    url: str
    sha256: str | None = None
    build: str | None = None

    @property
    def base_version(self) -> str:
        return self.version.split("+", 1)[0]

    @property
    def local_version(self) -> str | None:
        return self.version.split("+", 1)[1] if "+" in self.version else None


def parse_wheel(filename: str, url: str = "", sha256: str | None = None) -> Wheel | None:
    match = WHEEL_RE.match(filename)
    if match is None:
        return None
    return Wheel(
        name=match.group("name").lower(),
        version=match.group("version"),
        build=match.group("build"),
        py_tags=tuple(match.group("py").lower().split(".")),
        abi_tags=tuple(match.group("abi").lower().split(".")),
        plat_tags=tuple(match.group("plat").lower().split(".")),
        filename=filename,
        url=url,
        sha256=sha256,
    )


def _platform_score(tag: str, target: Target) -> int | None:
    tag = tag.lower()
    if tag == "any":
        return 1
    if target.os == "linux":
        if tag.startswith("linux_"):
            return 200 if _canon_arch(tag[6:]) == target.arch else None
        for name, minor in _LEGACY_MANYLINUX.items():
            if tag.startswith(name + "_"):
                if _canon_arch(tag[len(name) + 1:]) != target.arch:
                    return None
                if target.glibc is not None and (2, minor) > target.glibc:
                    return None
                return 1000 + minor
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)_(.+)", tag)
        if match:
            if _canon_arch(match.group(3)) != target.arch:
                return None
            needed = (int(match.group(1)), int(match.group(2)))
            if target.glibc is not None and needed > target.glibc:
                return None
            return 1000 + needed[1]
        return None
    if target.os == "macos":
        match = re.fullmatch(r"macosx_\d+_\d+_(.+)", tag)
        if match is None:
            return None
        arch = match.group(1)
        if arch == "universal2":
            return 220 if target.arch in ("x86_64", "arm64") else None
        return 200 if _canon_arch(arch) == target.arch else None
    if target.os == "windows":
        arch = {"win_amd64": "x86_64", "win_arm64": "arm64", "win32": "i686"}.get(tag)
        return 200 if arch == target.arch else None
    return None


def _py_abi_score(py: str, abi: str, target: Target) -> int | None:
    py = py.lower()
    abi = abi.lower()
    major, minor = target.python
    match = re.fullmatch(r"cp(\d)(\d+)", py)
    if match:
        wmajor, wminor = int(match.group(1)), int(match.group(2))
        if (wmajor, wminor) == (major, minor):
            if abi.startswith(py) or abi == "none":
                return 10000
            if abi == "abi3":
                return 9000
            return None
        if abi == "abi3" and wmajor == major and wminor < minor:
            return 8000 + wminor
        return None
    if abi != "none":
        return None
    if py == "py3":
        return 500
    match = re.fullmatch(r"py(\d)(\d+)", py)
    if match and int(match.group(1)) == major and int(match.group(2)) <= minor:
        return 500 + int(match.group(2))
    return None


def compatibility(wheel: Wheel, target: Target) -> tuple[int, int] | None:
    best_py = None
    for py in wheel.py_tags:
        for abi in wheel.abi_tags:
            score = _py_abi_score(py, abi, target)
            if score is not None and (best_py is None or score > best_py):
                best_py = score
    if best_py is None:
        return None
    best_plat = None
    for plat in wheel.plat_tags:
        score = _platform_score(plat, target)
        if score is not None and (best_plat is None or score > best_plat):
            best_plat = score
    if best_plat is None:
        return None
    return (best_py, best_plat)


def pick_wheel(wheels: Iterable[Wheel], target: Target) -> Wheel | None:
    best = None
    best_key = None
    for wheel in wheels:
        key = compatibility(wheel, target)
        if key is None:
            continue
        if best_key is None or key > best_key:
            best, best_key = wheel, key
    return best


def variant_number(variant: str) -> int | None:
    match = re.fullmatch(r"cu(\d+)", variant.lower())
    return int(match.group(1)) if match else None


def variant_cuda_major(variant: str) -> int | None:
    number = variant_number(variant)
    return None if number is None else number // 10


def compatible_variants(variants: Iterable[str], cuda_major: int) -> list[str]:
    usable = []
    for variant in variants:
        major = variant_cuda_major(variant)
        if major is not None and major <= cuda_major:
            usable.append(variant)
    usable.sort(key=lambda item: variant_number(item) or 0, reverse=True)
    return usable


def select_variant(variants: Iterable[str], cuda_major: int) -> str | None:
    usable = compatible_variants(variants, cuda_major)
    return usable[0] if usable else None
