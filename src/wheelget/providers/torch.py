from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from urllib.parse import urlsplit

from ..errors import WheelgetError
from ..http import FetchError, html_links, link_filename, link_sha256
from ..versions import is_prerelease, version_key
from ..wheels import Wheel, compatible_variants, parse_wheel, pick_wheel
from .base import Provider, Resolution

_PYPI_VERSIONS_URL = "https://pypi.org/simple/torch/"
_SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"
_INDEX_ROOT = "https://download.pytorch.org/whl"


def _join_variants(variants: Iterable[str]) -> str:
    ordered = sorted(variants, key=lambda item: int(item[2:]) if item[2:].isdigit() else 0, reverse=True)
    return ", ".join(ordered) if ordered else "none"


class TorchProvider(Provider):
    name = "torch"

    def latest_version(self, client) -> str:
        data = client.get_json(_PYPI_VERSIONS_URL, ttl=600, accept=_SIMPLE_JSON)
        versions = [item for item in data.get("versions", []) if not is_prerelease(item)]
        if not versions:
            raise WheelgetError("could not determine the latest torch version from PyPI")
        return max(versions, key=version_key)

    def variants(self, client) -> list[str]:
        url = f"{_INDEX_ROOT}/"
        text = client.get_text(url, ttl=3600)
        names = set()
        for href in html_links(text, url):
            name = urlsplit(href).path.rstrip("/").rsplit("/", 1)[-1]
            if re.fullmatch(r"cu\d+", name):
                names.add(name)
        return sorted(names)

    def wheels(self, client, variant: str) -> list[Wheel]:
        url = f"{_INDEX_ROOT}/{variant}/torch/"
        try:
            text = client.get_text(url, ttl=3600)
        except FetchError as exc:
            if exc.status == 404:
                return []
            raise
        wheels = []
        for href in html_links(text, url):
            filename = link_filename(href)
            if not filename.endswith(".whl"):
                continue
            wheel = parse_wheel(filename, url=href, sha256=link_sha256(href))
            if wheel is None or wheel.name != "torch":
                continue
            if wheel.local_version not in (None, variant):
                continue
            wheels.append(wheel)
        return wheels

    def _find(self, client, version: str, candidates: list[str], target) -> Resolution | None:
        for variant in candidates:
            matching = [item for item in self.wheels(client, variant) if item.base_version == version]
            pick = pick_wheel(matching, target)
            if pick is not None:
                return Resolution(package=self.name, version=version, variant=variant, wheel=pick)
        return None

    def _newest_available(self, client, candidates: list[str], target, skip: str) -> Resolution | None:
        best_key = None
        best = None
        for variant in candidates:
            groups = {}
            for wheel in self.wheels(client, variant):
                if is_prerelease(wheel.base_version):
                    continue
                groups.setdefault(wheel.base_version, []).append(wheel)
            for version, group in groups.items():
                if version == skip:
                    continue
                key = version_key(version)
                if best_key is not None and key <= best_key:
                    continue
                pick = pick_wheel(group, target)
                if pick is None:
                    continue
                best_key = key
                best = Resolution(package=self.name, version=version, variant=variant, wheel=pick)
        return best

    def resolve(self, client, *, version=None, variant=None, cuda, target) -> Resolution:
        variants = self.variants(client)
        if not variants:
            raise WheelgetError(f"no CUDA variants found at {_INDEX_ROOT}")
        if variant is not None:
            if variant not in variants:
                raise WheelgetError(
                    f"unknown torch variant {variant!r}; available: {_join_variants(variants)}"
                )
            candidates = [variant]
        else:
            candidates = compatible_variants(variants, cuda[0])
            if not candidates:
                raise WheelgetError(
                    f"no torch variant compatible with CUDA {cuda[0]}.x; "
                    f"available: {_join_variants(variants)}"
                )
        if version is not None:
            found = self._find(client, version, candidates, target)
            if found is None:
                raise WheelgetError(
                    f"torch {version} has no wheel for python "
                    f"{target.python[0]}.{target.python[1]} on {target.os}/{target.arch} "
                    f"in variants: {', '.join(candidates)}"
                )
            return found
        latest = self.latest_version(client)
        found = self._find(client, latest, candidates, target)
        if found is not None:
            return found
        fallback = self._newest_available(client, candidates, target, skip=latest)
        if fallback is None:
            raise WheelgetError(
                f"no torch wheel for python {target.python[0]}.{target.python[1]} "
                f"on {target.os}/{target.arch} compatible with CUDA {cuda[0]}.x"
            )
        return replace(
            fallback,
            note=(
                f"latest torch {latest} has no compatible CUDA {cuda[0]}.x wheel; "
                f"using {fallback.version}"
            ),
        )
