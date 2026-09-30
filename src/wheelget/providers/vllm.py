from __future__ import annotations

import os
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

from ..errors import WheelgetError
from ..http import FetchError, html_links, link_filename, link_sha256
from ..wheels import Wheel, compatible_variants, parse_wheel, pick_wheel
from .base import Provider, Resolution

_RELEASES_URL = "https://api.github.com/repos/vllm-project/vllm/releases?per_page=100"
_INDEX_ROOT = "https://wheels.vllm.ai"
_MAX_RELEASE_LOOKBACK = 60
_TORCH_PIN_RE = re.compile(
    r"^Requires-Dist:\s*torch\s*==\s*([^;\s]+)", re.IGNORECASE | re.MULTILINE
)


def torch_pin_from_wheel(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = next(
                (name for name in archive.namelist() if name.endswith(".dist-info/METADATA")),
                None,
            )
            if metadata is None:
                return None
            text = archive.read(metadata).decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile):
        return None
    match = _TORCH_PIN_RE.search(text)
    return match.group(1) if match else None


def _join_variants(variants: Iterable[str]) -> str:
    ordered = sorted(variants, key=lambda item: int(item[2:]) if item[2:].isdigit() else 0, reverse=True)
    return ", ".join(ordered) if ordered else "none"


class VllmProvider(Provider):
    name = "vllm"

    def releases(self, client) -> list[str]:
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        releases = client.get_json(_RELEASES_URL, ttl=600, headers=headers)
        versions = []
        for release in releases:
            if release.get("draft") or release.get("prerelease"):
                continue
            tag = str(release.get("tag_name", "")).strip()
            version = tag.removeprefix("v")
            if version:
                versions.append(version)
        if not versions:
            raise WheelgetError("could not list vllm releases from GitHub")
        return versions

    def variants(self, client, version: str) -> list[str]:
        url = f"{_INDEX_ROOT}/{version}/"
        try:
            text = client.get_text(url, ttl=3600)
        except FetchError as exc:
            if exc.status == 404:
                return []
            raise
        names = set()
        for href in html_links(text, url):
            name = urlsplit(href).path.rstrip("/").rsplit("/", 1)[-1]
            if re.fullmatch(r"cu\d+", name):
                names.add(name)
        return sorted(names)

    def wheels(self, client, version: str, variant: str) -> list[Wheel]:
        url = f"{_INDEX_ROOT}/{version}/{variant}/vllm/"
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
            if wheel is not None and wheel.name.replace("_", "-") == "vllm":
                wheels.append(wheel)
        return wheels

    def resolve(self, client, *, version=None, variant=None, cuda, target) -> Resolution:
        if version is None:
            versions = self.releases(client)
            latest = versions[0]
        else:
            versions = [version]
            latest = version
        last_problem: str | None = None
        for candidate_version in versions[:_MAX_RELEASE_LOOKBACK]:
            available = self.variants(client, candidate_version)
            if not available:
                continue
            if variant is not None:
                if variant not in available:
                    raise WheelgetError(
                        f"variant {variant!r} is not available for vllm {candidate_version}; "
                        f"available: {_join_variants(available)}"
                    )
                candidates = [variant]
            else:
                candidates = compatible_variants(available, cuda[0])
                if not candidates:
                    last_problem = (
                        f"vllm {candidate_version} has no CUDA {cuda[0]}.x wheel "
                        f"(available: {_join_variants(available)})"
                    )
                    continue
            for candidate in candidates:
                wheels = self.wheels(client, candidate_version, candidate)
                pick = pick_wheel(wheels, target)
                if pick is not None:
                    note = None
                    if version is None and candidate_version != latest:
                        note = (
                            f"latest vllm release {latest} has no compatible CUDA variant; "
                            f"using {candidate_version}"
                        )
                    return Resolution(
                        package=self.name,
                        version=candidate_version,
                        variant=candidate,
                        wheel=pick,
                        note=note,
                    )
                if wheels:
                    last_problem = (
                        f"vllm {candidate_version} [{candidate}] has no wheel for "
                        f"python {target.python[0]}.{target.python[1]} "
                        f"on {target.os}/{target.arch}"
                    )
            if version is not None:
                break
        if version is not None:
            raise WheelgetError(last_problem or f"no vllm wheel found for {version}")
        checked = versions[: min(len(versions), _MAX_RELEASE_LOOKBACK)]
        detail = f"; last check: {last_problem}" if last_problem else ""
        raise WheelgetError(
            f"no vllm release with a CUDA {cuda[0]}.x wheel found "
            f"(checked {len(checked)} releases, from {checked[0]} to {checked[-1]}){detail}"
        )
