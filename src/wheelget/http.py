from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

from . import __version__
from .errors import WheelgetError

USER_AGENT = f"wheelget/{__version__}"
_LINK_RE = re.compile(r"""<a\s[^>]*?href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


class FetchError(WheelgetError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "wheelget"


def html_links(html: str, base_url: str) -> list[str]:
    return [urljoin(base_url, match.group(1)) for match in _LINK_RE.finditer(html)]


def link_filename(url: str) -> str:
    return unquote(urlsplit(url).path.rsplit("/", 1)[-1])


def link_sha256(url: str) -> str | None:
    values = parse_qs(urlsplit(url).fragment).get("sha256")
    return values[0] if values else None


class HttpClient:
    def __init__(self, *, refresh: bool = False, quiet: bool = False, timeout: float = 60.0):
        self.refresh = refresh
        self.quiet = quiet
        self.timeout = timeout

    def get_bytes(
        self,
        url: str,
        *,
        ttl: float = 3600.0,
        cache: bool = True,
        accept: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        path = cache_dir() / hashlib.sha256(url.encode("utf-8")).hexdigest()
        if (
            cache
            and not self.refresh
            and path.is_file()
            and time.time() - path.stat().st_mtime < ttl
        ):
            return path.read_bytes()
        request_headers = {"User-Agent": USER_AGENT}
        if accept:
            request_headers["Accept"] = accept
        if headers:
            request_headers.update(headers)
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read()
        except HTTPError as exc:
            raise FetchError(f"HTTP {exc.code} for {url}", status=exc.code) from exc
        except URLError as exc:
            raise FetchError(f"cannot reach {url}: {exc.reason}") from exc
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return data

    def get_text(self, url: str, **kwargs: Any) -> str:
        return self.get_bytes(url, **kwargs).decode("utf-8", "replace")

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return json.loads(self.get_text(url, **kwargs))


def download(
    url: str,
    dest: Path,
    *,
    quiet: bool = False,
    force: bool = False,
    sha256: str | None = None,
) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    show_progress = not quiet and sys.stderr.isatty()
    digest = hashlib.sha256()
    request = Request(url.partition("#")[0], headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=600) as response, open(tmp, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if show_progress:
                    _progress(done, total)
    except HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"HTTP {exc.code} while downloading {url}", status=exc.code) from exc
    except URLError as exc:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"cannot download {url}: {exc.reason}") from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    if sha256 and digest.hexdigest() != sha256.lower():
        tmp.unlink(missing_ok=True)
        raise FetchError(f"checksum mismatch for {dest.name}")
    os.replace(tmp, dest)
    return dest


def _progress(done: int, total: int) -> None:
    if total:
        sys.stderr.write(
            f"\r  {100.0 * done / total:5.1f}%  {done / 1e6:8.1f} / {total / 1e6:.1f} MB"
        )
    else:
        sys.stderr.write(f"\r  {done / 1e6:8.1f} MB")
    sys.stderr.flush()
