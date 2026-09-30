from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .errors import WheelgetError

CudaVersion = tuple[int, int]


def parse_cuda(value: str) -> CudaVersion:
    text = value.strip().lower()
    for prefix in ("cuda", "cu"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if re.fullmatch(r"\d{3}", text):
        number = int(text)
        return (number // 10, number % 10)
    match = re.match(r"(\d+)(?:\.(\d+))?", text)
    if match is None or not re.fullmatch(r"\d+(?:\.\d+)*", text):
        raise WheelgetError(
            f"invalid CUDA version {value!r}: use e.g. 12.6, 12 or cu126"
        )
    return (int(match.group(1)), int(match.group(2) or 0))


def detect_cuda() -> tuple[CudaVersion, str] | None:
    for detector, source in (
        (_from_nvidia_smi, "nvidia-smi"),
        (_from_nvcc, "nvcc"),
        (_from_cuda_home, "CUDA_HOME"),
    ):
        version = detector()
        if version is not None:
            return version, source
    return None


def _run(command) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _from_nvidia_smi() -> CudaVersion | None:
    output = _run(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _from_nvcc() -> CudaVersion | None:
    output = _run(["nvcc", "--version"])
    if not output:
        return None
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _from_cuda_home() -> CudaVersion | None:
    roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ]
    for root in roots:
        if not root:
            continue
        base = Path(root)
        version_json = base / "version.json"
        if version_json.is_file():
            try:
                data = json.loads(version_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            raw = data.get("cuda", {}).get("version") or data.get("version")
            match = re.match(r"(\d+)\.(\d+)", str(raw or ""))
            if match:
                return (int(match.group(1)), int(match.group(2)))
        version_txt = base / "version.txt"
        if version_txt.is_file():
            try:
                text = version_txt.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            match = re.search(r"CUDA Version\s+(\d+)\.(\d+)", text)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    return None
