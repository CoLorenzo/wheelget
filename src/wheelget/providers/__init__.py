from __future__ import annotations

from ..errors import WheelgetError
from .base import Provider, Resolution
from .torch import TorchProvider
from .vllm import VllmProvider

__all__ = ["Provider", "Resolution", "get_provider"]

_PROVIDERS = {
    "vllm": VllmProvider,
    "torch": TorchProvider,
    "pytorch": TorchProvider,
}


def get_provider(name: str) -> Provider:
    try:
        factory = _PROVIDERS[name.strip().lower()]
    except KeyError:
        known = ", ".join(sorted({"vllm", "torch"}))
        raise WheelgetError(
            f"unsupported package {name!r}; supported: {known}"
        ) from None
    return factory()
