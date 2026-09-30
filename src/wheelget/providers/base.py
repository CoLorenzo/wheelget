from __future__ import annotations

from dataclasses import dataclass

from ..wheels import Wheel


@dataclass(frozen=True)
class Resolution:
    package: str
    version: str
    variant: str | None
    wheel: Wheel
    note: str | None = None


class Provider:
    name = ""

    def resolve(self, client, *, version, variant, cuda, target) -> Resolution:
        raise NotImplementedError
