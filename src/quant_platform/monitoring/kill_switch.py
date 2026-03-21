from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KillSwitchState:
    enabled: bool
    reason: str
