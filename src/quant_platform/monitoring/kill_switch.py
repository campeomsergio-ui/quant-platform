from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KillSwitchState:
    enabled: bool
    reason: str
    triggers: list[str] = field(default_factory=list)


def manual_kill(reason: str) -> KillSwitchState:
    return KillSwitchState(enabled=True, reason=reason, triggers=["manual_kill"])


def clear_kill() -> KillSwitchState:
    return KillSwitchState(enabled=False, reason="", triggers=[])
