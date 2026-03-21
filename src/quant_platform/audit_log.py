from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    payload: dict[str, Any]
