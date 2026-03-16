from __future__ import annotations

import hashlib
import json
from typing import Any


def _to_canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_to_canonical(item) for item in value]
    return value


def stable_hash(value: Any, *, namespace: str = "") -> str:
    payload = {
        "namespace": namespace,
        "value": _to_canonical(value),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

