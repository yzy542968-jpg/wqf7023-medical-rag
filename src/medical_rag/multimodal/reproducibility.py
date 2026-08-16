from __future__ import annotations

import json


def json_content_equal(left: bytes, right: bytes) -> bool:
    """Compare JSON values while ignoring encoding whitespace and line endings."""
    try:
        return json.loads(left.decode("utf-8")) == json.loads(right.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
