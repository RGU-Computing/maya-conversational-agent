from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

_DEFAULT_CONFIG = Path(__file__).with_name("config.jsonl")
_CONFIG_PATH = Path(os.getenv("MAYA_AGENTS_CONFIG", str(_DEFAULT_CONFIG)))

def _load_specs(path: Path = _CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
    """Read strict JSON or JSONL without rewriting text inside partner prompts."""
    raw = path.read_text(encoding="utf-8")
    try:
        objects = [json.loads(raw)]
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            raise
        objects = [json.loads(line) for line in raw.splitlines() if line.strip()]
    specs: Dict[str, Dict[str, Any]] = {}
    for data in objects:
        if not isinstance(data, dict):
            raise ValueError("Agent configuration must contain objects keyed by agent ID.")
        for agent_id, spec in data.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Agent '{agent_id}' must map to an object.")
            if agent_id in specs:
                raise ValueError(f"Duplicate agent ID: {agent_id!r}.")
            if "config_path" in spec:
                spec["config_path"] = str((path.parent / spec["config_path"]).resolve())
            specs[agent_id] = spec
    return specs


_SPECS = _load_specs()


def load_agent_specs() -> Dict[str, Dict[str, Any]]:
    return _SPECS
