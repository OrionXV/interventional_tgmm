from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        return {}

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file '{path}' must contain a top-level mapping.")
    return payload


def collect_section_defaults(config: dict[str, Any], sections: Iterable[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for section in sections:
        section_payload = config.get(section, {})
        if section_payload is None:
            continue
        if not isinstance(section_payload, dict):
            raise ValueError(f"Config section '{section}' must be a mapping.")
        defaults.update(section_payload)
    return defaults


def apply_config_defaults(
    parser: Any,
    config_path: str | Path | None,
    sections: Iterable[str],
) -> None:
    if not config_path:
        return
    config = load_yaml_config(config_path)
    if not config:
        return

    raw_defaults = collect_section_defaults(config, sections)
    valid_keys = {action.dest for action in parser._actions}
    defaults = {key: value for key, value in raw_defaults.items() if key in valid_keys}
    if defaults:
        parser.set_defaults(**defaults)
