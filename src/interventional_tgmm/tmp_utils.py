from __future__ import annotations

import shutil
from pathlib import Path


def rewrite_tmp_relative_path(path_value: str, tmp_dir: str) -> str:
    path = Path(path_value)
    if path.is_absolute() or not path.parts:
        return path_value

    if path.parts[0].startswith("tmp"):
        return str(Path(tmp_dir) / path)
    return path_value


def cleanup_tmp_dir(tmp_dir: str, show_tmp: bool) -> None:
    if show_tmp:
        return
    path = Path(tmp_dir)
    if path.exists():
        shutil.rmtree(path)
