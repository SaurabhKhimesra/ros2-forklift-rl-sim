"""Locating packaged data files.

Configs, worlds and models are installed into ``share/forklift_gym_env`` and
found through the ament index, with a source-tree fallback so the package also
works as a plain pip install with no ROS present. Nothing here reads out of
``build/``: that is a colcon scratch directory, not a search path.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE = "forklift_gym_env"


def package_share_dir() -> Path | None:
    """The installed ``share/forklift_gym_env`` directory, if ROS is available."""
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(_PACKAGE))
    except Exception:
        return None


def _source_root() -> Path:
    # forklift_gym_env/paths.py -> forklift_gym_env/ -> src/forklift_gym_env/
    return Path(__file__).resolve().parent.parent


def search_paths() -> list[Path]:
    """Directories searched for packaged data, most specific first."""
    candidates: list[Path] = []
    if override := os.environ.get("FORKLIFT_GYM_ENV_DATA"):
        candidates.append(Path(override))
    if share := package_share_dir():
        candidates.append(share)
    candidates.append(_source_root())
    return [c for c in candidates if c.is_dir()]


def find_data_file(relative: str | Path) -> Path:
    """Resolve a packaged file, e.g. ``config/td3_kinematic.yaml``.

    Raises a ``FileNotFoundError`` that names every place that was searched --
    far more useful than the original's bare "No such file or directory".
    """
    relative = Path(relative)
    if relative.is_absolute() and relative.is_file():
        return relative
    tried = []
    for root in search_paths():
        for candidate in (root / relative, root / relative.name):
            tried.append(candidate)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"could not find packaged file {relative!s}. Looked in:\n  "
        + "\n  ".join(str(t) for t in tried)
        + "\nSet FORKLIFT_GYM_ENV_DATA to override the search root."
    )


def find_config(name: str | Path) -> Path:
    """Resolve a config by path, by bare name, or by name without ``.yaml``."""
    name = Path(name)
    if name.is_file():
        return name
    for attempt in (name, Path("config") / name, Path("config") / f"{name.stem}.yaml"):
        try:
            return find_data_file(attempt)
        except FileNotFoundError:
            continue
    available = sorted(p.name for root in search_paths() for p in (root / "config").glob("*.yaml"))
    raise FileNotFoundError(f"no config named {name!s}. Available: {available or 'none found'}")
