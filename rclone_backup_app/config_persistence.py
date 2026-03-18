"""
config_persistence.py — Save and load GUI settings for rclone Backup Configurator.

Stores:
  - last_remote     : last rclone remote name used
  - last_output     : last output folder
  - last_log        : last log folder
  - saved_jobs      : dict of job_name -> JobConfig.to_dict()

All stored in a JSON file in the user's AppData/Local directory.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from models import JobConfig

logger = logging.getLogger(__name__)

_APP_NAME = "RcloneBackupConfigurator"


def _settings_path() -> Path:
    """Return the path to the settings JSON file."""
    base = Path.home() / "AppData" / "Local" / _APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base / "settings.json"


def _load_raw() -> Dict[str, Any]:
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load settings (%s). Starting fresh.", exc)
        return {}


def _save_raw(data: Dict[str, Any]) -> None:
    try:
        _settings_path().write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.error("Could not save settings: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_last_remote() -> str:
    return _load_raw().get("last_remote", "")


def load_last_output() -> str:
    return _load_raw().get("last_output", "")


def load_last_log() -> str:
    return _load_raw().get("last_log", "")


def save_ui_state(remote: str, output: str, log: str) -> None:
    data = _load_raw()
    data["last_remote"] = remote
    data["last_output"] = output
    data["last_log"]    = log
    _save_raw(data)


def save_job(config: JobConfig) -> None:
    """Save or overwrite a named job config in settings."""
    data = _load_raw()
    jobs = data.setdefault("saved_jobs", {})
    jobs[config.job_name] = config.to_dict()
    _save_raw(data)


def load_job(name: str) -> Optional[JobConfig]:
    """Load a named job config. Returns None if not found."""
    jobs = _load_raw().get("saved_jobs", {})
    if name not in jobs:
        return None
    try:
        return JobConfig.from_dict(jobs[name])
    except Exception as exc:
        logger.warning("Could not deserialize job '%s': %s", name, exc)
        return None


def list_saved_jobs() -> list[str]:
    """Return names of all saved jobs."""
    return list(_load_raw().get("saved_jobs", {}).keys())


def delete_job(name: str) -> None:
    data = _load_raw()
    data.get("saved_jobs", {}).pop(name, None)
    _save_raw(data)
