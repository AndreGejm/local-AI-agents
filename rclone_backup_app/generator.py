"""
generator.py — Artifact generator for rclone Backup Configurator.

Produces all output files for a JobConfig:
  - backup_<name>.py  (the runnable runtime script)
  - backup_<name>.bat (Windows launcher stub)
  - backup_<name>.json (config snapshot)
  - backup_<name>_task.txt (Task Scheduler command, delegated from scheduler.py)

Primary entry point: generate_artifacts(config, output_folder) -> GeneratorResult
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from models import BackupMode, JobConfig
from templates import LAUNCHER_BAT_TEMPLATE, RUNTIME_TEMPLATE


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GeneratorResult:
    success: bool
    output_folder: Path
    files_written: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quote(s: str) -> str:
    """Return s as a Python repr-string safe for embedding in generated script."""
    return repr(str(s))


def _list_repr(items: List[str]) -> str:
    """Return a Python list literal from a list of plain strings."""
    return "[" + ", ".join(repr(i) for i in items) + "]"


def _safe_write(path: Path, content: str, overwrite_prompt=None) -> None:
    """Write content to path; raise FileExistsError if path exists and overwrite is False."""
    if path.exists() and overwrite_prompt is not None and not overwrite_prompt(path):
        raise FileExistsError(f"File already exists (not overwritten): {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _build_template_vars(config: JobConfig) -> dict:
    """Build the substitution dict for RUNTIME_TEMPLATE.format_map()."""
    sources = [str(p) for p in config.source_folders]
    dest = f"{config.remote_name}:{config.remote_path}"

    return {
        "job_name":              config.job_name,
        "job_name_repr":         _quote(config.job_name),
        "source_folders_repr":   _list_repr(sources),
        "rclone_dest_repr":      _quote(dest),
        "log_folder_repr":       _quote(str(config.log_folder)),
        "backup_mode_repr":      _quote(config.backup_mode.value),
        "rclone_exe_repr":       _quote("rclone"),
        "retries":               config.retries,
        "low_level_retries":     config.low_level_retries,
        "checkers":              config.checkers,
        "transfers":             config.transfers,
        "log_level_repr":        _quote(config.log_level.value),
        "bwlimit_repr":          _quote(config.bwlimit.strip()),
        "checksum":              config.checksum,
        "exclude_patterns_repr": _list_repr(config.excludes.patterns),
        "exclude_from_repr":     _quote(
            str(config.excludes.exclude_from_file) if config.excludes.exclude_from_file else ""
        ),
        "post_copy_validation":  config.post_copy_validation,
        "lock_timeout_seconds":  config.lock_timeout_seconds,
        "retry_delay_seconds":   config.retry_delay_seconds,
    }


def render_script(config: JobConfig) -> str:
    """Render the runtime Python script for a JobConfig."""
    vars_ = _build_template_vars(config)
    return RUNTIME_TEMPLATE.format_map(vars_)


def render_launcher_bat(config: JobConfig, output_folder: Path) -> str:
    """Render the .bat launcher stub."""
    script_path = output_folder / f"backup_{config.job_name}.py"
    return LAUNCHER_BAT_TEMPLATE.format_map({
        "job_name":    config.job_name,
        "python_exe":  sys.executable,
        "script_path": str(script_path),
    })


def preview_rclone_command(config: JobConfig) -> str:
    """Return a human-readable rclone command preview (not a true subprocess list)."""
    dest = f"{config.remote_name}:{config.remote_path}"
    src  = str(config.source_folders[0]) if config.source_folders else "<source>"

    parts = [
        "rclone",
        "copy" if config.backup_mode == BackupMode.COPY else config.backup_mode.value,
        f'"{src}"',
        f'"{dest}"',
        f"--retries {config.retries}",
        f"--low-level-retries {config.low_level_retries}",
        f"--checkers {config.checkers}",
        f"--transfers {config.transfers}",
        f"--log-level {config.log_level.value}",
    ]
    if config.backup_mode == BackupMode.DRY_RUN:
        parts.append("--dry-run")
    if config.checksum:
        parts.append("--checksum")
    if config.bwlimit.strip():
        parts += [f"--bwlimit {config.bwlimit.strip()}"]
    for pat in config.excludes.patterns:
        parts.append(f'--exclude "{pat}"')
    if config.excludes.exclude_from_file:
        parts.append(f'--exclude-from "{config.excludes.exclude_from_file}"')

    return " \\\n  ".join(parts)


# ---------------------------------------------------------------------------
# Artifact generator
# ---------------------------------------------------------------------------

def generate_artifacts(
    config: JobConfig,
    output_folder: Optional[Path] = None,
    overwrite: bool = False,
) -> GeneratorResult:
    """
    Generate all output artifacts for a JobConfig.

    Parameters
    ----------
    config        : validated JobConfig
    output_folder : override the config.output_folder destination
    overwrite     : if True, silently overwrite existing files;
                    if False, raise FileExistsError on collision

    Returns a GeneratorResult with the list of written files.
    """
    folder = Path(output_folder) if output_folder else Path(config.output_folder)
    folder.mkdir(parents=True, exist_ok=True)

    name    = config.job_name
    written: List[Path] = []
    errors:  List[str]  = []

    # -- Runtime Python script --
    try:
        py_path = folder / f"backup_{name}.py"
        script  = render_script(config)
        _safe_write(py_path, script, overwrite_prompt=None if overwrite else _default_overwrite)
        written.append(py_path)
    except Exception as e:
        errors.append(f"Failed to write script: {e}")

    # -- .bat launcher --
    try:
        bat_path = folder / f"backup_{name}.bat"
        bat      = render_launcher_bat(config, folder)
        _safe_write(bat_path, bat, overwrite_prompt=None if overwrite else _default_overwrite)
        written.append(bat_path)
    except Exception as e:
        errors.append(f"Failed to write launcher: {e}")

    # -- Job config JSON snapshot --
    try:
        json_path = folder / f"backup_{name}.json"
        _safe_write(json_path, config.to_json(), overwrite_prompt=None if overwrite else _default_overwrite)
        written.append(json_path)
    except Exception as e:
        errors.append(f"Failed to write config snapshot: {e}")

    # -- Task Scheduler command text --
    try:
        from scheduler import build_task_command
        task_text = build_task_command(config, folder)
        task_path = folder / f"backup_{name}_task.txt"
        _safe_write(task_path, task_text, overwrite_prompt=None if overwrite else _default_overwrite)
        written.append(task_path)
    except Exception as e:
        errors.append(f"Failed to write task command: {e}")

    return GeneratorResult(
        success=len(errors) == 0,
        output_folder=folder,
        files_written=written,
        errors=errors,
    )


def _default_overwrite(path: Path) -> bool:
    """Default overwrite policy: never silently overwrite. Let caller handle."""
    return False
