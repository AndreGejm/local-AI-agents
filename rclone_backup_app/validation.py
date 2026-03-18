"""
validation.py — Pre-generation validation for rclone Backup Configurator.

Call validate_job(config) before any script generation.
Returns ValidationResult with all errors collected (not fail-fast).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from models import BackupMode, Frequency, JobConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BWLIMIT_RE   = re.compile(r"^\d+[KMGT]?$")
TIME_RE      = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_job(config: JobConfig) -> ValidationResult:
    """
    Validate all fields in a JobConfig.
    Collects all errors — does not stop at the first failure.
    Returns a ValidationResult; check .is_valid or bool(result).
    """
    errors: List[str] = []

    # -- Job name --
    if not config.job_name.strip():
        errors.append("Job name is required.")
    elif not SAFE_NAME_RE.match(config.job_name):
        errors.append(
            "Job name may only contain letters, digits, hyphens, and underscores."
        )

    # -- Source folders --
    if not config.source_folders:
        errors.append("At least one source folder is required.")
    else:
        for src in config.source_folders:
            p = Path(src)
            if not p.exists():
                errors.append(f"Source folder does not exist: {p}")
            elif not p.is_dir():
                errors.append(f"Source path is not a directory: {p}")

    # -- Remote --
    if not config.remote_name.strip():
        errors.append("Remote name is required.")
    elif " " in config.remote_name:
        errors.append("Remote name must not contain spaces.")

    if not config.remote_path.strip():
        errors.append("Remote path is required.")

    # -- rclone numeric flags --
    if config.retries < 1:
        errors.append("Retries must be >= 1.")
    if config.low_level_retries < 1:
        errors.append("Low-level retries must be >= 1.")
    if config.checkers < 1:
        errors.append("Checkers must be >= 1.")
    if config.transfers < 1:
        errors.append("Transfers must be >= 1.")
    if config.lock_timeout_seconds < 0:
        errors.append("Lock timeout must be >= 0.")

    # -- Bandwidth limit (optional) --
    bwlimit = config.bwlimit.strip()
    if bwlimit and not BWLIMIT_RE.match(bwlimit):
        errors.append(
            "Bandwidth limit must be a number optionally followed by K, M, G, or T "
            "(e.g. '50M', '1G'). Leave empty for no limit."
        )

    # -- Exclude-from file (optional) --
    if config.excludes.exclude_from_file:
        ef = Path(config.excludes.exclude_from_file)
        if not ef.exists():
            errors.append(f"Exclude-from file does not exist: {ef}")
        elif not ef.is_file():
            errors.append(f"Exclude-from path is not a file: {ef}")

    # -- Scheduler --
    sched = config.scheduler
    if sched.frequency != Frequency.MANUAL:
        if not TIME_RE.match(sched.trigger_time):
            errors.append(
                f"Trigger time '{sched.trigger_time}' must be in HH:MM format."
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
