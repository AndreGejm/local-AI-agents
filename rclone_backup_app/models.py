"""
models.py — Core data models for rclone Backup Configurator.

All types used throughout gui.py, generator.py, scheduler.py, and validation.py
are defined here. Pure stdlib, no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BackupMode(str, Enum):
    COPY      = "copy"
    DRY_RUN   = "dry_run"
    CHECK_ONLY = "check_only"


class Frequency(str, Enum):
    MANUAL  = "manual"
    HOURLY  = "hourly"
    DAILY   = "daily"
    WEEKLY  = "weekly"


class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """Windows Task Scheduler settings for a job."""
    frequency: Frequency = Frequency.MANUAL
    # HH:MM for daily/weekly
    trigger_time: str = "02:00"
    # Monday–Sunday for weekly
    day_of_week: str = "Monday"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency": self.frequency.value,
            "trigger_time": self.trigger_time,
            "day_of_week": self.day_of_week,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SchedulerConfig:
        return cls(
            frequency=Frequency(d.get("frequency", "manual")),
            trigger_time=d.get("trigger_time", "02:00"),
            day_of_week=d.get("day_of_week", "Monday"),
        )


@dataclass
class ExcludeConfig:
    """Exclusion rules applied to every source folder."""
    patterns: List[str] = field(default_factory=list)
    # Path to a file containing one pattern per line (passed to --exclude-from)
    exclude_from_file: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patterns": list(self.patterns),
            "exclude_from_file": str(self.exclude_from_file) if self.exclude_from_file else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExcludeConfig:
        ef = d.get("exclude_from_file")
        return cls(
            patterns=list(d.get("patterns", [])),
            exclude_from_file=Path(ef) if ef else None,
        )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@dataclass
class JobConfig:
    """
    Complete configuration for one backup job.

    This is the single source of truth passed around the application.
    generator.py consumes it to produce the backup script and all artifacts.
    scheduler.py consumes it to produce Task Scheduler commands.
    """

    # -- Identity --
    job_name: str = ""

    # -- Source --
    source_folders: List[Path] = field(default_factory=list)

    # -- Destination --
    remote_name: str = ""           # e.g. "gdrive"
    remote_path: str = ""           # e.g. "Backups/MyPC"

    # -- Output paths --
    output_folder: Path = field(default_factory=lambda: Path("."))
    log_folder: Path = field(default_factory=lambda: Path("logs"))

    # -- Mode --
    backup_mode: BackupMode = BackupMode.COPY

    # -- rclone flags --
    retries: int = 3
    low_level_retries: int = 10
    checkers: int = 8
    transfers: int = 4
    log_level: LogLevel = LogLevel.INFO
    bwlimit: str = ""               # e.g. "10M" — empty = no limit
    checksum: bool = False
    # Allow sync ONLY if explicitly enabled behind a warning dialog
    allow_sync: bool = False

    # -- Exclude rules --
    excludes: ExcludeConfig = field(default_factory=ExcludeConfig)

    # -- Stability --
    # seconds to wait for a stale lock before aborting
    lock_timeout_seconds: int = 3600
    # retry delay in seconds between top-level retries in the runtime script
    retry_delay_seconds: int = 60

    # -- Validation --
    post_copy_validation: bool = False

    # -- Scheduler --
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "source_folders": [str(p) for p in self.source_folders],
            "remote_name": self.remote_name,
            "remote_path": self.remote_path,
            "output_folder": str(self.output_folder),
            "log_folder": str(self.log_folder),
            "backup_mode": self.backup_mode.value,
            "retries": self.retries,
            "low_level_retries": self.low_level_retries,
            "checkers": self.checkers,
            "transfers": self.transfers,
            "log_level": self.log_level.value,
            "bwlimit": self.bwlimit,
            "checksum": self.checksum,
            "allow_sync": self.allow_sync,
            "excludes": self.excludes.to_dict(),
            "lock_timeout_seconds": self.lock_timeout_seconds,
            "retry_delay_seconds": self.retry_delay_seconds,
            "post_copy_validation": self.post_copy_validation,
            "scheduler": self.scheduler.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> JobConfig:
        return cls(
            job_name=d.get("job_name", ""),
            source_folders=[Path(p) for p in d.get("source_folders", [])],
            remote_name=d.get("remote_name", ""),
            remote_path=d.get("remote_path", ""),
            output_folder=Path(d.get("output_folder", ".")),
            log_folder=Path(d.get("log_folder", "logs")),
            backup_mode=BackupMode(d.get("backup_mode", "copy")),
            retries=int(d.get("retries", 3)),
            low_level_retries=int(d.get("low_level_retries", 10)),
            checkers=int(d.get("checkers", 8)),
            transfers=int(d.get("transfers", 4)),
            log_level=LogLevel(d.get("log_level", "INFO")),
            bwlimit=d.get("bwlimit", ""),
            checksum=bool(d.get("checksum", False)),
            allow_sync=bool(d.get("allow_sync", False)),
            excludes=ExcludeConfig.from_dict(d.get("excludes", {})),
            lock_timeout_seconds=int(d.get("lock_timeout_seconds", 3600)),
            retry_delay_seconds=int(d.get("retry_delay_seconds", 60)),
            post_copy_validation=bool(d.get("post_copy_validation", False)),
            scheduler=SchedulerConfig.from_dict(d.get("scheduler", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> JobConfig:
        return cls.from_dict(json.loads(text))
