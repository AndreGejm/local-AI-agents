# rclone Backup Configurator — README

## Launch

```
python main.py
```
Requires Python 3.11+ and rclone installed and in PATH.

## What it generates

| File | Purpose |
|------|---------|
| `backup_<name>.py` | Runnable backup script — schedule this |
| `backup_<name>.bat` | Windows launcher stub |
| `backup_<name>.json` | Job config snapshot |
| `backup_<name>_task.txt` | Task Scheduler commands (schtasks + PowerShell) |

## Task Scheduler (quick start)

1. Generate the job in the GUI
2. Click **Copy Task Cmd** to copy the schtasks command
3. Paste into an elevated Command Prompt and run

## How retries work

The runtime script retries the full rclone copy up to `retries` times per
source folder. Between each attempt it waits `retry_delay_seconds`. The
`--low-level-retries` flag is passed to rclone for transient network errors
within a single run.

## How locking works

A `.lock` file is written to the log folder before the backup runs.
If another instance is launched while the lock exists, it checks whether
the holding PID is still alive and whether the lock is older than
`lock_timeout_seconds`. A stale lock is removed and the new run proceeds.
A live lock aborts the new run with exit code 3.

## How validation works

After a copy run, if **Post-copy validation** is enabled, the script runs
`rclone check source destination` per source folder and reports mismatches.

## Safety defaults

- Mode defaults to `copy` — never `sync`
- `--dry-run` mode makes zero changes
- GUI warns before overwriting generated files
- `allow_sync` field exists in the model but the GUI does not expose it —
  it must be explicitly set in code to protect against accidental destructive runs
