"""
gui.py — Tkinter GUI for rclone Backup Configurator.

Assembled from two local-expert triage passes (no escalation).
Orchestrator corrections applied: fixed backup_mode radio values,
corrected Combobox import, completed all event handler methods.
"""
from __future__ import annotations

import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional

import config_persistence as cp
from generator import GeneratorResult, generate_artifacts, preview_rclone_command
from models import BackupMode, ExcludeConfig, Frequency, JobConfig, LogLevel, SchedulerConfig
from validation import validate_job


class BackupConfigApp(tk.Frame):
    """Main application frame."""

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._source_paths: List[str] = []
        self._last_result: Optional[GeneratorResult] = None

        # -- tk variables --
        self.job_name         = tk.StringVar()
        self.remote_name      = tk.StringVar(value=cp.load_last_remote())
        self.remote_path      = tk.StringVar()
        self.output_folder    = tk.StringVar(value=cp.load_last_output())
        self.log_folder       = tk.StringVar(value=cp.load_last_log())
        self.backup_mode      = tk.StringVar(value=BackupMode.COPY.value)
        self.retries          = tk.IntVar(value=3)
        self.low_level_retries= tk.IntVar(value=10)
        self.checkers         = tk.IntVar(value=8)
        self.transfers        = tk.IntVar(value=4)
        self.log_level        = tk.StringVar(value=LogLevel.INFO.value)
        self.bwlimit          = tk.StringVar()
        self.checksum         = tk.BooleanVar()
        self.post_copy_val    = tk.BooleanVar()
        self.exclude_from_file= tk.StringVar()
        self.frequency        = tk.StringVar(value=Frequency.MANUAL.value)
        self.trigger_time     = tk.StringVar(value="02:00")
        self.day_of_week      = tk.StringVar(value="Monday")

        self._build_ui()
        self._load_saved_jobs_dropdown()

        # Auto-refresh preview when key fields change
        for var in (self.job_name, self.remote_name, self.remote_path,
                    self.backup_mode, self.retries, self.low_level_retries,
                    self.checkers, self.transfers, self.log_level,
                    self.bwlimit, self.checksum):
            var.trace_add("write", lambda *_: self.refresh_previews())

    # --------------------------------------------------------------------------
    # UI construction
    # --------------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_top_bar()
        content = tk.Frame(self)
        content.pack(fill=tk.BOTH, expand=True)
        self._build_left_panel(content)
        self._build_middle_panel(content)
        self._build_right_panel(content)
        self._build_bottom_previews()
        self._build_button_bar()

    def _build_top_bar(self) -> None:
        bar = tk.Frame(self)
        bar.pack(fill=tk.X, pady=(0, 6))

        tk.Label(bar, text="Job Name:").pack(side=tk.LEFT)
        tk.Entry(bar, textvariable=self.job_name, width=25).pack(side=tk.LEFT, padx=4)

        tk.Label(bar, text="Saved Jobs:").pack(side=tk.LEFT, padx=(16, 4))
        self._job_combo = ttk.Combobox(bar, values=[], width=22, state="readonly")
        self._job_combo.pack(side=tk.LEFT, padx=4)
        self._job_combo.bind("<<ComboboxSelected>>", lambda e: self.on_load_job())

        tk.Button(bar, text="Load",   command=self.on_load_job).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Save",   command=self.on_save_job).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Delete", command=self.on_delete_job).pack(side=tk.LEFT, padx=2)

    def _build_left_panel(self, parent: tk.Frame) -> None:
        fr = tk.LabelFrame(parent, text="Source Folders", padx=4, pady=4)
        fr.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))

        self._source_listbox = tk.Listbox(fr, width=32, height=16, selectmode=tk.SINGLE)
        sb = tk.Scrollbar(fr, orient=tk.VERTICAL, command=self._source_listbox.yview)
        self._source_listbox.configure(yscrollcommand=sb.set)
        self._source_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        btns = tk.Frame(fr)
        btns.pack(pady=4)
        tk.Button(btns, text="Add",    command=self.add_source_folder).pack(side=tk.LEFT, padx=2)
        tk.Button(btns, text="Remove", command=self.remove_source_folder).pack(side=tk.LEFT, padx=2)

    def _build_middle_panel(self, parent: tk.Frame) -> None:
        fr = tk.Frame(parent)
        fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def row(label: str, var: tk.Variable, browse=None) -> None:
            r = tk.Frame(fr)
            r.pack(fill=tk.X, pady=2)
            tk.Label(r, text=label, width=14, anchor=tk.W).pack(side=tk.LEFT)
            tk.Entry(r, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
            if browse:
                tk.Button(r, text="Browse", command=browse).pack(side=tk.LEFT)

        row("Remote Name:",  self.remote_name)
        row("Remote Path:",  self.remote_path)
        row("Output Folder:", self.output_folder, self.browse_output_folder)
        row("Log Folder:",   self.log_folder,    self.browse_log_folder)

        nb = ttk.Notebook(fr)
        nb.pack(fill=tk.BOTH, expand=True, pady=6)
        self._build_backup_tab(nb)
        self._build_exclude_tab(nb)
        self._build_scheduler_tab(nb)

    def _build_backup_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="Backup Settings")

        # Backup mode
        mf = tk.LabelFrame(tab, text="Backup Mode", padx=4, pady=4)
        mf.pack(fill=tk.X, pady=4)
        for label, val in [("Copy (safe default)", BackupMode.COPY.value),
                           ("Dry-run (no changes)", BackupMode.DRY_RUN.value),
                           ("Check only", BackupMode.CHECK_ONLY.value)]:
            tk.Radiobutton(mf, text=label, variable=self.backup_mode, value=val).pack(anchor=tk.W)

        # Flags grid
        fg = tk.Frame(tab)
        fg.pack(fill=tk.X)
        items = [
            ("Retries",           self.retries,           1, 20),
            ("Low-level Retries", self.low_level_retries, 1, 50),
            ("Checkers",          self.checkers,          1, 64),
            ("Transfers",         self.transfers,         1, 64),
        ]
        for i, (lbl, var, lo, hi) in enumerate(items):
            tk.Label(fg, text=lbl+":", anchor=tk.W, width=18).grid(row=i, column=0, sticky=tk.W, pady=1)
            tk.Spinbox(fg, from_=lo, to=hi, textvariable=var, width=6).grid(row=i, column=1, sticky=tk.W)

        r = len(items)
        tk.Label(fg, text="Log Level:", anchor=tk.W, width=18).grid(row=r, column=0, sticky=tk.W, pady=1)
        ttk.Combobox(fg, textvariable=self.log_level, values=[e.value for e in LogLevel],
                     width=10, state="readonly").grid(row=r, column=1, sticky=tk.W)

        r += 1
        tk.Label(fg, text="Bwlimit:", anchor=tk.W, width=18).grid(row=r, column=0, sticky=tk.W, pady=1)
        tk.Entry(fg, textvariable=self.bwlimit, width=10).grid(row=r, column=1, sticky=tk.W)
        tk.Label(fg, text="e.g. 50M, 1G — blank = unlimited", fg="grey").grid(row=r, column=2, sticky=tk.W, padx=4)

        r += 1
        tk.Checkbutton(fg, text="Use checksum verification", variable=self.checksum).grid(
            row=r, column=0, columnspan=3, sticky=tk.W)
        r += 1
        tk.Checkbutton(fg, text="Post-copy validation (rclone check)", variable=self.post_copy_val).grid(
            row=r, column=0, columnspan=3, sticky=tk.W)

    def _build_exclude_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="Exclude Rules")

        tk.Label(tab, text="Patterns (one per line):").pack(anchor=tk.W)
        self._exclude_text = tk.Text(tab, height=8, width=40)
        self._exclude_text.pack(fill=tk.BOTH, expand=True, pady=4)
        self._exclude_text.bind("<<Modified>>", lambda _: self.refresh_previews())

        ef = tk.Frame(tab)
        ef.pack(fill=tk.X, pady=2)
        tk.Label(ef, text="Exclude-from file:", width=16, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(ef, textvariable=self.exclude_from_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(ef, text="Browse", command=self.browse_exclude_file).pack(side=tk.LEFT)

    def _build_scheduler_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="Scheduler")

        tk.Label(tab, text="Frequency:").pack(anchor=tk.W)
        for label, val in [("Manual (no scheduled task)", Frequency.MANUAL.value),
                           ("Hourly", Frequency.HOURLY.value),
                           ("Daily",  Frequency.DAILY.value),
                           ("Weekly", Frequency.WEEKLY.value)]:
            tk.Radiobutton(tab, text=label, variable=self.frequency, value=val).pack(anchor=tk.W)

        tf = tk.Frame(tab)
        tf.pack(fill=tk.X, pady=4)
        tk.Label(tf, text="Trigger time (HH:MM):").pack(side=tk.LEFT)
        tk.Entry(tf, textvariable=self.trigger_time, width=8).pack(side=tk.LEFT, padx=4)

        df = tk.Frame(tab)
        df.pack(fill=tk.X, pady=2)
        tk.Label(df, text="Day of week:").pack(side=tk.LEFT)
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        ttk.Combobox(df, textvariable=self.day_of_week, values=days, width=12,
                     state="readonly").pack(side=tk.LEFT, padx=4)

    def _build_right_panel(self, parent: tk.Frame) -> None:
        fr = tk.LabelFrame(parent, text="Validation Errors", padx=4, pady=4)
        fr.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))
        self._error_text = tk.Text(fr, width=28, bg="#fff0f0", fg="#cc0000",
                                   state=tk.DISABLED, wrap=tk.WORD)
        self._error_text.pack(fill=tk.BOTH, expand=True)

    def _build_bottom_previews(self) -> None:
        pf = tk.Frame(self)
        pf.pack(fill=tk.BOTH, expand=True, pady=6)

        lf1 = tk.LabelFrame(pf, text="rclone Command Preview", padx=4, pady=4)
        lf1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self._cmd_preview = scrolledtext.ScrolledText(lf1, height=5, state=tk.DISABLED,
                                                      font=("Consolas", 9))
        self._cmd_preview.pack(fill=tk.BOTH, expand=True)

        lf2 = tk.LabelFrame(pf, text="Script Preview (first 80 lines)", padx=4, pady=4)
        lf2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._script_preview = scrolledtext.ScrolledText(lf2, height=5, state=tk.DISABLED,
                                                         font=("Consolas", 9))
        self._script_preview.pack(fill=tk.BOTH, expand=True)

    def _build_button_bar(self) -> None:
        bar = tk.Frame(self)
        bar.pack(fill=tk.X, pady=(4, 0))
        tk.Button(bar, text="Validate",         command=self.on_validate,
                  width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Generate",         command=self.on_generate,
                  width=12, bg="#d0e8d0").pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Open Output",      command=self.on_open_output,
                  width=14).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Copy Task Cmd",    command=self.on_copy_task_cmd,
                  width=16).pack(side=tk.LEFT, padx=2)

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _get_exclude_patterns(self) -> List[str]:
        raw = self._exclude_text.get("1.0", tk.END)
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.config(state=tk.DISABLED)

    def _set_errors(self, errors: List[str]) -> None:
        self._set_text(self._error_text, "\n".join(errors) if errors else "✓ No issues found.")
        self._error_text.config(bg="#fff0f0" if errors else "#f0fff0",
                                fg="#cc0000" if errors else "#008800")

    # --------------------------------------------------------------------------
    # Public state builder
    # --------------------------------------------------------------------------

    def build_job_config(self) -> JobConfig:
        return JobConfig(
            job_name         = self.job_name.get().strip(),
            source_folders   = [Path(p) for p in self._source_paths],
            remote_name      = self.remote_name.get().strip(),
            remote_path      = self.remote_path.get().strip(),
            output_folder    = Path(self.output_folder.get().strip() or "."),
            log_folder       = Path(self.log_folder.get().strip() or "logs"),
            backup_mode      = BackupMode(self.backup_mode.get()),
            retries          = self.retries.get(),
            low_level_retries= self.low_level_retries.get(),
            checkers         = self.checkers.get(),
            transfers        = self.transfers.get(),
            log_level        = LogLevel(self.log_level.get()),
            bwlimit          = self.bwlimit.get().strip(),
            checksum         = self.checksum.get(),
            post_copy_validation = self.post_copy_val.get(),
            excludes         = ExcludeConfig(
                patterns          = self._get_exclude_patterns(),
                exclude_from_file = Path(self.exclude_from_file.get().strip())
                                    if self.exclude_from_file.get().strip() else None,
            ),
            scheduler        = SchedulerConfig(
                frequency    = Frequency(self.frequency.get()),
                trigger_time = self.trigger_time.get().strip(),
                day_of_week  = self.day_of_week.get(),
            ),
        )

    def _populate_from_config(self, cfg: JobConfig) -> None:
        self.job_name.set(cfg.job_name)
        self._source_paths = [str(p) for p in cfg.source_folders]
        self._source_listbox.delete(0, tk.END)
        for p in self._source_paths:
            self._source_listbox.insert(tk.END, p)
        self.remote_name.set(cfg.remote_name)
        self.remote_path.set(cfg.remote_path)
        self.output_folder.set(str(cfg.output_folder))
        self.log_folder.set(str(cfg.log_folder))
        self.backup_mode.set(cfg.backup_mode.value)
        self.retries.set(cfg.retries)
        self.low_level_retries.set(cfg.low_level_retries)
        self.checkers.set(cfg.checkers)
        self.transfers.set(cfg.transfers)
        self.log_level.set(cfg.log_level.value)
        self.bwlimit.set(cfg.bwlimit)
        self.checksum.set(cfg.checksum)
        self.post_copy_val.set(cfg.post_copy_validation)
        self._exclude_text.delete("1.0", tk.END)
        self._exclude_text.insert("1.0", "\n".join(cfg.excludes.patterns))
        self.exclude_from_file.set(str(cfg.excludes.exclude_from_file or ""))
        self.frequency.set(cfg.scheduler.frequency.value)
        self.trigger_time.set(cfg.scheduler.trigger_time)
        self.day_of_week.set(cfg.scheduler.day_of_week)
        self.refresh_previews()

    # --------------------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------------------

    def add_source_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Source Folder")
        if folder and folder not in self._source_paths:
            self._source_paths.append(folder)
            self._source_listbox.insert(tk.END, folder)
            self.refresh_previews()

    def remove_source_folder(self) -> None:
        sel = self._source_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._source_paths.pop(idx)
        self._source_listbox.delete(idx)
        self.refresh_previews()

    def browse_output_folder(self) -> None:
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_folder.set(d)

    def browse_log_folder(self) -> None:
        d = filedialog.askdirectory(title="Select Log Folder")
        if d:
            self.log_folder.set(d)

    def browse_exclude_file(self) -> None:
        f = filedialog.askopenfilename(title="Select Exclude-From File")
        if f:
            self.exclude_from_file.set(f)

    def refresh_previews(self) -> None:
        try:
            cfg = self.build_job_config()
            self._set_text(self._cmd_preview, preview_rclone_command(cfg))
        except Exception as e:
            self._set_text(self._cmd_preview, f"(preview unavailable: {e})")

        try:
            from generator import render_script
            cfg = self.build_job_config()
            script = render_script(cfg)
            first_80 = "\n".join(script.splitlines()[:80])
            self._set_text(self._script_preview, first_80)
        except Exception as e:
            self._set_text(self._script_preview, f"(preview unavailable: {e})")

    def on_validate(self) -> None:
        cfg    = self.build_job_config()
        result = validate_job(cfg)
        self._set_errors(result.errors)
        if result.is_valid:
            messagebox.showinfo("Validation", "All checks passed.")

    def on_generate(self) -> None:
        cfg    = self.build_job_config()
        result = validate_job(cfg)
        self._set_errors(result.errors)
        if not result.is_valid:
            messagebox.showerror("Validation Failed",
                                 "Fix validation errors before generating:\n\n" +
                                 "\n".join(result.errors))
            return

        out = Path(cfg.output_folder)
        # Warn if any file already exists
        target = out / f"backup_{cfg.job_name}.py"
        if target.exists():
            if not messagebox.askyesno(
                "Overwrite?",
                f"Output '{target}' already exists.\nOverwrite?"
            ):
                return

        gen = generate_artifacts(cfg, overwrite=True)
        self._last_result = gen
        if gen.success:
            cp.save_ui_state(cfg.remote_name, str(cfg.output_folder), str(cfg.log_folder))
            messagebox.showinfo(
                "Generated",
                f"Files written to:\n{gen.output_folder}\n\n" +
                "\n".join(f.name for f in gen.files_written)
            )
        else:
            messagebox.showerror("Generation Errors", "\n".join(gen.errors))

    def on_open_output(self) -> None:
        folder = self.output_folder.get().strip() or "."
        try:
            os.startfile(folder)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")

    def on_copy_task_cmd(self) -> None:
        if not self._last_result or not self._last_result.files_written:
            messagebox.showwarning("Generate First", "Generate the job first.")
            return
        task_files = [f for f in self._last_result.files_written if f.suffix == ".txt"]
        if not task_files:
            messagebox.showwarning("No Task File", "No task command file found.")
            return
        content = task_files[0].read_text(encoding="utf-8")
        self.clipboard_clear()
        self.clipboard_append(content)
        messagebox.showinfo("Copied", "Task Scheduler commands copied to clipboard.")

    def on_load_job(self) -> None:
        name = self._job_combo.get()
        if not name:
            return
        cfg = cp.load_job(name)
        if cfg is None:
            messagebox.showerror("Not Found", f"Job '{name}' not found.")
            return
        self._populate_from_config(cfg)

    def on_save_job(self) -> None:
        cfg = self.build_job_config()
        if not cfg.job_name:
            messagebox.showwarning("Missing Name", "Enter a job name first.")
            return
        cp.save_job(cfg)
        self._load_saved_jobs_dropdown()
        messagebox.showinfo("Saved", f"Job '{cfg.job_name}' saved.")

    def on_delete_job(self) -> None:
        name = self._job_combo.get()
        if not name:
            return
        if messagebox.askyesno("Delete", f"Delete job '{name}'?"):
            cp.delete_job(name)
            self._load_saved_jobs_dropdown()

    def _load_saved_jobs_dropdown(self) -> None:
        jobs = cp.list_saved_jobs()
        self._job_combo["values"] = jobs
        if jobs:
            self._job_combo.set(jobs[-1])
