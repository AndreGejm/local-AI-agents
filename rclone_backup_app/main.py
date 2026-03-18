"""
main.py — Entry point for rclone Backup Configurator.
"""
from __future__ import annotations

import tkinter as tk
from gui import BackupConfigApp


def main() -> None:
    root = tk.Tk()
    root.title("rclone Backup Configurator")
    root.geometry("1100x680")
    root.minsize(900, 600)
    BackupConfigApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
