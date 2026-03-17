import os
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

"""
Folder Dump Tool
----------------
A production-grade utility to export folder structures and contents.
Features:
- GUI with fallback 'Select Folder' button.
- Robust binary file detection.
- Delimiter escaping to prevent output corruption.
- Memory-efficient processing.
- Handles permission errors and various encodings.

Standard Library only. No external dependencies.
"""

def is_binary_file(filepath):
    """
    Checks if a file is binary using a combination of null-byte detection 
    and common binary extension checks.
    """
    # Common binary extensions as a secondary heuristic
    binary_exts = {'.exe', '.dll', '.so', '.pyc', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.tar', '.gz'}
    if os.path.splitext(filepath)[1].lower() in binary_exts:
        return True
        
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(4096)
            if b'\0' in chunk:
                return True
            # Check for non-textual characters (heuristic)
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
            if any(byte not in text_chars for byte in chunk):
                return True
    except:
        return True # Default to binary if unreadable
    return False

def get_tree_structure(startpath):
    """Generates an ASCII tree structure of the folder."""
    tree = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f"{subindent}{f}")
    return "\n".join(tree)

def safe_write_content(f, content):
    """Writes content while escaping the delimiter to avoid format corruption."""
    # Simple escaping: if a line starts with '===', prefix it with a space or marker
    lines = content.splitlines()
    escaped_lines = []
    for line in lines:
        if line.startswith('==='):
            escaped_lines.append(' ' + line)
        else:
            escaped_lines.append(line)
    f.write("\n".join(escaped_lines))

def process_folder(folder_path):
    output_name = "folder_dump.txt"
    output_path = os.path.join(os.getcwd(), output_name)
    
    try:
        with open(output_path, 'w', encoding='utf-8', errors='replace') as out:
            # 1. Write Header
            out.write("=== FOLDER TREE ===\n")
            out.write(get_tree_structure(folder_path))
            out.write("\n\n")
            
            # 2. Iterate and write contents
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, folder_path)
                    
                    out.write(f"=== FILE: {rel_path} ===\n")
                    
                    if is_binary_file(full_path):
                        out.write("[BINARY FILE SKIPPED]\n")
                    else:
                        try:
                            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                                # Process in chunks to handle large files
                                while True:
                                    chunk = f.read(65536)
                                    if not chunk:
                                        break
                                    safe_write_content(out, chunk)
                        except Exception as e:
                            out.write(f"[ERROR READING FILE: {str(e)}]\n")
                            
                    out.write("\n=== END FILE ===\n\n")
                    
        return output_path
    except Exception as e:
        raise e

class FolderDumpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Folder Content Exporter")
        self.root.geometry("400x250")
        
        self.label = tk.Label(root, text="Folder Content Exporter", font=("Helvetica", 14, "bold"), pady=20)
        self.label.pack()
        
        self.info = tk.Label(root, text="Select a folder to generate folder_dump.txt", pady=10)
        self.info.pack()
        
        self.btn = tk.Button(root, text="Select Folder", command=self.select_folder, 
                             bg="#0078d7", fg="white", font=("Helvetica", 10, "bold"), 
                             padx=20, pady=10)
        self.btn.pack(pady=20)
        
        self.status = tk.Label(root, text="", fg="green")
        self.status.pack()

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            try:
                path = process_folder(folder)
                self.status.config(text=f"Success! Saved to folder_dump.txt", fg="green")
                messagebox.showinfo("Export Complete", f"File generated successfully at:\n{path}")
            except Exception as e:
                self.status.config(text="Error occurred", fg="red")
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main_root = tk.Tk()
    app = FolderDumpApp(main_root)
    main_root.mainloop()
