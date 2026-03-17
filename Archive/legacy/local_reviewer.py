import subprocess
import os
import sys

# Configuration
OLLAMA_BINARY = r"F:\Programs\Deepseek\ollama.exe"
MODEL_NAME = "qwen3-coder:30b"

def get_git_root():
    """Finds the root directory of the current git repository."""
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        return root
    except:
        return None

def get_git_diff():
    """Captures the current staged and unstaged git diff."""
    try:
        # Get staged changes
        staged = subprocess.check_output(["git", "diff", "--cached"], stderr=subprocess.STDOUT).decode("utf-8")
        # Get unstaged changes
        unstaged = subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode("utf-8")
        return staged + "\n" + unstaged
    except subprocess.CalledProcessError:
        return None

def run_local_review(diff_text, repo_root):
    """Sends the diff to the local Ollama instance for review."""
    if not diff_text.strip():
        print("No changes detected to review.")
        return

    prompt = f"""You are an expert senior software engineer and security auditor.
Your task is to review the following GIT DIFF and provide critical feedback.

Focus on:
1. BUG DETECTION: Identify logical errors, edge cases, or potential crashes.
2. SECURITY: Flag any secrets, hardcoded credentials, or vulnerable patterns (SQLi, XSS, etc).
3. PERFORMANCE: Suggest optimizations for speed or memory usage.
4. BEST PRACTICES: Comment on naming conventions, readability, and modularity.

Format your response as a professional markdown report. Start with a high-level summary.

GIT DIFF:
{diff_text}
"""

    print(f"--- Triggering Local Review ({MODEL_NAME}) ---")
    
    try:
        # We use 'ollama run' via subprocess to handle the interaction
        process = subprocess.Popen(
            [OLLAMA_BINARY, "run", MODEL_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        stdout, stderr = process.communicate(input=prompt)
        
        if process.returncode == 0:
            print("\n--- REVIEW COMPLETE ---\n")
            
            # Use specific output path in the repo root if possible
            output_path = os.path.join(repo_root if repo_root else ".", "latest_review.md")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(stdout)
            
            print(f"Review saved to: {output_path}")
            print("\nPreview of Summary:")
            # Print the first few lines of the output
            print("\n".join(stdout.splitlines()[:15]) + "...")
        else:
            print(f"Ollama Error: {stderr}")
            
    except Exception as e:
        print(f"Execution Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Verifying environment...")
        print(f"Ollama Path: {OLLAMA_BINARY}")
        if os.path.exists(OLLAMA_BINARY):
            print("[OK] Binary found.")
        else:
            print("[FAIL] Binary NOT found.")
    else:
        repo_root = get_git_root()
        if not repo_root:
            print("Error: This script must be run inside a Git repository.")
            sys.exit(1)
            
        diff = get_git_diff()
        if diff:
            run_local_review(diff, repo_root)
        else:
            print("No changes found to review.")