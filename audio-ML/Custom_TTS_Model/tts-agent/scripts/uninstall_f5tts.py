"""
Remove all F5-TTS artifacts from this machine (macOS and Windows).

What this script removes
------------------------
  1. f5-tts pip package (from the current Python environment)
  2. HuggingFace Hub model cache for all known F5-TTS checkpoints
  3. The local models/f5tts directory (if present in this project)

Usage
-----
  python scripts/uninstall_f5tts.py
  python scripts/uninstall_f5tts.py --dry-run   # show what would be deleted
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── F5-TTS HuggingFace repo IDs → folder names in the HF cache ───────────────
# huggingface_hub names cache dirs as  models--{owner}--{repo}
F5_HF_CACHE_PREFIXES: list[str] = [
    "models--SWivid--F5-TTS",            # base English model
    "models--SPRINGLab--F5-Hindi-24KHz", # Hindi language-specific
    "models--Jmica--F5TTS",              # Japanese (5 GB .pt dumps)
]


def _hf_cache_root() -> Path:
    """Return the HuggingFace Hub cache directory for this platform."""
    if sys.platform == "win32":
        base = Path(os.environ.get("USERPROFILE", Path.home()))
    else:
        base = Path.home()
    return base / ".cache" / "huggingface" / "hub"


def main(dry_run: bool = False) -> None:
    prefix = "[DRY RUN] " if dry_run else ""

    # ── 1. Uninstall pip package ──────────────────────────────────────────────
    print(f"{prefix}Uninstalling f5-tts pip package…")
    cmd = [sys.executable, "-m", "pip", "uninstall", "f5-tts", "-y"]
    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ f5-tts uninstalled")
        else:
            # pip exits non-zero when package is not installed — that's fine
            msg = (result.stdout + result.stderr).strip()
            if "not installed" in msg.lower() or "WARNING" in msg:
                print("  ○ f5-tts was not installed — skipping")
            else:
                print(f"  ! pip uninstall: {msg}")

    # Also remove any dangling transitive deps that aren't shared with other packages
    for pkg in ("vocos", "pykakasi", "cached-path"):
        cmd2 = [sys.executable, "-m", "pip", "uninstall", pkg, "-y"]
        if dry_run:
            print(f"  Would run: {' '.join(cmd2)}")
        else:
            subprocess.run(cmd2, capture_output=True)

    # ── 2. HuggingFace cache ──────────────────────────────────────────────────
    hf_root = _hf_cache_root()
    print(f"\n{prefix}Scanning HuggingFace cache: {hf_root}")

    if not hf_root.exists():
        print("  HF cache directory does not exist — nothing to remove.")
    else:
        found_any = False
        for prefix_name in F5_HF_CACHE_PREFIXES:
            p = hf_root / prefix_name
            if p.exists():
                found_any = True
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
                if dry_run:
                    print(f"  Would delete: {p}  ({size_mb:.0f} MB)")
                else:
                    shutil.rmtree(p)
                    print(f"  ✓ Deleted: {p}  ({size_mb:.0f} MB)")
        if not found_any:
            print("  ○ No F5-TTS model cache found.")

    # ── 3. Local project model directory ─────────────────────────────────────
    project_root = Path(__file__).resolve().parents[1]
    local_f5 = project_root / "models" / "f5tts"
    print(f"\n{prefix}Checking local model dir: {local_f5}")
    if local_f5.exists() and any(local_f5.iterdir()):
        if dry_run:
            print(f"  Would delete: {local_f5}")
        else:
            shutil.rmtree(local_f5)
            local_f5.mkdir()          # recreate empty dir so git structure is intact
            print(f"  ✓ Deleted contents of {local_f5}")
    else:
        print("  ○ Local f5tts model dir is empty — nothing to remove.")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}F5-TTS removal complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove F5-TTS from this machine")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting anything.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
