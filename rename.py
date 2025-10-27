#!/usr/bin/env python3
"""
Rename every path (dirs + files) whose *name* contains 'graphglue' -> 'annnet'.
- Does NOT touch file contents.
- Skips common build/cache dirs and VCS metadata.
- Uses `git mv` when inside a Git repo (fallback: os.rename).
- Dry-run by default; use --apply to execute.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

OLD = "graphglue"
NEW = "annnet"

EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "build", "dist", "*.egg-info", ".eggs",
    ".ruff_cache", ".tox", ".nox",
    ".ipynb_checkpoints", ".DS_Store",
    ".venv", "venv", "env", ".env",
}

def should_prune(dirname: str) -> bool:
    # simple glob-ish check for '*.egg-info'
    if dirname.endswith(".egg-info"):
        return True
    return dirname in EXCLUDE_DIRS

def is_git_repo(root: Path) -> bool:
    return (root / ".git").is_dir() and shutil.which("git") is not None

def find_targets(root: Path, old: str) -> list[Path]:
    targets: list[Path] = []
    # Walk topdown so we can prune, but collect and later sort by depth desc.
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if not should_prune(d)]
        # collect dirs that need rename
        for d in dirnames:
            if old in d:
                targets.append(Path(dirpath) / d)
        # collect files that need rename
        for f in filenames:
            if old in f:
                targets.append(Path(dirpath) / f)
    # rename deepest paths first (more parts = deeper). If tie, files before dirs.
    def sort_key(p: Path):
        try:
            is_dir = p.is_dir()
        except Exception:
            is_dir = False
        return (-len(p.parts), 1 if is_dir else 0)
    targets.sort(key=sort_key)
    return targets

def plan_moves(targets: list[Path], old: str, new: str) -> list[tuple[Path, Path]]:
    moves = []
    for src in targets:
        dst = src.with_name(src.name.replace(old, new))
        if src == dst:
            continue
        moves.append((src, dst))
    # Also handle the top-level package dir if user forgot: graphglue/ -> annnet/
    # (Already covered above, but ensure parent doesn’t get missed if only root dir matches)
    return moves

def perform_moves(moves: list[tuple[Path, Path]], git: bool, apply: bool, force: bool) -> int:
    failures = 0
    for src, dst in moves:
        action = "RENAME"
        if not apply:
            print(f"[DRY] {action}: {src}  ->  {dst}")
            continue

        # ensure parent exists (should already), and handle existing dst
        if dst.exists():
            if not force:
                print(f"[SKIP] Destination exists (use --force to override): {dst}", file=sys.stderr)
                failures += 1
                continue
            # with git mv, -f will overwrite; with os.rename, remove file first if possible
            if not git:
                try:
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                except Exception as e:
                    print(f"[FAIL] Cannot remove existing destination: {dst} ({e})", file=sys.stderr)
                    failures += 1
                    continue

        if git:
            cmd = ["git", "mv"]
            if force:
                cmd.append("-f")
            cmd += [str(src), str(dst)]
            try:
                subprocess.run(cmd, check=True)
                print(f"[OK ] git mv: {src} -> {dst}")
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] git mv failed for {src} -> {dst}: {e}", file=sys.stderr)
                failures += 1
        else:
            try:
                os.rename(src, dst)
                print(f"[OK ] mv: {src} -> {dst}")
            except Exception as e:
                print(f"[FAIL] rename failed for {src} -> {dst}: {e}", file=sys.stderr)
                failures += 1
    return failures

def main():
    ap = argparse.ArgumentParser(description="Rename path names 'graphglue' -> 'annnet' across the repo.")
    ap.add_argument("--root", default=".", help="Root folder (default: current dir)")
    ap.add_argument("--from", dest="old", default=OLD, help="Old substring (default: graphglue)")
    ap.add_argument("--to", dest="new", default=NEW, help="New substring (default: annnet)")
    ap.add_argument("--apply", action="store_true", help="Actually perform changes (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="Overwrite if destination exists")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(2)

    old = args.old
    new = args.new
    if old == new:
        print("Old and new substrings are identical; nothing to do.", file=sys.stderr)
        sys.exit(0)

    targets = find_targets(root, old)

    moves = plan_moves(targets, old, new)

    if not moves:
        print("No paths to rename.")
        return

    print(f"Planned renames: {len(moves)}")
    git = is_git_repo(root)
    if git:
        print("Detected Git repo: will use `git mv` for atomic history.")
    else:
        print("No Git detected: will use filesystem renames.")

    failures = perform_moves(moves, git=git, apply=args.apply, force=args.force)

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to perform changes.")
    else:
        if failures:
            print(f"\nCompleted with {failures} failure(s). Review messages above.", file=sys.stderr)
            sys.exit(1)
        else:
            print("\nAll renames completed successfully.")

if __name__ == "__main__":
    main()
