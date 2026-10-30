"""Check GitHub file and directory upload limits for this repository."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES_PER_DIRECTORY = 100


def main() -> None:
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    oversized = [(path, path.stat().st_size) for path in files if path.stat().st_size >= MAX_FILE_BYTES]
    counts = Counter(path.parent.relative_to(ROOT) for path in files)
    crowded = [(directory, count) for directory, count in counts.items() if count >= MAX_FILES_PER_DIRECTORY]
    print(f"files={len(files)}")
    print(f"largest={max(files, key=lambda path: path.stat().st_size).relative_to(ROOT)}")
    if oversized:
        raise SystemExit(f"oversized files: {oversized}")
    if crowded:
        raise SystemExit(f"directories with at least 100 files: {crowded}")
    print("GitHub size/count limits: OK")


if __name__ == "__main__":
    main()
