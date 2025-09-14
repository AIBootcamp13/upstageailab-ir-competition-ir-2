#!/usr/bin/env python3
"""
Lightweight progress tracker to tick/untick checklist items in
`docs/notes/strategies/profiling-progress.md`.

Usage:
  poetry run python scripts/data/update_progress.py --item "Phase 1: vocab overlap matrix" --done 1
  poetry run python scripts/data/update_progress.py --item "Source glossary per src" --done 0

Matching is fuzzy: it finds the first checklist line containing the provided phrase.
"""
import argparse
import re
from pathlib import Path

PROGRESS_PATH = Path("docs/notes/strategies/profiling-progress.md")

CHECKBOX_RE = re.compile(r"^(\s*- \[)( |x)(\]\s*)(.*)$")


def update_item(text: str, phrase: str, done: bool) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = CHECKBOX_RE.match(line)
        if not m:
            continue
        label = m.group(4)
        if phrase.lower() in label.lower():
            new_mark = "x" if done else " "
            lines[i] = f"{m.group(1)}{new_mark}{m.group(3)}{label}"
            break
    return "\n".join(lines) + ("\n" if not text.endswith("\n") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True, help="Substring of the checklist item to toggle")
    ap.add_argument("--done", type=int, choices=[0, 1], required=True, help="1 to mark done, 0 to unmark")
    args = ap.parse_args()

    if not PROGRESS_PATH.exists():
        raise SystemExit(f"Progress file not found: {PROGRESS_PATH}")

    content = PROGRESS_PATH.read_text(encoding="utf-8")
    updated = update_item(content, args.item, bool(args.done))
    PROGRESS_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated: {args.item} -> {'done' if args.done else 'not done'}")


if __name__ == "__main__":
    main()
