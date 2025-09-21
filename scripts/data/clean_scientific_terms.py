#!/usr/bin/env python3
"""
Clean and normalize a list of scientific terms from a JSON or text file.

Features:
- Strips leading numbering (e.g., "12. ", "3)") and extra punctuation
- Splits on commas/colons/semicolons by default
- Optionally splits multi-word phrases into single-word tokens (default)
- Removes non-word characters (keeps Hangul, A-Z, a-z, digits for formulas like H2O)
- Deduplicates while preserving order (case-insensitive for Latin)
- Optional in-place update of the source file

Examples:
  # Clean the curated terms in-place
  uv run python scripts/data/clean_scientific_terms.py \
    --input conf/scientific_terms.json --in-place 1

  # Clean, keep multi-word phrases, and write to a new file
  uv run python scripts/data/clean_scientific_terms.py \
    --input conf/scientific_terms.json --output conf/scientific_terms.cleaned.json \
    --keep-multiword 1

Input formats:
- JSON array of strings (recommended)
- .txt file with one term per line (splitting will still apply)
"""
import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

# Keep common short scientific tokens even if len < 2 after cleaning
WHITELIST_SHORT = {"pH"}
WHITELIST_TOKENS = {"DNA", "RNA", "H2O", "CO2", "HIV", "ATP", "NADH", "NADPH"}

# Patterns
LEADING_NUMBER_RE = re.compile(r"^\s*\d+[\.)]?\s*")
# For single-word tokens, remove anything except Hangul, Latin letters, and digits
NON_WORD_INNER_RE = re.compile(r"[^A-Za-z0-9가-힣]")
# For phrases, convert disallowed chars to spaces and collapse
NON_WORD_TO_SPACE_RE = re.compile(r"[^A-Za-z0-9가-힣]+")

SPLIT_SEPARATORS = [",", ":", ";", "/", "·"]


def _load_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ".jsonl"}:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            # Fallback to line-based
            pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_pieces(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for s in items:
        # Remove leading numbering like "12. ", "3) "
        s = LEADING_NUMBER_RE.sub("", s)
        # Split on common separators
        pieces = [s]
        for sep in SPLIT_SEPARATORS:
            next_pieces: List[str] = []
            for p in pieces:
                next_pieces.extend([q for q in p.split(sep)])
            pieces = next_pieces
        out.extend([p.strip() for p in pieces if p.strip()])
    return out


def _tokenize(words: Iterable[str], keep_multiword: bool) -> List[str]:
    tokens: List[str] = []
    for w in words:
        if keep_multiword:
            # Convert non-word chunks to spaces and collapse
            cleaned = NON_WORD_TO_SPACE_RE.sub(" ", w).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cleaned:
                tokens.append(cleaned)
        else:
            # Split on whitespace into single tokens
            for t in re.split(r"\s+", w):
                t = t.strip()
                if not t:
                    continue
                # Remove inner non-word characters
                t = NON_WORD_INNER_RE.sub("", t)
                # If token is only digits, skip
                if t.isdigit():
                    continue
                # Keep short whitelisted tokens (e.g., pH)
                if len(t) < 2 and t not in WHITELIST_SHORT:
                    continue
                tokens.append(t)
    return tokens


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        # Normalize Latin case for dedupe; keep Hangul as-is
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def clean_terms(input_path: Path, keep_multiword: bool) -> List[str]:
    raw = _load_lines(input_path)
    pieces = _split_pieces(raw)
    tokens = _tokenize(pieces, keep_multiword=keep_multiword)
    cleaned = [t for t in tokens if t]
    return _dedupe_preserve_order(cleaned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Input JSON array or .txt file")
    ap.add_argument("--output", type=Path, default=None, help="Output JSON file (default: in-place if --in-place=1 else .cleaned.json next to input)")
    ap.add_argument("--keep-multiword", type=int, choices=[0, 1], default=0, help="Keep multi-word phrases (default 0 = single-word tokens)")
    ap.add_argument("--in-place", type=int, choices=[0, 1], default=0, help="Overwrite input file in-place (JSON)")
    args = ap.parse_args()

    terms = clean_terms(args.input, keep_multiword=bool(args.keep_multiword))

    # Decide output path
    if args.in_place:
        out_path = args.input
    else:
        if args.output is not None:
            out_path = args.output
        else:
            out_path = args.input.with_suffix(".cleaned.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(terms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(terms)} terms -> {out_path}")


if __name__ == "__main__":
    main()
