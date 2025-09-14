#!/usr/bin/env python3
"""
Persist scientific terms into conf/scientific_terms.json for env-free usage.

By default, reads the latest artifact at
  outputs/reports/data_profile/latest/scientific_terms_extracted.json
and writes a curated, deduplicated list to conf/scientific_terms.json.

Usage examples:
  poetry run python scripts/data/persist_scientific_terms.py
  poetry run python scripts/data/persist_scientific_terms.py \
    --input outputs/reports/data_profile/20250914_120000/scientific_terms_extracted.json \
    --merge-base 1

If --merge-base=1, the base fallback terms from constants are included and deduped.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_INPUT = Path("outputs/reports/data_profile/latest/scientific_terms_extracted.json")
DEFAULT_OUTPUT = Path("conf/scientific_terms.json")


def load_terms(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input artifact not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    terms: List[str] = []
    if isinstance(data, list):
        terms = [str(t).strip() for t in data if str(t).strip()]
    elif isinstance(data, dict):
        # Accept {src: [terms]} or {"scientific_terms": [...]} formats
        if "scientific_terms" in data and isinstance(data["scientific_terms"], list):
            terms = [str(t).strip() for t in data["scientific_terms"] if str(t).strip()]
        else:
            for _src, lst in data.items():
                if isinstance(lst, list):
                    terms.extend([str(t).strip() for t in lst if str(t).strip()])
    # Deduplicate preserve order
    seen = set()
    out: List[str] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to extracted terms artifact (JSON)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write persistent JSON list")
    ap.add_argument("--merge-base", type=int, default=0, choices=[0, 1], help="Merge with base fallback terms from constants")
    args = ap.parse_args()

    terms = load_terms(args.input)

    if args.merge_base == 1:
        try:
            # Import lazily to avoid project import side effects if not needed
            from ir_core.analysis.constants import SCIENTIFIC_TERMS_BASE
            base_terms = list(SCIENTIFIC_TERMS_BASE)
        except Exception:
            base_terms = []
        merged: List[str] = []
        seen = set()
        for t in base_terms + terms:
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
        terms = merged

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(terms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(terms)} terms -> {args.output}")


if __name__ == "__main__":
    main()
