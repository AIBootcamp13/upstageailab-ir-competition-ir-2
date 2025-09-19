"""
Scan data/documents.jsonl for duplicate docid values and report counts.
Optionally prints the first few duplicate docids and their counts.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Add src to path for imports
scripts_dir = Path(__file__).parent
repo_dir = scripts_dir.parent
src_dir = repo_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from ir_core.utils import get_current_documents_path
    DATA_PATH = Path(get_current_documents_path())
except ImportError:
    # Fallback if utils module is not available
    DATA_PATH = Path("data/documents_ko.jsonl")

if not DATA_PATH.exists():
    print(f"Data file not found: {DATA_PATH}")
    raise SystemExit(1)

counter = Counter()

with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        docid = obj.get("docid") or obj.get("doc_id") or obj.get("id")
        if docid:
            counter[docid] += 1

total = sum(counter.values())
unique = len(counter)
dups = {k: v for k, v in counter.items() if v > 1}

print(f"Total docs read: {total}")
print(f"Unique docids: {unique}")
print(f"Duplicate docids count: {len(dups)}")

if dups:
    print("Sample duplicates (up to 20):")
    for i, (k, v) in enumerate(sorted(dups.items(), key=lambda x: -x[1])):
        if i >= 20:
            break
        print(f"{k}: {v}")

print("Done.")
