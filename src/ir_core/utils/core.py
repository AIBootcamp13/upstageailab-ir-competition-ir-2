"""IO helpers for JSONL and small utility functions.
Core utilities.
Moved from `ir_core.utils` to a submodule for clearer structure.
"""
import json


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, objs):
    with open(path, 'w', encoding='utf-8') as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
