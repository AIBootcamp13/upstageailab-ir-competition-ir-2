#!/usr/bin/env python3
"""
Generate hypothetical questions/answers (HyDE-like) using the configured LLM and
`prompts/question_generation/question_generation_v1.jinja2` outside evaluation mode.

Usage examples:
  PYTHONPATH=src python scripts/tools/generate_hyde.py \
    --input data/eval.jsonl --output outputs/hypotheticals.jsonl --limit 50

  PYTHONPATH=src python scripts/tools/generate_hyde.py \
    --query "광합성 과정의 핵심 단계는?" --output outputs/hypotheticals.jsonl

Notes:
- The script looks for the last user message in an eval-style record (msg list) if --input is provided.
- Each input query yields one hypothetical question/answer text using the question_generation template.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

# Ensure src is importable when running directly
REPO_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ir_core.generation import get_generator
from omegaconf import OmegaConf


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def extract_query(item: Dict[str, Any]) -> str:
    # Try eval-style schema with messages
    msgs = item.get("msg")
    if isinstance(msgs, list) and msgs:
        users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        if users:
            return users[-1].get("content", "") or ""
    # Fallback single field
    return item.get("query", "")


def build_min_cfg() -> Any:
    """Build a minimal OmegaConf config to initialize the generator.
    Uses current settings for generator type and model name from environment via ir_core.config.settings
    but we only need generator + prompts here.
    """
    from ir_core.config import settings
    cfg_dict = {
        'pipeline': {
            'generator_type': getattr(settings, 'GENERATOR_TYPE', 'openai'),
            'generator_model_name': getattr(settings, 'GENERATOR_MODEL_NAME', 'gpt-4o-mini'),
        },
        'prompts': {
            'generation_qa': getattr(settings, 'GENERATOR_SYSTEM_MESSAGE_FILE', ''),
            'persona': getattr(settings, 'GENERATOR_SYSTEM_MESSAGE_FILE', ''),
        }
    }
    return OmegaConf.create(cfg_dict)


def generate_hypothetical(query: str, template_path: str, generator) -> str:
    # We pass the template path explicitly to make sure the question_generation template is used
    return generator.generate(query=query, context_docs=[], prompt_template_path=template_path)


def main():
    parser = argparse.ArgumentParser(description="Generate HyDE-like hypothetical content using question_generation template")
    parser.add_argument("--input", type=str, help="Input JSONL file (optional)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--query", type=str, help="Single ad-hoc query (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit when using --input")
    args = parser.parse_args()

    template_path = "prompts/question_generation/question_generation_v1.jinja2"

    # Initialize generator with minimal config
    cfg = build_min_cfg()
    generator = get_generator(cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        # Single query mode
        if args.query:
            hyp = generate_hypothetical(args.query, template_path, generator)
            fout.write(json.dumps({
                "query": args.query,
                "hypothetical": hyp
            }, ensure_ascii=False) + "\n")
            count += 1
        # Batch mode from file
        if args.input:
            for item in read_jsonl(Path(args.input)):
                q = extract_query(item)
                if not q:
                    continue
                hyp = generate_hypothetical(q, template_path, generator)
                rec = {
                    "eval_id": item.get("eval_id"),
                    "query": q,
                    "hypothetical": hyp
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                if args.limit and count >= args.limit:
                    break

    print(f"Wrote {count} hypothetical records to {out_path}")


if __name__ == "__main__":
    # Do not set RAG_EVALUATION_MODE here; this is a standalone utility
    main()
