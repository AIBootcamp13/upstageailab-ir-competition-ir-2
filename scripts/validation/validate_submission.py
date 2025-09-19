#!/usr/bin/env python3
"""
Submission Format Validator

Validates submission files against the required format for the IR competition.
Checks for required fields, data types, and basic content validation.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_csv(file_path: str) -> List[Dict]:
    """Load CSV file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def validate_submission_format(data: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate submission format.

    Required fields:
    - eval_id: int
    - standalone_query: str
    - topk: list of str (document IDs)
    - answer: str
    - references: list of dict with 'score' (float) and 'content' (str)
    """
    errors = []
    required_fields = ['eval_id', 'standalone_query', 'topk', 'answer', 'references']

    if not data:
        return False, ["No data found in submission"]

    for i, record in enumerate(data):
        # Check required fields
        for field in required_fields:
            if field not in record:
                errors.append(f"Record {i}: Missing required field '{field}'")

        if any(field not in record for field in required_fields):
            continue  # Skip further validation for this record

        # Validate field types and content
        try:
            # eval_id should be int
            if not isinstance(record['eval_id'], int):
                if isinstance(record['eval_id'], str):
                    record['eval_id'] = int(record['eval_id'])
                else:
                    errors.append(f"Record {i}: eval_id should be int, got {type(record['eval_id'])}")

            # standalone_query should be str
            if not isinstance(record['standalone_query'], str):
                errors.append(f"Record {i}: standalone_query should be str, got {type(record['standalone_query'])}")

            # topk should be list of str
            if not isinstance(record['topk'], list):
                errors.append(f"Record {i}: topk should be list, got {type(record['topk'])}")
            else:
                for j, doc_id in enumerate(record['topk']):
                    if not isinstance(doc_id, str):
                        errors.append(f"Record {i}: topk[{j}] should be str, got {type(doc_id)}")

            # answer should be str
            if not isinstance(record['answer'], str):
                errors.append(f"Record {i}: answer should be str, got {type(record['answer'])}")

            # references should be list of dict
            if not isinstance(record['references'], list):
                errors.append(f"Record {i}: references should be list, got {type(record['references'])}")
            else:
                for j, ref in enumerate(record['references']):
                    if not isinstance(ref, dict):
                        errors.append(f"Record {i}: references[{j}] should be dict, got {type(ref)}")
                    else:
                        if 'score' not in ref:
                            errors.append(f"Record {i}: references[{j}] missing 'score' field")
                        elif not isinstance(ref['score'], (int, float)):
                            errors.append(f"Record {i}: references[{j}]['score'] should be numeric, got {type(ref['score'])}")

                        if 'content' not in ref:
                            errors.append(f"Record {i}: references[{j}] missing 'content' field")
                        elif not isinstance(ref['content'], str):
                            errors.append(f"Record {i}: references[{j}]['content'] should be str, got {type(ref['content'])}")

        except Exception as e:
            errors.append(f"Record {i}: Validation error: {e}")

    return len(errors) == 0, errors

def compare_submissions(sub1: List[Dict], sub2: List[Dict], name1: str, name2: str) -> None:
    """Compare two submissions."""
    print(f"\n🔍 Comparing {name1} vs {name2}")
    print("=" * 50)

    # Check lengths
    if len(sub1) != len(sub2):
        print(f"⚠️  Different number of records: {name1}={len(sub1)}, {name2}={len(sub2)}")

    # Check common eval_ids
    ids1 = {r['eval_id'] for r in sub1 if 'eval_id' in r}
    ids2 = {r['eval_id'] for r in sub2 if 'eval_id' in r}
    common = ids1 & ids2
    only1 = ids1 - ids2
    only2 = ids2 - ids1

    print(f"📊 Common eval_ids: {len(common)}")
    if only1:
        print(f"📊 Only in {name1}: {sorted(list(only1))[:10]}{'...' if len(only1) > 10 else ''}")
    if only2:
        print(f"📊 Only in {name2}: {sorted(list(only2))[:10]}{'...' if len(only2) > 10 else ''}")

    # Compare field presence
    fields1 = set()
    for r in sub1:
        fields1.update(r.keys())
    fields2 = set()
    for r in sub2:
        fields2.update(r.keys())

    print(f"📋 Fields in {name1}: {sorted(fields1)}")
    print(f"📋 Fields in {name2}: {sorted(fields2)}")
    print(f"📋 Missing in {name1}: {sorted(fields2 - fields1)}")
    print(f"📋 Extra in {name1}: {sorted(fields1 - fields2)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == '.jsonl':
            data = load_jsonl(file_path)
        elif file_ext == '.csv':
            data = load_csv(file_path)
        else:
            print(f"❌ Unsupported file format: {file_ext}")
            sys.exit(1)

        print(f"✅ Loaded {len(data)} records from {file_path}")

        # Validate format
        is_valid, errors = validate_submission_format(data)
        if is_valid:
            print("✅ Submission format is valid")
        else:
            print("❌ Submission format has errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"   {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")

        # Compare with sample if available
        sample_path = "outputs/sample_submission.jsonl"
        if Path(sample_path).exists():
            sample_data = load_csv(sample_path)
            compare_submissions(data, sample_data, Path(file_path).name, "sample_submission.jsonl")

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()